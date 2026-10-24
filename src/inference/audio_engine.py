"""Audio synthesis and asynchronous spoken Urdu audio playback engine for PSL.

Academic Explainability:
    To maintain the target real-time processing rate (>= 30 FPS) without jitter,
    network-bound text-to-speech (TTS) calls must NEVER occur inside the OpenCV
    frame processing loop. This module:
    1. Pre-caches natural Urdu pronunciation MP3s into `data/audio_cache/`.
    2. Provides an asynchronous, non-blocking playback mechanism with software debouncing.
    3. Supports multiple backend audio drivers: pygame.mixer, macOS `afplay`, and Windows media player.
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import Dict, Optional

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    gTTS = None
    GTTS_AVAILABLE = False

try:
    import pygame
    import pygame.mixer
    PYGAME_MIXER_AVAILABLE = True
except (ImportError, NotImplementedError):
    pygame = None
    PYGAME_MIXER_AVAILABLE = False

from src.config import Config, VocabularyItem, load_config

logger = logging.getLogger("AudioEngine")


class AudioEngine:
    """Manages pre-cached Urdu spoken audio generation and asynchronous playback."""

    def __init__(
        self,
        config: Optional[Config] = None,
        cooldown_sec: float = 2.5,
        enabled: bool = True,
    ) -> None:
        """Initializes audio playback engine.
        
        Args:
            config: Optional pre-loaded Config instance.
            cooldown_sec: Minimum seconds between repeated triggers of the same word.
            enabled: Master switch for audio playback.
        """
        self.config = config or load_config()
        self.cache_dir = self.config.paths.audio_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_sec = cooldown_sec
        self.enabled = enabled and self.config.inference.audio_enabled

        self.last_played_word: Optional[str] = None
        self.last_played_time: float = 0.0
        self._playback_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        # Initialize pygame mixer if supported
        self._use_pygame = False
        if PYGAME_MIXER_AVAILABLE:
            try:
                pygame.mixer.init()
                self._use_pygame = True
                logger.info("Initialized Pygame audio mixer driver.")
            except Exception as e:
                logger.warning("Could not initialize pygame mixer: %s. Falling back to native system player.", e)

    def get_audio_path(self, word_name: str, communicative: bool = True) -> Path:
        """Returns the local path for a vocabulary item's cached MP3 file.
        
        Args:
            word_name: Canonical vocabulary word identifier (e.g. 'pani').
            communicative: If True, full conversational sentence; else literal sign.
            
        Returns:
            Path: Resolved path inside data/audio_cache/.
        """
        suffix = "sentence" if communicative else "literal"
        return self.cache_dir / f"{word_name}_{suffix}.mp3"

    def precache_vocabulary(self, force: bool = False) -> Dict[str, bool]:
        """Pre-renders spoken Urdu audio files for all target vocabulary signs using gTTS.
        
        Args:
            force: If True, overwrites existing audio cache files.
            
        Returns:
            Dict[str, bool]: Mapping of word name to generation success status.
        """
        if not GTTS_AVAILABLE:
            logger.error("gTTS is not installed. Run: pip install gTTS")
            return {}

        results: Dict[str, bool] = {}
        logger.info("Pre-caching Urdu spoken audio files in %s...", self.cache_dir)

        for item in self.config.vocabulary:
            # 1. Communicative full sentence (e.g. "مجھے پانی چاہیے۔")
            sentence_path = self.get_audio_path(item.name, communicative=True)
            sent_text = item.sentence_ur or item.urdu

            # 2. Literal sign word (e.g. "پانی")
            literal_path = self.get_audio_path(item.name, communicative=False)
            lit_text = item.urdu

            for path, text in [(sentence_path, sent_text), (literal_path, lit_text)]:
                if path.exists() and not force:
                    results[path.stem] = True
                    continue

                try:
                    tts = gTTS(text=text, lang="ur")
                    tts.save(str(path))
                    logger.info("Cached audio for [%s]: %s (%d KB)", item.name, path.name, path.stat().st_size // 1024)
                    results[path.stem] = True
                except Exception as exc:
                    logger.warning("Failed to render audio for [%s] ('%s'): %s", item.name, text, exc)
                    results[path.stem] = False

        return results

    def play_word(self, word_name: str, communicative: bool = True, async_mode: bool = True) -> bool:
        """Asynchronously plays the pre-cached Urdu pronunciation for a given sign.
        
        Mathematical / Temporal Debouncing:
            Prevents audio overlapping and rapid acoustic chatter by checking:
            elapsed_time = current_time - last_played_time
            if (word == last_word and elapsed_time < cooldown_sec):
                return False
                
        Args:
            word_name: Canonical vocabulary sign name.
            communicative: Whether to play communicative sentence vs literal word.
            async_mode: Whether playback executes in background without blocking caller.
            
        Returns:
            bool: True if playback was triggered.
        """
        if not self.enabled:
            return False

        current_time = time.time()
        with self._lock:
            # Debounce repeated triggers of the same word
            if self.last_played_word == word_name and (current_time - self.last_played_time) < self.cooldown_sec:
                return False

            audio_path = self.get_audio_path(word_name, communicative=communicative)
            # Fallback to literal if sentence doesn't exist, or vice versa
            if not audio_path.exists():
                fallback_path = self.get_audio_path(word_name, communicative=not communicative)
                if fallback_path.exists():
                    audio_path = fallback_path
                else:
                    logger.warning("Audio cache miss for '%s'. File not found: %s", word_name, audio_path.name)
                    return False

            self.last_played_word = word_name
            self.last_played_time = current_time

        if async_mode:
            thread = threading.Thread(target=self._play_file_blocking, args=(audio_path,), daemon=True)
            thread.start()
        else:
            self._play_file_blocking(audio_path)

        return True

    def _play_file_blocking(self, file_path: Path) -> None:
        """Platform-agnostic playback executor."""
        current_os = platform.system()

        # Driver 1: Pygame Mixer
        if self._use_pygame:
            try:
                pygame.mixer.music.load(str(file_path))
                pygame.mixer.music.play()
                return
            except Exception as e:
                logger.debug("Pygame playback error: %s. Trying OS player.", e)

        # Driver 2: macOS native `afplay`
        if current_os == "Darwin":
            try:
                if self._playback_process and self._playback_process.poll() is None:
                    self._playback_process.terminate()
                self._playback_process = subprocess.Popen(
                    ["/usr/bin/afplay", str(file_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._playback_process.wait()
                return
            except Exception as e:
                logger.warning("macOS afplay error: %s", e)

        # Driver 3: Windows PowerShell media player
        elif current_os == "Windows":
            try:
                # PowerShell media player non-blocking script
                ps_cmd = f"(New-Object Media.SoundPlayer '{file_path}').PlaySync()"
                subprocess.run(["powershell", "-c", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception as e:
                logger.warning("Windows audio playback error: %s", e)

    def stop(self) -> None:
        """Stops any currently playing audio stream."""
        if self._use_pygame:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

        if self._playback_process and self._playback_process.poll() is None:
            try:
                self._playback_process.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Pre-cache and test PSL spoken Urdu audio files.")
    parser.add_argument("--precache", action="store_true", help="Pre-render all vocabulary Urdu MP3s into data/audio_cache/")
    parser.add_argument("--play", type=str, default=None, help="Test playback of a specific word (e.g. 'salam', 'pani')")
    parser.add_argument("--force", action="store_true", help="Force regenerate audio even if cached file exists.")
    args = parser.parse_args()

    engine = AudioEngine()
    if args.precache:
        engine.precache_vocabulary(force=args.force)

    if args.play:
        print(f"Playing word: {args.play}...")
        engine.play_word(args.play, communicative=True, async_mode=False)
