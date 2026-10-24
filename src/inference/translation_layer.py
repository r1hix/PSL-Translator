"""Bilingual Natural Language Generation and Translation Layer for PSL.

Translates discrete sign classification tokens into fluent communicative sentences
in English and Urdu, supporting real-time runtime language mode toggling.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, Optional

from src.config import Config, VocabularyItem, load_config

logger = logging.getLogger("TranslationLayer")


class LanguageMode(str, Enum):
    """Supported presentation language modes."""
    URDU = "urdu"
    ENGLISH = "english"
    BILINGUAL = "bilingual"


class SentenceStyle(str, Enum):
    """Output phrasing granularity."""
    COMMUNICATIVE = "communicative"  # e.g. "I need water, please." / "مجھے پانی چاہیے۔"
    LITERAL = "literal"              # e.g. "Water" / "پانی"


@dataclass
class TranslatedOutput:
    """Encapsulates bilingual translation payload."""
    word_name: str
    class_id: int
    confidence: float
    english_text: str
    urdu_text: str
    roman_text: str
    primary_display_text: str
    secondary_display_text: str


class BilingualTranslator:
    """Manages bilingual sentence synthesis and language switching for HUD & audio engines."""

    def __init__(
        self,
        config: Optional[Config] = None,
        default_mode: LanguageMode = LanguageMode.BILINGUAL,
        default_style: SentenceStyle = SentenceStyle.COMMUNICATIVE,
    ) -> None:
        """Initializes the translation engine with configuration metadata.
        
        Args:
            config: Optional pre-loaded Config object. Defaults to project config.
            default_mode: Initial active language mode.
            default_style: Phrasing style (communicative full sentence vs literal sign).
        """
        self.config = config or load_config()
        self.mode = default_mode
        self.style = default_style
        self._vocab_map: Dict[int, VocabularyItem] = {item.id: item for item in self.config.vocabulary}

    def toggle_language_mode(self) -> LanguageMode:
        """Cycles through language modes (Bilingual -> Urdu -> English -> Bilingual).
        
        Returns:
            LanguageMode: The newly activated language mode.
        """
        cycle = [LanguageMode.BILINGUAL, LanguageMode.URDU, LanguageMode.ENGLISH]
        current_idx = cycle.index(self.mode)
        self.mode = cycle[(current_idx + 1) % len(cycle)]
        logger.info("Switched Language Mode to: %s", self.mode.value)
        return self.mode

    def toggle_sentence_style(self) -> SentenceStyle:
        """Toggles between full communicative sentences and literal words.
        
        Returns:
            SentenceStyle: The newly activated sentence style.
        """
        self.style = (
            SentenceStyle.LITERAL
            if self.style == SentenceStyle.COMMUNICATIVE
            else SentenceStyle.COMMUNICATIVE
        )
        logger.info("Switched Sentence Style to: %s", self.style.value)
        return self.style

    def translate(self, class_id: int, confidence: float = 1.0) -> Optional[TranslatedOutput]:
        """Translates a predicted class ID into localized sentence output.
        
        Args:
            class_id: Predicted sign integer index.
            confidence: Model classification softmax confidence [0.0 - 1.0].
            
        Returns:
            Optional[TranslatedOutput]: Structured translation payload or None if invalid ID.
        """
        item = self._vocab_map.get(class_id)
        if not item:
            logger.warning("Unmapped class ID %d received in translation layer.", class_id)
            return None

        if self.style == SentenceStyle.COMMUNICATIVE:
            en = item.sentence_en
            ur = item.sentence_ur
            roman = item.sentence_roman
        else:
            en = item.roman
            ur = item.urdu
            roman = item.roman

        # Determine primary and secondary display text based on active mode
        if self.mode == LanguageMode.URDU:
            primary = ur
            secondary = roman
        elif self.mode == LanguageMode.ENGLISH:
            primary = en
            secondary = roman
        else:  # BILINGUAL
            primary = ur
            secondary = en

        return TranslatedOutput(
            word_name=item.name,
            class_id=item.id,
            confidence=confidence,
            english_text=en,
            urdu_text=ur,
            roman_text=roman,
            primary_display_text=primary,
            secondary_display_text=secondary,
        )
