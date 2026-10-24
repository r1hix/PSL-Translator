"""Web scraper and video downloader for Deaf Reach Pakistan Sign Language (PSL).

Fetches official 720p gesture video demonstrations directly from `psl.org.pk`
and maps them to organized class folders in `data/raw_videos/<word_name>/`.
"""

import argparse
import logging
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

from src.config import Config, load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PSLScraper")

BASE_URL = "https://psl.org.pk"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Curated direct paths for high-frequency words on psl.org.pk
DIRECT_WORD_MAP: Dict[str, str] = {
    "doctor": "/dictionary/42-health-medical-care/2311-doctor",
    "hospital": "/dictionary/42-health-medical-care/2347-hospital",
    "dawa": "/dictionary/42-health-medical-care/2374-medicine",
    "dard": "/dictionary/42-health-medical-care/2388-pain",
    "pani": "/dictionary/31-drinks/1580-mineral-water",
    "khana": "/dictionary/36-food-general/1792-food",
    "ghar": "/dictionary/20-buildings-places/1004-house",
    "madad": "/dictionary/72-verbs/4553-help",
    "theek": "/dictionary/65-sentences/7514-it-is-fine.",
    "shukriya": "/dictionary/65-sentences/4369-thank-you",
    "salam": "/dictionary/83-islamic-signs/7924-assalam-o-alaikum",
    "khuda_hafiz": "/dictionary/65-sentences/4122-good-bye",
    "haan": "/dictionary/65-sentences/4396-yes",
    "nahi": "/dictionary/65-sentences/4155-no",
    "aap_kaise_hain": "/dictionary/65-sentences/4130-how-are-you",

}


def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetches HTML content from a URL with custom user-agent.
    
    Args:
        url: Full web URL to request.
        timeout: Network timeout in seconds.
        
    Returns:
        Optional[str]: Decoded HTML string or None upon failure.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def extract_video_metadata(html: str) -> Optional[Tuple[str, str, str]]:
    """Extracts video source URL, English label, and Urdu text from a word page HTML.
    
    Args:
        html: Raw HTML content of a word detail page.
        
    Returns:
        Optional[Tuple[str, str, str]]: (video_url, english_word, urdu_text)
    """
    # 1. Video source URL (CloudFront CDN)
    video_match = re.search(r'<source\s+src=["\']([^"\']+\.mp4)["\']', html)
    if not video_match:
        # Fallback search for any mp4 link
        video_match = re.search(r'["\'](https://[^"\']+\.mp4)["\']', html)

    if not video_match:
        return None
    video_url = video_match.group(1)

    # 2. English label
    english_match = re.search(r'<h3[^>]*class=["\'][^"\']*text-gray-900[^"\']*["\'][^>]*>(.*?)</h3>', html)
    english_word = english_match.group(1).strip() if english_match else "unknown"

    # 3. Urdu script label
    urdu_match = re.search(r'<h3[^>]*class=["\'][^"\']*font-nastaliq[^"\']*["\'][^>]*>(.*?)</h3>', html)
    urdu_text = urdu_match.group(1).strip() if urdu_match else ""

    return video_url, english_word, urdu_text


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Downloads a binary file with retry support.
    
    Args:
        url: Source file URL.
        dest_path: Destination local Path.
        max_retries: Number of retry attempts upon network drop.
        
    Returns:
        bool: True if downloaded successfully.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        logger.info("File already exists: %s", dest_path.name)
        return True

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as resp, open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(16384)
                    if not chunk:
                        break
                    f.write(chunk)
            logger.info("Successfully downloaded: %s", dest_path.name)
            return True
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, url, exc)
            time.sleep(1.5)

    return False


def scrape_target_word(
    canonical_name: str,
    output_dir: Path,
    path_override: Optional[str] = None,
) -> bool:
    """Scrapes and downloads video for a specific target vocabulary word.
    
    Args:
        canonical_name: Name of word in config vocabulary (e.g. 'doctor', 'pani').
        output_dir: Root directory for raw videos (e.g. data/raw_videos).
        path_override: Optional relative word path on psl.org.pk.
        
    Returns:
        bool: True if successfully retrieved.
    """
    word_path = path_override or DIRECT_WORD_MAP.get(canonical_name)
    if not word_path:
        logger.warning("No known path for target word: '%s'", canonical_name)
        return False

    detail_url = f"{BASE_URL}{word_path}"
    logger.info("Scraping '%s' from %s", canonical_name, detail_url)
    html = fetch_url(detail_url)
    if not html:
        return False

    meta = extract_video_metadata(html)
    if not meta:
        logger.warning("No video source found on page for '%s'", canonical_name)
        return False

    video_url, english_word, urdu_text = meta
    logger.info("Found video: %s (English: %s, Urdu: %s)", video_url, english_word, urdu_text)

    # Destination: data/raw_videos/<canonical_name>/psl_portal_<slug>.mp4
    slug = Path(word_path).name
    save_file = output_dir / canonical_name / f"psl_portal_{slug}.mp4"
    return download_file(video_url, save_file)


def run_scraper(words: Optional[List[str]] = None, config_path: Optional[Path] = None) -> None:
    """Main execution orchestrator for PSL video acquisition.
    
    Args:
        words: List of specific vocabulary words to scrape, or None for all in config.
        config_path: Path to configuration YAML.
    """
    config: Config = load_config(config_path)
    raw_videos_dir = config.paths.raw_videos

    target_list = words if words else [item.name for item in config.vocabulary]
    logger.info("Starting PSL scraper for %d target words...", len(target_list))
    logger.info("Destination directory: %s", raw_videos_dir)

    success_count = 0
    for word in target_list:
        ok = scrape_target_word(word, raw_videos_dir)
        if ok:
            success_count += 1
        time.sleep(0.5)  # Respectful crawling rate limit

    logger.info("Scraping finished: %d/%d words retrieved successfully.", success_count, len(target_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official video demonstrations from psl.org.pk")
    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="Comma-separated list of target words (e.g. 'doctor,hospital,pani'). Default is all.",
    )
    args = parser.parse_args()

    selected_words = [w.strip() for w in args.words.split(",")] if args.words else None
    run_scraper(selected_words)
