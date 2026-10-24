"""Web scraper and video downloader for Deaf Reach Pakistan Sign Language (PSL).

Fetches official 720p gesture video demonstrations directly from `psl.org.pk`
and maps them to organized class folders in `data/raw_videos/<word_name>/`.
"""

import argparse
import json
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
API_BASE_URL = "https://admin.psl.org.pk/api"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Curated and verified paths for high-frequency vocabulary on psl.org.pk
# Note: psl.org.pk routes exclusively by numeric concept ID: /dictionary/<cat_id>-<cat_slug>/<concept_id>-<concept_slug>
DIRECT_WORD_MAP: Dict[str, str] = {
    # 1. Greetings
    "salam": "/dictionary/65-sentences/4268-assalam-o-alaikum",       # ID 4268 -> Assalam-o-Alaikum
    "shukriya": "/dictionary/65-sentences/4369-thank-you",           # ID 4369 -> thank you
    "aap_kaise_hain": "/dictionary/65-sentences/4310-how-are-you",   # ID 4310 -> how are you?
    "theek": "/dictionary/65-sentences/7514-it-is-fine.",            # ID 7514 -> It is fine.
    "khuda_hafiz": "/dictionary/65-sentences/4305-goodbye",          # ID 4305 -> goodbye

    # 2. Essentials
    "pani": "/dictionary/31-drinks/7070-water",                      # ID 7070 -> Water
    "khana": "/dictionary/72-verbs/8070-eat",                        # ID 8070 -> Eat (کهانا)
    "ghar": "/dictionary/10-around-the-house/8060-house",            # ID 8060 -> House (گھر)
    "madad": "/dictionary/72-verbs/5071-help",                       # ID 5071 -> help (مدد)
    "haan": "/dictionary/65-sentences/4396-yes",                     # ID 4396 -> yes (جی ہاں)
    "nahi": "/dictionary/2-adverbs/8096-never",                      # ID 8096 -> never (کبھی نہیں)

    # 3. Emergency / Medical
    "doctor": "/dictionary/62-professions/3957-doctor",              # ID 3957 -> doctor (ڈاکٹر)
    "hospital": "/dictionary/42-health-medical-care/2371-hospital",   # ID 2371 -> hospital (ہسپتال)
    "dawa": "/dictionary/42-health-medical-care/8074-medicine",      # ID 8074 -> Medicine (دوائی)
    "dard": "/dictionary/42-health-medical-care/2456-sick",          # ID 2456 -> sick (بیمار)
}


def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetches HTML or text content from a URL with custom user-agent.
    
    Args:
        url: Full web URL to request.
        timeout: Network timeout in seconds.
        
    Returns:
        Optional[str]: Decoded string or None upon failure.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def fetch_video_via_api(concept_id: int) -> Optional[Tuple[str, str, str]]:
    """Directly fetches video URL and metadata from the official PSL backend API.
    
    Args:
        concept_id: Integer database concept ID (e.g. 8060 for house).
        
    Returns:
        Optional[Tuple[str, str, str]]: (video_url, english_word, urdu_text)
    """
    api_url = f"{API_BASE_URL}/concept/{concept_id}/videos"
    content = fetch_url(api_url)
    if not content:
        return None
        
    try:
        data = json.loads(content).get("data", {})
        concept = data.get("concept", {})
        videos = data.get("videos", [])
        if not videos:
            return None
            
        video_url = videos[0].get("video_url")
        english_word = concept.get("title", "unknown")
        urdu_text = concept.get("title_secondary", "")
        if video_url:
            return video_url, english_word, urdu_text
    except Exception as exc:
        logger.warning("Error parsing API response for concept ID %d: %s", concept_id, exc)
        
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


def download_file(url: str, dest_path: Path, max_retries: int = 3, force: bool = False) -> bool:
    """Downloads a binary file with retry support.
    
    Args:
        url: Source file URL.
        dest_path: Destination local Path.
        max_retries: Number of retry attempts upon network drop.
        force: If True, redownload even if destination file already exists.
        
    Returns:
        bool: True if downloaded successfully.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and dest_path.exists() and dest_path.stat().st_size > 1024:
        logger.info("File already exists: %s (use --force to overwrite)", dest_path.name)
        return True

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=25) as resp, open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(16384)
                    if not chunk:
                        break
                    f.write(chunk)
            logger.info("Successfully downloaded: %s (%d KB)", dest_path.name, dest_path.stat().st_size // 1024)
            return True
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, url, exc)
            time.sleep(1.5)

    return False


def scrape_target_word(
    canonical_name: str,
    output_dir: Path,
    path_override: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Scrapes and downloads video for a specific target vocabulary word.
    
    Args:
        canonical_name: Name of word in config vocabulary (e.g. 'doctor', 'pani').
        output_dir: Root directory for raw videos (e.g. data/raw_videos).
        path_override: Optional relative word path on psl.org.pk.
        force: If True, redownload even if destination file already exists.
        
    Returns:
        bool: True if successfully retrieved.
    """
    word_path = path_override or DIRECT_WORD_MAP.get(canonical_name)
    if not word_path:
        logger.warning("No known path for target word: '%s'", canonical_name)
        return False

    meta = None
    # 1. Try extracting numeric concept ID for direct API retrieval (last segment)
    id_match = re.search(r'/(\d+)-[^/]*$', word_path)
    if id_match:
        concept_id = int(id_match.group(1))
        meta = fetch_video_via_api(concept_id)

    # 2. Fallback to HTML scraping if API returned no video
    if not meta:
        detail_url = f"{BASE_URL}{word_path}"
        logger.info("Fetching HTML for '%s' from %s", canonical_name, detail_url)
        html = fetch_url(detail_url)
        if html:
            meta = extract_video_metadata(html)

    if not meta:
        logger.warning("No video source found on portal for '%s' (%s)", canonical_name, word_path)
        return False

    video_url, english_word, urdu_text = meta
    logger.info("Target: '%s' -> Title: '%s' | Urdu: '%s'", canonical_name, english_word, urdu_text)
    logger.info("Video URL: %s", video_url)

    # Destination: data/raw_videos/<canonical_name>/psl_portal_<slug>.mp4
    slug = Path(word_path).name
    save_file = output_dir / canonical_name / f"psl_portal_{slug}.mp4"
    return download_file(video_url, save_file, force=force)


def run_scraper(
    words: Optional[List[str]] = None,
    config_path: Optional[Path] = None,
    force: bool = False,
) -> None:
    """Main execution orchestrator for PSL video acquisition.
    
    Args:
        words: List of specific vocabulary words to scrape, or None for all in config.
        config_path: Path to configuration YAML.
        force: If True, overwrite existing video files.
    """
    config: Config = load_config(config_path)
    raw_videos_dir = config.paths.raw_videos

    target_list = words if words else [item.name for item in config.vocabulary]
    logger.info("Starting PSL scraper for %d target words...", len(target_list))
    logger.info("Destination directory: %s", raw_videos_dir)

    success_count = 0
    for word in target_list:
        ok = scrape_target_word(word, raw_videos_dir, force=force)
        if ok:
            success_count += 1
        time.sleep(0.3)  # Respectful crawling rate limit

    logger.info("Scraping finished: %d/%d words retrieved successfully.", success_count, len(target_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official video demonstrations from psl.org.pk")
    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="Comma-separated list of target words (e.g. 'doctor,hospital,pani'). Default is all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload and overwrite existing video files.",
    )
    args = parser.parse_args()

    selected_words = [w.strip() for w in args.words.split(",")] if args.words else None
    run_scraper(selected_words, force=args.force)
