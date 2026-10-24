"""Data acquisition and collection module for Pakistani Sign Language.

Includes scrapers for official psl.org.pk video resources, interactive webcam
sample recorders, and data integrity verification utilities.
"""

# Avoid eager submodule imports when running scripts directly via `python -m`
__all__ = [
    "psl_scraper",
    "record_samples",
    "verify_data",
]
