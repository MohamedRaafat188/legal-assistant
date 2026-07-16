# -*- coding: utf-8 -*-
"""Ingestion layer: read Egyptian-law PDFs into clean, indexable Arabic text."""
from .arabic_text import (
    arabic_digits_to_ascii,
    clean_text,
    normalize_for_embedding,
    repair_pua,
    strip_directional_controls,
)
from .arabic_text import normalize_for_match
from .pdf_extractor import (
    PageText,
    PdfTextExtractor,
    PdftotextNotFoundError,
)
from .structure import (
    ArticleIntegrity,
    ArticleMarker,
    DetectedStructure,
    Level,
    StructuralMarker,
    detect_structure,
)

__all__ = [
    # extraction
    "PageText",
    "PdfTextExtractor",
    "PdftotextNotFoundError",
    # normalization
    "clean_text",
    "normalize_for_embedding",
    "normalize_for_match",
    "repair_pua",
    "strip_directional_controls",
    "arabic_digits_to_ascii",
    # structure
    "Level",
    "StructuralMarker",
    "ArticleMarker",
    "DetectedStructure",
    "ArticleIntegrity",
    "detect_structure",
]
