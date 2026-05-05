from dataclasses import dataclass, field
from statistics import mean

from app.models.chunk import Chunk
from app.models.document import Document


@dataclass
class IngestionDiagnostics:
    file_name: str
    file_type: str
    file_size: int
    character_count: int
    word_count: int
    chunk_count: int
    average_chunk_length: float
    readiness_status: str
    readiness_score: int
    recommended_chunking_strategy: str
    recommendation_reason: str
    warnings: list[str] = field(default_factory=list)
    page_count: int | None = None


def _looks_like_heading(line: str) -> bool:
    """Return True when a line looks like a structural heading."""
    stripped = line.strip()

    if not stripped:
        return False

    if len(stripped) > 120:
        return False

    # Avoid treating discussion questions as headings.
    if stripped.endswith("?"):
        return False

    # Avoid long sentence-like lines.
    if len(stripped.split()) > 12:
        return False

    return (
        stripped.isupper()
        or stripped.endswith(":")
        or stripped.startswith(("1.", "2.", "3.", "4.", "5."))
    )


def run_ingestion_diagnostics(
    doc: Document,
    chunks: list[Chunk],
) -> IngestionDiagnostics:
    """
    Run lightweight diagnostics after extraction and chunking.

    This does not modify the source document.
    It provides retrieval-readiness and chunking guidance signals for the UI.
    """

    raw_text = doc.raw_text or ""
    file_name = doc.metadata.get("file_name", "")
    file_size = doc.metadata.get("file_size", 0)
    file_type = doc.file_type

    character_count = len(raw_text)
    word_count = len(raw_text.split())
    chunk_count = len(chunks)
    chunk_lengths = [len(chunk.text or "") for chunk in chunks]
    average_chunk_length = mean(chunk_lengths) if chunk_lengths else 0

    pages = doc.metadata.get("pages")
    page_count = len(pages) if isinstance(pages, list) else None

    warnings: list[str] = []

    if character_count < 500:
        warnings.append(
            "Very little text was extracted. This document may be scanned, image-based, or too short for reliable retrieval."
        )

    if word_count < 100:
        warnings.append(
            "Low word count detected. Answers may be weak because there is limited searchable content."
        )

    if chunk_count == 0:
        warnings.append(
            "No chunks were created. This document cannot be searched."
        )

    if chunk_count == 1 and character_count > 2500:
        warnings.append(
            "Only one chunk was created from a relatively large document. Consider smaller chunks or a different chunking strategy."
        )

    if average_chunk_length < 300:
        warnings.append(
            "Average chunk length is quite small. Retrieval may return fragmented context."
        )

    if average_chunk_length > 2500:
        warnings.append(
            "Average chunk length is quite large. Retrieval may return overly broad context."
        )

    if file_type == "pdf":
        if not page_count:
            warnings.append(
                "No page-level text was extracted from this PDF. It may require OCR."
            )
        elif character_count / max(page_count, 1) < 300:
            warnings.append(
                "Low text density per PDF page detected. This may indicate a scanned PDF, poor extraction quality, or image-heavy document."
            )

    heading_like_lines = [
        line.strip()
        for line in raw_text.splitlines()
        if _looks_like_heading(line)
    ]

    if len(heading_like_lines) < 2 and character_count > 1500:
        warnings.append(
            "Few obvious headings were detected. Section-aware retrieval may be weaker unless the document has clearer structure."
        )

    readiness_score = max(0, 100 - (15 * len(warnings)))

    if readiness_score >= 90:
        readiness_status = "Good"
    elif readiness_score >= 70:
        readiness_status = "Needs Review"
    else:
        readiness_status = "Poor"

    recommended_chunking_strategy = "character"
    recommendation_reason = (
        "No strong page, heading, or delimiter structure was detected, so character chunking is the safest general-purpose option."
    )

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    line_counts: dict[str, int] = {}

    for line in lines:
        if 0 < len(line) < 60:
            line_counts[line] = line_counts.get(line, 0) + 1

    delimiter_candidates = [
        line
        for line, count in line_counts.items()
        if count >= 3 and count <= max(2, len(lines) // 4)
    ]

    # Heading structure must be strong before it beats PDF page chunking.
    if len(heading_like_lines) >= 10:
        recommended_chunking_strategy = "heading"
        recommendation_reason = (
            "Strong heading structure was detected, suggesting the document may benefit from section-aware chunking."
        )

    elif file_type == "pdf" and page_count and page_count >= 2:
        recommended_chunking_strategy = "page"
        recommendation_reason = (
            "This PDF has multiple extracted pages and no very strong heading structure, so page chunking may preserve useful page-level context and citation traceability."
        )

    elif delimiter_candidates and character_count > 1000:
        recommended_chunking_strategy = "delimiter"
        recommendation_reason = (
            "Repeated short lines were detected, which may indicate a recurring delimiter or repeated section marker."
        )

    return IngestionDiagnostics(
        file_name=file_name,
        file_type=file_type,
        file_size=file_size,
        character_count=character_count,
        word_count=word_count,
        chunk_count=chunk_count,
        average_chunk_length=round(average_chunk_length, 1),
        readiness_status=readiness_status,
        readiness_score=readiness_score,
        recommended_chunking_strategy=recommended_chunking_strategy,
        recommendation_reason=recommendation_reason,
        warnings=warnings,
        page_count=page_count,
    )