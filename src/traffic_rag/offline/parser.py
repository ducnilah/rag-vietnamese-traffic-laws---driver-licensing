import hashlib
import re
from pathlib import Path
from typing import List

from .models import SourceDocument

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _doc_id(path: Path, text: str) -> str:
    digest = hashlib.sha1((str(path) + text[:120]).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def parse_document(path: Path) -> SourceDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension '{suffix}' for {path}. "
            "M1 currently supports .txt/.md only."
        )

    raw = path.read_text(encoding="utf-8")
    cleaned = _clean_text(raw)
    return SourceDocument(
        doc_id=_doc_id(path, cleaned),
        title=path.stem,
        text=cleaned,
        source_path=str(path),
    )


def parse_directory(raw_dir: Path) -> List[SourceDocument]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    docs = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(parse_document(path))

    if not docs:
        raise RuntimeError(
            f"No supported documents found under {raw_dir}. "
            "Add .txt/.md files first."
        )
    return docs
