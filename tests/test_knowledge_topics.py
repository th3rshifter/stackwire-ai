from pathlib import Path

from app.config import ROOT_DIR
from app.rag import validate_knowledge


def test_all_knowledge_markdown_files_are_nonempty_and_retrievable() -> None:
    knowledge_dir = ROOT_DIR / "docs" / "knowledge"
    markdown_files = sorted(knowledge_dir.glob("*.md"))

    assert markdown_files
    assert not validate_knowledge(knowledge_dir)
    assert all(path.read_text(encoding="utf-8").strip() for path in markdown_files)
