from pathlib import Path

import pytest

import tools
from tools import MAX_NOTE_LENGTH, write_note


@pytest.fixture
def notes_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the notes folder to a temporary folder that does not exist yet."""
    directory = tmp_path / "notes"
    monkeypatch.setattr(tools, "NOTES_DIR", directory)
    return directory


def test_requires_confirmation() -> None:
    assert write_note.requires_confirmation


def test_creates_the_notes_folder_and_the_note(notes_dir: Path) -> None:
    assert not notes_dir.exists()

    path = write_note.run(title="Meeting", content="Thursday at 10")

    assert Path(path) == notes_dir / "Meeting.md"
    assert (notes_dir / "Meeting.md").read_text(encoding="utf-8") == "Thursday at 10"


def test_keeps_accents_in_the_title_and_the_content(notes_dir: Path) -> None:
    write_note.run(title="Réunion à 10h", content="Préparer l'ordre du jour ✓")

    note = notes_dir / "Réunion à 10h.md"
    assert note.read_text(encoding="utf-8") == "Préparer l'ordre du jour ✓"


def test_never_overwrites_an_existing_note(notes_dir: Path) -> None:
    write_note.run(title="Meeting", content="first")

    with pytest.raises(ValueError, match="'Meeting' already exists"):
        write_note.run(title="Meeting", content="second")

    assert (notes_dir / "Meeting.md").read_text(encoding="utf-8") == "first"


@pytest.mark.parametrize(
    "title",
    [
        "Meeting",
        "todo_2026-11",
        "a" * 50,
    ],
)
def test_accepts_valid_titles(notes_dir: Path, title: str) -> None:
    write_note.run(title=title, content="ok")

    assert (notes_dir / f"{title}.md").exists()


@pytest.mark.parametrize(
    "title",
    [
        "../outside",
        "../../.bashrc",
        "/etc/passwd",
        "sub/folder",
        "back\\slash",
        "with.dot",
        "line\nbreak",
        "",
        "a" * 51,
    ],
)
def test_rejects_invalid_titles(notes_dir: Path, title: str) -> None:
    with pytest.raises(ValueError, match="Invalid title"):
        write_note.run(title=title, content="x")

    assert not notes_dir.exists()


def test_accepts_the_longest_allowed_content(notes_dir: Path) -> None:
    write_note.run(title="Long", content="x" * MAX_NOTE_LENGTH)

    assert (notes_dir / "Long.md").stat().st_size == MAX_NOTE_LENGTH


def test_rejects_a_too_long_content(notes_dir: Path) -> None:
    with pytest.raises(ValueError, match="too long"):
        write_note.run(title="Long", content="x" * (MAX_NOTE_LENGTH + 1))

    assert not notes_dir.exists()
