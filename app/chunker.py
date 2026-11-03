"""Loads a text document and splits it into section-aware chunks.

How it works:
  1. Find the document title (the first line).
  2. Find section headings, for example:
       "# Revenue" or "## Details"       (markdown)
       "2. Revenue" or "2.1 Details"     (numbered)
       "RISKS"                           (all capitals)
       "Revenue Breakdown"               (short line on its own, no full stop)
  3. Cut the text into sections at those headings, remembering the
     heading path, e.g. "2. Revenue > 2.1 Details".
  4. A small section becomes one chunk. A big section is split further
     with LangChain's recursive splitter (paragraph > line > sentence > word).
  5. Every chunk starts with a header line saying which document and
     section it belongs to, so the chunk makes sense on its own.

So a chunk never mixes two sections, and never loses its section name.
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.errors import PreCueError

CHUNK_SIZE = 800  # max characters of section text per chunk (about 130-150 words)
CHUNK_OVERLAP = 100  # characters shared between chunks when a big section is split
MAX_HEADING_LENGTH = 80
SUPPORTED_FILES = {".txt", ".md"}

NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")  # "2. Revenue", "2.1 Details"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


def load_text(path):
    """Read a .txt or .md file and return its text."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_FILES:
        raise PreCueError(f"'{path.name}' is not supported. Use a .txt or .md file.")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PreCueError(f"Could not read '{path}': {error}") from error
    if not text.strip():
        raise PreCueError(f"'{path.name}' is empty.")
    return text.replace("\r\n", "\n")  # Windows line endings -> normal


def heading_level(line, stands_alone):
    """Return the heading level (1 = top) if the line is a heading, otherwise None.

    stands_alone = the line has a blank line (or the start/end of the text)
    above and below it. Plain-text headings almost always look like that.
    """
    line = line.strip()

    # Markdown: "# Title" is level 1, "## Title" is level 2 ...
    if line.startswith("#"):
        hashes = len(line) - len(line.lstrip("#"))
        return hashes if line[hashes:].strip() else None

    # Other headings must stand alone, be short, and not end like a sentence.
    if not stands_alone or len(line) > MAX_HEADING_LENGTH:
        return None
    if line.endswith((".", ",", ";", ":", "?", "!")):
        return None

    # Numbered: "2. Revenue" is level 1, "2.1 Details" is level 2
    match = NUMBERED_HEADING.match(line)
    if match:
        return match.group(1).count(".") + 1

    # All capitals: "RISKS"
    if line.isupper() and any(char.isalpha() for char in line):
        return 1

    # A short line on its own that starts with a capital letter: "Revenue Breakdown"
    if line[0].isupper() and len(line.split()) <= 8:
        return 1

    return None


def split_into_sections(text):
    """Return (title, sections). Each section is {"path": [...], "text": "..."}."""
    lines = text.split("\n")

    # The first non-empty line is the document title.
    first = next(i for i, line in enumerate(lines) if line.strip())
    title = lines[first].strip().lstrip("#").strip()
    lines = lines[first + 1 :]

    sections = []
    path = []  # current heading path, e.g. ["2. Revenue", "2.1 Details"]
    levels = []  # the level of each heading in path, e.g. [1, 2]
    body = []

    def save_section():
        section_text = "\n".join(body).strip()
        if section_text:
            sections.append({"path": list(path), "text": section_text})

    for i, line in enumerate(lines):
        blank_above = i == 0 or not lines[i - 1].strip()
        blank_below = i == len(lines) - 1 or not lines[i + 1].strip()
        level = heading_level(line, blank_above and blank_below) if line.strip() else None

        if level is None:
            body.append(line)
            continue

        save_section()  # the previous section ends where a new heading starts
        body = []
        # Close every open heading at the same or deeper level, then open this one.
        # "## Costs" after "### Revenue by city" closes both Revenue headings.
        while levels and levels[-1] >= level:
            levels.pop()
            path.pop()
        levels.append(level)
        path.append(line.strip().lstrip("#").strip())

    save_section()
    return title, sections


def make_chunks(text, source):
    """Split text into a list of LangChain Documents, section by section.

    Each chunk also remembers where its text sits in the document
    ("start" and "end" character positions). Late chunking needs this.
    """
    text = text.replace("\r\n", "\n")
    title, sections = split_into_sections(text)

    chunks = []
    cursor = 0  # we search forward, so repeated text is matched in the right place
    for section in sections:
        section_name = " > ".join(section["path"])
        header = f"[{title} > {section_name}]" if section_name else f"[{title}]"

        for piece in splitter.split_text(section["text"]):
            start = text.find(piece, cursor)
            if start == -1:
                raise PreCueError("Chunking error: a chunk was not found in the document.")
            cursor = start + 1  # +1, not the end: overlapping chunks start before the previous end

            info = {
                "source": source,
                "title": title,
                "section": section_name,
                "chunk": len(chunks) + 1,
                "start": start,
                "end": start + len(piece),
            }
            chunks.append(Document(page_content=f"{header}\n{piece}", metadata=info))

    return chunks
