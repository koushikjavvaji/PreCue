"""Loads a text document and splits it into section-aware chunks.

How it works:
  1. Find the document title (the first line, if it looks like a title;
     otherwise the file name).
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

from app.documents import read_document
from app.errors import PreCueError

CHUNK_SIZE = 800  # max characters of section text per chunk (about 130-150 words)
CHUNK_OVERLAP = 100  # characters shared between chunks when a big section is split
MAX_HEADING_LENGTH = 80
# Raise this when the way chunks are made changes, so saved indexes are rebuilt (see index.py).
CHUNKER_VERSION = "2"  # 2: split tables repeat their column names

NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")  # "2. Revenue", "2.1 Details"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


def load_text(path):
    """Read a document (.txt, .md, .pdf, .docx or .pptx) and return its text (see documents.py)."""
    return read_document(path)


def heading_level(line, stands_alone, markdown_only=False):
    """Return the heading level (1 = top) if the line is a heading, otherwise None.

    stands_alone = the line has a blank line (or the start/end of the text)
    above and below it. Plain-text headings almost always look like that.
    """
    line = line.strip()

    # Markdown: "# Title" is level 1, "## Title" is level 2 ...
    if line.startswith("#"):
        hashes = len(line) - len(line.lstrip("#"))
        return hashes if line[hashes:].strip() else None

    # A document with markdown headings (every PDF, Word and PowerPoint file, see documents.py)
    # already says where its headings are. Guessing more would turn author names or figure labels into sections.
    if markdown_only:
        return None

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


def split_into_sections(text, default_title=""):
    """Return (title, sections). Each section is {"path": [...], "text": "..."}."""
    lines = text.split("\n")

    # The first non-empty line is the title, if it looks like one (short, not a sentence).
    # Otherwise it is normal text and must stay in the chunks; we use default_title instead.
    first = next(i for i, line in enumerate(lines) if line.strip())
    if heading_level(lines[first], stands_alone=True) is not None:
        title = lines[first].strip().lstrip("#").strip()
        lines = lines[first + 1 :]
    else:
        title = default_title
        lines = lines[first:]

    markdown_only = sum(re.match(r"#{2,} ", line) is not None for line in lines) >= 2  # "## ...", "### ..."
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
        level = heading_level(line, blank_above and blank_below, markdown_only) if line.strip() else None

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


def table_header_before(text, start):
    """If the text at `start` is in the middle of a markdown table, return that table's header lines.

    "| Revenue | ₹112.4 crore | ₹131.2 crore |" alone does not say which column is which quarter.
    The header is the row just above the "|---|" line. Returns "" when not inside a table.
    """
    if not text[start:].lstrip(" ").startswith("|"):
        return ""
    above = text[:start].rstrip("\n").split("\n")
    for i in range(len(above) - 1, 0, -1):
        line = above[i].strip()
        if not line.startswith("|"):
            return ""  # the table starts inside this piece, so the piece has the header itself
        if set(line) <= set("|-: "):  # the "|---|---|" line under the column names
            return f"{above[i - 1].strip()}\n{line}"
    return ""


def make_chunks(text, source):
    """Split text into a list of LangChain Documents, section by section.

    Each chunk also remembers where its text sits in the document
    ("start" and "end" character positions). Late chunking needs this.
    """
    text = text.replace("\r\n", "\n")
    title, sections = split_into_sections(text, default_title=Path(source).stem)

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
            # A long table split into pieces: repeat its column names, or "38.5%" means nothing.
            columns = table_header_before(text, start)
            body = f"{columns}\n{piece}" if columns else piece
            chunks.append(Document(page_content=f"{header}\n{body}", metadata=info))

    return chunks
