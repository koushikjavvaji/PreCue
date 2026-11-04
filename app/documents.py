"""Reads a document file and returns plain text that the chunker understands.

The chunker works on text with headings ("## 2. Revenue") and paragraphs
separated by blank lines. Every format is turned into exactly that:

  .txt / .md  -> read as they are
  .pdf        -> headings found by font (bigger or bold), tables as text tables,
                 two-column pages read column by column,
                 page headers, footers and page numbers removed
  .docx       -> Word's heading styles, lists and tables, in document order
  .pptx       -> one section per slide (named by its title, or "Slide 3" if it has none),
                 bullets, tables and speaker notes

The output always starts with "# <document title>", then "## " headings.
Tables become markdown tables ("| a | b |"), which the chunker keeps together.

All three libraries (pdfplumber, python-docx, python-pptx) are MIT-licensed.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

from app.cache import cache_file, read_json, write_json
from app.errors import PreCueError

SUPPORTED_FILES = {".txt", ".md", ".pdf", ".docx", ".pptx"}
MAX_HEADING_LENGTH = 80  # characters; a longer line is a sentence, not a heading
HEADING_SIZE_RATIO = 1.15  # a line this much bigger than normal text is a heading
EDGE = 0.08  # the top and bottom 8% of a page: where headers, footers and page numbers live
WORD_GAP = 1.5  # points; a smaller gap between letters is inside a word. (pdfplumber's default, 3,
                # glues the words of tightly printed PDFs together: "Thedominantsequence...")


def read_document(path):
    """Return the text of a document file, ready for the chunker."""
    path = Path(path)
    kind = path.suffix.lower()
    if kind not in SUPPORTED_FILES:
        names = ", ".join(sorted(SUPPORTED_FILES))
        raise PreCueError(f"'{path.name}' is not supported. Use one of: {names}.")
    try:
        if kind in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
        else:
            text = _read_cached(path, kind)
    except PreCueError:
        raise
    except Exception as error:  # missing, broken or locked file: each library raises its own errors
        raise PreCueError(f"Could not read '{path.name}': {error}") from error

    text = text.replace("\r\n", "\n")  # Windows line endings -> normal
    if not text.strip():
        raise PreCueError(f"'{path.name}' is empty.")
    return text


def _read_cached(path, kind):
    """Reading a 200-page PDF takes ~25 seconds, so the text is saved in the cache.
    The key is a fingerprint of the file's bytes: a changed file is read again."""
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = cache_file("text", kind, fingerprint)
    saved = read_json(cache)
    if saved is not None:
        return saved["text"]
    text = _convert(path, kind)
    write_json(cache, {"file": path.name, "text": text})
    return text


def _convert(path, kind):
    if kind == ".pdf":
        return pdf_to_text(path)
    if kind == ".docx":
        return docx_to_text(path)
    return pptx_to_text(path)


# ---------------- PDF ----------------

def pdf_to_text(path):
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages = [_read_page(page) for page in pdf.pages]
        title = (pdf.metadata or {}).get("Title", "")

    lines = [line for page in pages for line in page if line["kind"] == "line"]
    if not lines and not any(page for page in pages):
        raise PreCueError(
            f"'{path.name}' has no text that can be read. It may be a scanned PDF "
            "(pictures of pages); PreCue cannot read those yet."
        )

    repeated = _repeated_edge_lines(pages)
    body_size = _most_common_size(lines)
    pages = [_join_wrapped_headings(page, body_size) for page in pages]
    # The real content: without page headers, footers and page numbers.
    lines = [line for page in pages for line in page
             if line["kind"] == "line" and not _is_page_furniture(line, repeated)]
    heading_sizes = sorted({line["size"] for line in lines if _is_big(line, body_size)}, reverse=True)

    # The title: the line in the biggest font on the first page, if no other line has that size.
    first_page = [line for line in pages[0] if line["kind"] == "line" and not _is_page_furniture(line, repeated)]
    biggest = max(first_page, key=lambda line: line["size"], default=None)
    if biggest and _is_big(biggest, body_size) and not _is_heading_of_a_section(biggest, lines):
        title = biggest["text"]
        biggest["skip"] = True
        heading_sizes = sorted({line["size"] for line in lines if _is_big(line, body_size) and not line.get("skip")},
                               reverse=True)

    out = [f"# {title.strip() or path.stem}", ""]
    paragraph = []

    def end_paragraph():
        if paragraph:
            out.extend([_join_lines(paragraph), ""])
            paragraph.clear()

    for page in pages:
        previous = None
        for item in page:
            if item["kind"] == "table":
                end_paragraph()
                out.extend([item["text"], ""])
                previous = None
                continue
            if item.get("skip") or _is_page_furniture(item, repeated):
                continue

            level = _heading_level(item, body_size, heading_sizes)
            if level:
                end_paragraph()
                out.extend(["#" * level + " " + item["text"], ""])
                previous = None
                continue

            # A bigger gap than usual between two lines means a new paragraph.
            if previous and item["top"] - previous["bottom"] > 0.6 * (item["bottom"] - item["top"]):
                end_paragraph()
            paragraph.append(item["text"])
            previous = item
        end_paragraph()  # a page break ends a paragraph

    return "\n".join(out)


def _read_page(page):
    """The page as a list of items in reading order: text lines and tables.

    Two-column pages are read like a person reads them: the left column top to bottom,
    then the right column. Text that spans both columns (a title, a wide table) stays in place.
    """
    # Leave out sideways text (e.g. a stamp printed up the margin): it would come out scrambled.
    page = page.filter(lambda obj: obj.get("object_type") != "char" or obj.get("upright", True))
    items = []
    for region in _reading_regions(page):
        items.extend(_read_region(page.within_bbox(region), page.height))
    return items


def _read_region(region, page_height):
    """The text lines and tables of one part of a page, top to bottom."""
    items = []
    boxes = []
    for table in region.find_tables():
        rows = table.extract()
        if _is_real_table(rows):  # otherwise its lines are read as normal text below
            boxes.append(table.bbox)
            items.append({"kind": "table", "top": table.bbox[1], "text": markdown_table(rows)})

    for line in region.extract_text_lines(return_chars=True, x_tolerance=WORD_GAP):
        text = line["text"].strip()
        middle = (line["top"] + line["bottom"]) / 2
        inside_table = any(x0 <= line["x0"] <= x1 and top <= middle <= bottom for x0, top, x1, bottom in boxes)
        if not text or inside_table:
            continue
        chars = [c for c in line["chars"] if c["text"].strip()]
        items.append({
            "kind": "line",
            "text": text,
            "top": line["top"],
            "bottom": line["bottom"],
            "size": round(sum(c["size"] for c in chars) / len(chars) * 2) / 2,  # rounded to 0.5 pt
            "bold": all(_is_bold_font(c["fontname"]) for c in chars),
            "at_edge": line["top"] < page_height * EDGE or line["bottom"] > page_height * (1 - EDGE),
        })
    return sorted(items, key=lambda item: item["top"])


def _reading_regions(page):
    """The parts of the page in reading order, as boxes (x0, top, x1, bottom).

    One column: the whole page. Two columns: walking down the page, every band of
    full-width text is one part, and between those bands come the left column, then the right.
    """
    whole = (0, 0, page.width, page.height)
    words = page.extract_words(x_tolerance=WORD_GAP)
    gutter = _find_gutter(words, page.width)
    if gutter is None:
        return [whole]

    # Full-width text: words that cross the gutter. Their rows become full-width bands.
    bands = []
    for word in sorted((w for w in words if w["x0"] < gutter < w["x1"]), key=lambda w: w["top"]):
        if bands and word["top"] <= bands[-1][1] + 2:
            bands[-1][1] = max(bands[-1][1], word["bottom"])
        else:
            bands.append([word["top"], word["bottom"]])

    regions = []
    y = 0
    for top, bottom in bands + [[page.height, page.height]]:
        if top > y:  # the two columns between the previous band and this one
            regions += [(0, y, gutter, top), (gutter, y, page.width, top)]
        if bottom > top:
            regions.append((0, top, page.width, bottom))  # the full-width band itself
        y = max(y, bottom)
    return regions


def _find_gutter(words, page_width):
    """The x position of the empty strip between two columns, or None for a one-column page.

    We look in the middle of the page for the x that the fewest words cross. It is a gutter if:
      - almost no words cross it (in one-column text, nearly every line crosses the middle)
      - each side is a real column of text: 20+ words, spread over a quarter of the page width.
        (A table without borders has narrow columns: those must stay rows, "Revenue  10 crore".)
    """
    if len(words) < 40:
        return None
    best_x, fewest = None, len(words)
    for x in range(int(page_width * 0.35), int(page_width * 0.65)):
        crossing = sum(1 for w in words if w["x0"] < x < w["x1"])
        if crossing < fewest:
            best_x, fewest = x, crossing
    if fewest > len(words) * 0.03:
        return None
    left = [w for w in words if w["x1"] <= best_x]
    right = [w for w in words if w["x0"] >= best_x]
    return best_x if _is_text_column(left, page_width) and _is_text_column(right, page_width) else None


def _is_text_column(words, page_width):
    if len(words) < 20:
        return False
    width = max(w["x1"] for w in words) - min(w["x0"] for w in words)
    return width >= page_width * 0.25


def _is_bold_font(name):
    """Font names say the weight: "Arial-BoldMT", "NimbusRomNo9L-Medi", "Inter-SemiBold"."""
    name = name.lower()
    return any(weight in name for weight in ("bold", "medi", "semibold", "heavy", "black"))


def _is_real_table(rows):
    """At least 2 rows and 2 filled columns. A "table" with one column is a misread layout, not a table."""
    filled = [sum(1 for cell in row if cell and cell.strip()) for row in rows or []]
    return len(filled) >= 2 and sum(count >= 2 for count in filled) >= len(filled) / 2


def _join_wrapped_headings(page, body_size):
    """A long heading can wrap onto a second line: "Q2 FY2026 BUSINESS" + "REVIEW" is one heading."""
    joined = []
    for item in page:
        previous = joined[-1] if joined else None
        if (previous and item["kind"] == previous["kind"] == "line"
                and item["size"] == previous["size"] >= body_size * HEADING_SIZE_RATIO
                and item["top"] - previous["bottom"] < 0.6 * (item["bottom"] - item["top"])):
            previous["text"] += " " + item["text"]
            previous["bottom"] = item["bottom"]
        else:
            joined.append(item)
    return joined


def _repeated_edge_lines(pages):
    """Header and footer lines: the same text near the top or bottom of most pages."""
    if len(pages) < 3:
        return set()
    seen = Counter()
    for page in pages:
        seen.update({_normalise(item["text"]) for item in page if item["kind"] == "line" and item["at_edge"]})
    return {text for text, count in seen.items() if count >= len(pages) / 2}


def _is_page_furniture(line, repeated):
    """Page headers, footers and page numbers: on every page, but not part of the content."""
    return _normalise(line["text"]) in repeated or _is_page_number(line["text"])


def _normalise(text):
    """Page 3 and Page 4 should look the same: replace digits."""
    return re.sub(r"\d+", "#", text.lower())


def _is_page_number(text):
    return re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", text.strip().lower()) is not None


def _most_common_size(lines):
    sizes = Counter()
    for line in lines:
        sizes[line["size"]] += len(line["text"])  # weighted by length: body text has the most letters
    return sizes.most_common(1)[0][0] if sizes else 0


def _is_big(line, body_size):
    return line["size"] >= body_size * HEADING_SIZE_RATIO and _looks_like_a_heading(line["text"])


def _is_heading_of_a_section(line, lines):
    """True if other lines share this size: then it is a section heading, not a one-off title."""
    return sum(1 for other in lines if other["size"] == line["size"]) > 1


def _looks_like_a_heading(text):
    return len(text) <= MAX_HEADING_LENGTH and not text.endswith((".", ",", ";", ":"))


def _heading_level(line, body_size, heading_sizes):
    """2 for the biggest headings, 3 for the next size ... Bold lines of normal size are the lowest level."""
    if not _looks_like_a_heading(line["text"]):
        return None  # a paragraph printed in a big font is still a paragraph
    if line["size"] in heading_sizes:
        return min(2 + heading_sizes.index(line["size"]), 4)
    if line["bold"] and line["size"] >= body_size:
        return min(2 + len(heading_sizes), 4)
    return None


def _join_lines(lines):
    """Join the lines of one paragraph. "transac-" + "tion" -> "transaction"."""
    text = lines[0]
    for line in lines[1:]:
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text += " " + line
    return text


# ---------------- Word ----------------

def docx_to_text(path):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    title = document.core_properties.title or ""
    out = []
    for block in document.element.body.iterchildren():  # paragraphs and tables, in document order
        if block.tag.endswith("}tbl"):
            rows = [[cell.text for cell in row.cells] for row in Table(block, document).rows]
            out.extend([markdown_table(rows), ""])
            continue
        if not block.tag.endswith("}p"):
            continue
        paragraph = Paragraph(block, document)
        text = paragraph.text.strip()
        if not text:
            continue
        style = paragraph.style.name.lower() if paragraph.style is not None else ""
        if style == "title" and not title and not out:
            title = text  # the document's own title at the very top
        elif style.startswith("heading") or style == "title":
            level = int(style.split()[-1]) if style.split()[-1].isdigit() else 1
            out.extend(["#" * min(level + 1, 4) + " " + text, ""])
        elif "list" in style:
            out.append("- " + text)
        else:
            out.extend([text, ""])
    return "\n".join([f"# {title.strip() or path.stem}", ""] + out)


# ---------------- PowerPoint ----------------

def pptx_to_text(path):
    from pptx import Presentation

    presentation = Presentation(path)
    title = presentation.core_properties.title or path.stem
    out = [f"# {title.strip()}", ""]
    for number, slide in enumerate(presentation.slides, start=1):
        slide_title = slide.shapes.title.text.strip() if slide.shapes.title is not None else ""
        out.extend([f"## {slide_title or f'Slide {number}'}", ""])  # the slide's title names its section
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            out.extend(_shape_text(shape))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                out.extend([f"Speaker notes: {notes}", ""])
    return "\n".join(out)


def _shape_text(shape):
    """The text of one shape (text box, table or a group of shapes) as lines."""
    lines = []
    if shape.shape_type == 6:  # a group: read the shapes inside it
        for inner in shape.shapes:
            lines.extend(_shape_text(inner))
    elif getattr(shape, "has_table", False) and shape.has_table:
        rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
        lines.extend([markdown_table(rows), ""])
    elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            text = "".join(run.text for run in paragraph.runs).strip()
            if text:
                lines.append("- " + text)
        lines.append("")
    return lines


# ---------------- shared ----------------

def markdown_table(rows):
    """[["Metric", "Q1"], ["Revenue", "8.5"]] -> a markdown table, one line per row."""
    clean = [[(cell or "").replace("\n", " ").replace("|", "/").strip() for cell in row] for row in rows]
    lines = ["| " + " | ".join(clean[0]) + " |", "|" + " --- |" * len(clean[0])]
    lines += ["| " + " | ".join(row) + " |" for row in clean[1:]]
    return "\n".join(lines)
