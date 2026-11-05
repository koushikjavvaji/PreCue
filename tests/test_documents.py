"""Reading PDF, Word and PowerPoint files (app/documents.py).

Test files are built on the fly with reportlab, python-docx and python-pptx, each one
made to have a known problem real documents have (two columns, page footers, wrapped
headings, scanned pages...).
"""

import sys
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer

import app.cache as cache
from app.chunker import make_chunks
from app.documents import (
    _is_page_number,
    _is_real_table,
    _join_lines,
    markdown_table,
    pdf_to_text,
    read_document,
)
from app.errors import PreCueError

SAMPLE_SECTIONS = ["1. Summary", "2. Revenue", "3. Customers", "4. Operations and Technology",
                   "5. Product Launches", "6. Team and Costs", "7. Risks", "8. Plan for Q3"]
SAMPLE_TITLE = "NIMBUS PAYMENTS - Q2 FY2026 BUSINESS REVIEW"
SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    """Each test gets an empty cache, so a saved conversion never hides a bug."""
    monkeypatch.setattr(cache, "CACHE_FOLDER", tmp_path / "cache")


def sections_of(path):
    chunks = make_chunks(read_document(path), str(path))
    return chunks[0].metadata["title"], {c.metadata["section"].split(" > ")[0] for c in chunks}


# ---------------- the three sample files (made by scripts/make_sample_documents.py) ----------------

@pytest.mark.parametrize("kind", ["pdf", "docx", "pptx"])
def test_every_format_gives_the_same_title_and_sections_as_the_text_file(kind):
    title, sections = sections_of(f"samples/sample_report.{kind}")
    assert title == SAMPLE_TITLE
    assert set(SAMPLE_SECTIONS) <= sections


@pytest.mark.parametrize("kind", ["pdf", "docx", "pptx"])
def test_every_format_keeps_the_numbers_and_the_table(kind):
    text = read_document(f"samples/sample_report.{kind}")
    for value in ["₹10 crore", "97.4%", "52,000", "1,850", "₹35 lakh", "₹150 crore"]:
        assert value in text, f"{value} lost in the {kind} file"
    assert "| Merchant fees | ₹6.2 crore | 62% |" in text  # the table, as a markdown table


def test_pdf_page_headers_footers_and_numbers_are_removed():
    text = read_document("samples/sample_report.pdf")
    assert "Nimbus Payments - Internal" not in text
    assert "Page " not in text


def test_pptx_keeps_speaker_notes():
    assert "Speaker notes: Mention that merchant fees are 62% of revenue." in read_document("samples/sample_report.pptx")


def test_docx_lists_become_bullets():
    assert "- UPI AutoPay for subscriptions, now used by 4,200 merchants." in read_document("samples/sample_report.docx")


# ---------------- PDFs with the problems real ones have ----------------

def make_two_column_pdf(path):
    """A title across the page, then two columns, each with its own heading and sentences."""
    heading = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=18)
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=13)
    title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20, leading=24)
    width, height = A4
    frames = [
        Frame(40, height - 110, width - 80, 60, id="title"),
        Frame(40, 60, (width - 100) / 2, height - 180, id="left"),
        Frame(60 + (width - 100) / 2, 60, (width - 100) / 2, height - 180, id="right"),
    ]
    left = " ".join(f"Left sentence {i} says the revenue was {i} crore." for i in range(1, 25))
    right = " ".join(f"Right sentence {i} says the churn was {i} percent." for i in range(1, 25))
    story = [Paragraph("Two Column Report", title), Spacer(1, 30),
             Paragraph("1. Revenue", heading), Paragraph(left, body),
             Paragraph("2. Churn", heading), Paragraph(right, body)]
    document = BaseDocTemplate(str(path), pagesize=A4)
    document.addPageTemplates([PageTemplate(frames=frames)])
    document.build(story)


def test_two_column_pdf_is_read_column_by_column(tmp_path):
    path = tmp_path / "two_columns.pdf"
    make_two_column_pdf(path)
    text = read_document(path)

    left = " ".join(f"Left sentence {i} says the revenue was {i} crore." for i in range(1, 25))
    right = " ".join(f"Right sentence {i} says the churn was {i} percent." for i in range(1, 25))
    assert left in text, "the left column is mixed with the right one"
    assert right in text
    assert text.index("1. Revenue") < text.index(left) < text.index("2. Churn") < text.index(right)


def test_two_column_pdf_sections(tmp_path):
    path = tmp_path / "two_columns.pdf"
    make_two_column_pdf(path)
    title, sections = sections_of(path)
    assert title == "Two Column Report"
    assert {"1. Revenue", "2. Churn"} <= sections


def test_borderless_two_column_table_stays_in_rows(tmp_path):
    """Labels on the left, values on the right, no lines: each label must stay next to its value."""
    path = tmp_path / "table.pdf"
    page = canvas.Canvas(str(path), pagesize=A4)
    page.setFont("Helvetica", 11)
    for i in range(30):
        page.drawString(60, 780 - i * 16, f"Metric number {i}")
        page.drawString(330, 780 - i * 16, f"{i * 11} crore")
    page.save()
    text = read_document(path)
    for i in range(30):
        assert f"Metric number {i} {i * 11} crore" in text


def test_heading_that_wraps_onto_two_lines_is_one_heading(tmp_path):
    path = tmp_path / "wrapped.pdf"
    page = canvas.Canvas(str(path), pagesize=A4)
    page.setFont("Helvetica-Bold", 22)
    page.drawString(40, 780, "A Very Long Report Title That")
    page.drawString(40, 754, "Wraps Onto Two Lines")
    page.setFont("Helvetica-Bold", 14)
    page.drawString(40, 700, "1. First Section")
    page.drawString(40, 560, "2. Second Section")
    page.setFont("Helvetica", 10)
    for i in range(8):
        page.drawString(40, 680 - i * 13, f"Body line {i} of the first section, with the value {i * 7} crore")
        page.drawString(40, 540 - i * 13, f"Body line {i} of the second section, with the value {i * 3} lakh")
    page.save()

    title, sections = sections_of(path)
    assert title == "A Very Long Report Title That Wraps Onto Two Lines"
    assert {"1. First Section", "2. Second Section"} <= sections


def test_scanned_pdf_gives_a_clear_error(tmp_path):
    path = tmp_path / "scanned.pdf"
    page = canvas.Canvas(str(path), pagesize=A4)
    page.rect(50, 50, 400, 600, fill=1)  # only a drawing: no text, and no picture of text either
    page.save()
    with pytest.raises(PreCueError, match="no text that can be read"):
        read_document(path)


def test_sideways_margin_text_is_left_out(tmp_path):
    path = tmp_path / "stamp.pdf"
    page = canvas.Canvas(str(path), pagesize=A4)
    page.setFont("Helvetica", 11)
    for i in range(10):
        page.drawString(60, 760 - i * 14, f"Normal line {i} with the value {i} crore.")
    page.saveState()
    page.translate(20, 300)
    page.rotate(90)
    page.drawString(0, 0, "arXiv:1706.03762v7 [cs.CL]")  # printed up the margin, like on arXiv papers
    page.restoreState()
    page.save()
    text = read_document(path)
    assert "Normal line 3 with the value 3 crore." in text
    assert "arXiv" not in text and "viXra" not in text


def test_conversion_is_cached_and_redone_when_the_file_changes(tmp_path, monkeypatch):
    import app.documents as documents

    path = tmp_path / "two_columns.pdf"
    make_two_column_pdf(path)
    calls = []
    real = documents.pdf_to_text
    monkeypatch.setattr(documents, "pdf_to_text", lambda p: calls.append(p) or real(p))

    first = read_document(path)
    assert read_document(path) == first and len(calls) == 1  # the second read comes from the cache

    make_two_column_pdf(path)  # same content, but a new file (reportlab writes a new id)
    read_document(path)
    assert len(calls) == 2


# ---------------- errors ----------------

def test_unsupported_file_type(tmp_path):
    path = tmp_path / "notes.rtf"
    path.write_text("hello")
    with pytest.raises(PreCueError, match="not supported"):
        read_document(path)


def test_missing_file():
    with pytest.raises(PreCueError, match="Could not read"):
        read_document("samples/no_such_file.pdf")


def test_broken_file(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(PreCueError, match="Could not read"):
        read_document(path)


def test_empty_word_file(tmp_path):
    import docx

    path = tmp_path / "empty.docx"
    docx.Document().save(path)
    text = read_document(path)  # just the title line from the file name
    assert text.strip() == "# empty"


# ---------------- small helpers ----------------

def test_markdown_table():
    assert markdown_table([["Metric", "Q1"], ["Revenue", "8.5 | 9"], ["Notes", None]]) == (
        "| Metric | Q1 |\n| --- | --- |\n| Revenue | 8.5 / 9 |\n| Notes |  |")


@pytest.mark.parametrize("rows, expected", [
    ([["a", "b"], ["1", "2"]], True),
    ([["only one column"], ["second row"]], False),
    ([["a", "b"]], False),
    (None, False),
])
def test_is_real_table(rows, expected):
    assert _is_real_table(rows) is expected


@pytest.mark.parametrize("text, expected", [
    ("12", True), ("Page 3", True), ("page 3 of 10", True), ("3 / 10", True),
    ("12 crore", False), ("Page layout", False),
])
def test_is_page_number(text, expected):
    assert _is_page_number(text) is expected


def test_join_lines_rejoins_hyphenated_words():
    assert _join_lines(["the transac-", "tion rate was", "Tier-", "2 cities"]) == "the transaction rate was Tier- 2 cities"


def test_too_big_files_are_refused(tmp_path, monkeypatch):
    import app.documents as documents

    big = tmp_path / "big.txt"
    big.write_text("x" * 2048)
    monkeypatch.setattr(documents, "MAX_FILE_MB", 0.001)
    with pytest.raises(PreCueError, match="limit"):
        documents.read_document(big)


def test_zip_bomb_is_refused(tmp_path, monkeypatch):
    import zipfile

    import app.documents as documents

    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "0" * 5_000_000)  # 5 MB that zips to a few KB
    assert bomb.stat().st_size < 50_000
    monkeypatch.setattr(documents, "MAX_UNZIPPED_MB", 1)
    with pytest.raises(PreCueError, match="unpacks to"):
        documents.read_document(bomb)


def test_too_much_text_is_refused(tmp_path, monkeypatch):
    import app.documents as documents

    long = tmp_path / "long.md"
    long.write_text("# T\n\n" + "word " * 1000)
    monkeypatch.setattr(documents, "MAX_TEXT_CHARACTERS", 100)
    with pytest.raises(PreCueError, match="too much text"):
        documents.read_document(long)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no Unix file permissions")
def test_cache_files_are_private(tmp_path):
    path = tmp_path / "cache" / "text_x.json"
    cache.write_json(path, {"text": "secret plans"})
    assert cache.read_json(path) == {"text": "secret plans"}
    assert path.stat().st_mode & 0o077 == 0  # no one else can read it
    assert not list(path.parent.glob("*.tmp"))


# ---------------- scanned PDFs (pictures of pages) ----------------

def scanned_pdf(path, pages):
    """A PDF whose pages are only pictures of text, like a scan: no text inside the PDF."""
    from PIL import Image, ImageDraw

    images = []
    for lines in pages:
        image = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(image)
        for number, line in enumerate(lines):
            draw.text((100, 120 + number * 60), line, fill="black")
        images.append(image)
    images[0].save(path, "PDF", save_all=True, append_images=images[1:], resolution=150)
    return path


def test_scanned_pages_are_read_from_their_pictures(tmp_path, monkeypatch):
    import app.ocr

    asked = []

    def fake_reader(png, llm):
        asked.append(png[:8])
        return f"## Page {len(asked)}\n\nRevenue was 1{len(asked)} crore."

    monkeypatch.setattr(app.ocr, "read_page_image", fake_reader)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    path = scanned_pdf(tmp_path / "scan.pdf", [["Revenue was 11 crore."], ["Revenue was 12 crore."]])
    text = read_document(path)
    assert asked == [b"\x89PNG\r\n\x1a\n"] * 2  # each page sent once, as a PNG picture
    assert "## Page 1" in text and "Revenue was 12 crore." in text


def test_normal_pdfs_never_use_the_picture_reader(monkeypatch):
    import app.ocr

    def not_called(*args):
        raise AssertionError("a PDF with real text was sent to the picture reader")

    monkeypatch.setattr(app.ocr, "read_page_image", not_called)
    assert "Revenue" in pdf_to_text(SAMPLES / "sample_report.pdf")


def test_too_many_scanned_pages_are_refused(tmp_path, monkeypatch):
    import app.ocr

    monkeypatch.setattr(app.ocr, "MAX_SCANNED_PAGES", 1)
    path = scanned_pdf(tmp_path / "scan.pdf", [["one"], ["two"]])
    with pytest.raises(PreCueError, match="scanned pages"):
        read_document(path)


def test_a_scanned_first_page_gives_the_title(tmp_path, monkeypatch):
    import app.ocr

    answers = iter(["# ORBIT FOODS REVIEW\n\n## 1. Summary\n\nRevenue was 64.2 crore.", "## 2. People\n\nHeadcount was 812."])
    monkeypatch.setattr(app.ocr, "read_page_image", lambda png, llm: next(answers))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    chunks = make_chunks(read_document(scanned_pdf(tmp_path / "scan.pdf", [["a"], ["b"]])), "scan.pdf")
    assert chunks[0].metadata["title"] == "ORBIT FOODS REVIEW"  # not the file name, not a second title line
    assert {c.metadata["section"] for c in chunks} == {"1. Summary", "2. People"}
