"""Builds PDF, Word and PowerPoint versions of samples/sample_report.txt, for testing.

Run from the project folder:
  python -m scripts.make_sample_documents

Writes samples/sample_report.pdf, .docx and .pptx. The PDF has what real PDFs have:
a header and a footer with page numbers on every page, a table, and bullet points.
Needs reportlab (only for this script): pip install reportlab
"""

import re
from pathlib import Path

SOURCE = Path("samples/sample_report.txt")
HEADING = re.compile(r"^\d+\.\s+\S")  # "2. Revenue"

# A table that repeats numbers from "2. Revenue", so the PDF has a real table to read.
REVENUE_TABLE = [
    ["Revenue source", "Q2 amount", "Share"],
    ["Merchant fees", "₹6.2 crore", "62%"],
    ["Subscription plans", "₹2.5 crore", "25%"],
    ["Value-added services", "₹1.3 crore", "13%"],
]


def read_source():
    """Return (title, note, sections). Each section is (heading, [paragraphs and "- bullets"])."""
    lines = SOURCE.read_text(encoding="utf-8").split("\n")
    title, note = lines[0].strip(), lines[1].strip()
    sections = []
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        if HEADING.match(line):
            sections.append((line, []))
        else:
            sections[-1][1].append(line)
    return title, note, sections


def make_pdf(path, title, note, sections):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import ListFlowable, Paragraph, SimpleDocTemplate, Spacer, Table

    # A font that has the ₹ sign (many, even Arial, do not). macOS ships Georgia, which does.
    regular, bold = "Helvetica", "Helvetica-Bold"
    fonts = Path("/System/Library/Fonts/Supplemental")
    if (fonts / "Georgia.ttf").exists() and (fonts / "Georgia Bold.ttf").exists():
        pdfmetrics.registerFont(TTFont("Georgia", str(fonts / "Georgia.ttf")))
        pdfmetrics.registerFont(TTFont("Georgia-Bold", str(fonts / "Georgia Bold.ttf")))
        regular, bold = "Georgia", "Georgia-Bold"
    else:
        print("Warning: no font with the ₹ sign found; the PDF will show it as a blank.")

    title_style = ParagraphStyle("title", fontName=bold, fontSize=18, leading=24, spaceAfter=6)
    heading_style = ParagraphStyle("heading", fontName=bold, fontSize=14, leading=18, spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("body", fontName=regular, fontSize=10.5, leading=14, spaceAfter=6)

    def header_and_footer(canvas, document):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, "Nimbus Payments - Internal")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {document.page}")
        canvas.restoreState()

    story = [Paragraph(title, title_style), Paragraph(note, body_style)]
    for heading, items in sections:
        story.append(Paragraph(heading, heading_style))
        bullets = [item[2:] for item in items if item.startswith("- ")]
        for item in items:
            if not item.startswith("- "):
                story.append(Paragraph(item, body_style))
        if bullets:
            story.append(ListFlowable([Paragraph(b, body_style) for b in bullets], bulletType="bullet"))
        if heading == "2. Revenue":
            table = Table(REVENUE_TABLE)
            table.setStyle([("FONTNAME", (0, 0), (-1, -1), regular), ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("GRID", (0, 0), (-1, -1), 0.5, "#888888")])
            story += [Spacer(1, 6), table, Spacer(1, 6)]
        story.append(Spacer(1, 120))  # push sections apart, so the report spans several pages

    document = SimpleDocTemplate(str(path), pagesize=A4, title="", topMargin=2 * cm, bottomMargin=2 * cm)
    document.build(story, onFirstPage=header_and_footer, onLaterPages=header_and_footer)


def make_docx(path, title, note, sections):
    import docx

    document = docx.Document()
    document.add_paragraph(title, style="Title")
    document.add_paragraph(note)
    for heading, items in sections:
        document.add_heading(heading, level=1)
        for item in items:
            if item.startswith("- "):
                document.add_paragraph(item[2:], style="List Bullet")
            else:
                document.add_paragraph(item)
        if heading == "2. Revenue":
            table = document.add_table(rows=0, cols=3)
            for row in REVENUE_TABLE:
                cells = table.add_row().cells
                for cell, value in zip(cells, row):
                    cell.text = value
    document.save(path)


def make_pptx(path, title, note, sections):
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    presentation.core_properties.title = title
    first = presentation.slides.add_slide(presentation.slide_layouts[0])
    first.shapes.title.text = title
    first.placeholders[1].text = note
    for heading, items in sections:
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])  # title + content
        slide.shapes.title.text = heading
        body = slide.placeholders[1].text_frame
        body.text = items[0].removeprefix("- ")
        for item in items[1:]:
            body.add_paragraph().text = item.removeprefix("- ")
        if heading == "2. Revenue":
            shape = slide.shapes.add_table(len(REVENUE_TABLE), 3, Inches(1), Inches(5), Inches(8), Inches(1.5))
            for r, row in enumerate(REVENUE_TABLE):
                for c, value in enumerate(row):
                    shape.table.cell(r, c).text = value
            slide.notes_slide.notes_text_frame.text = "Mention that merchant fees are 62% of revenue."
    presentation.save(path)


def main():
    title, note, sections = read_source()
    for kind, make in [("pdf", make_pdf), ("docx", make_docx), ("pptx", make_pptx)]:
        path = SOURCE.with_suffix(f".{kind}")
        make(path, title, note, sections)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
