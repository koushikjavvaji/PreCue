"""Chunking, numbers and search words."""

import pytest

from app.chunker import make_chunks
from app.facts import is_grounded, numbers_in
from app.retriever import has_number, words


@pytest.mark.parametrize("value, text, expected", [
    ("₹10 crore", "Revenue was ₹10 crore.", True),
    ("1,850", "Peak load was 1850 per second", True),
    ("1850", "Peak load was 1,850 per second", True),
    ("87%", "It was 87.0% in Q1", True),
    ("₹1 crore", "Revenue was ₹10 crore.", False),  # "1" is inside "10", but it is not the same number
    ("5%", "Efficiency improved by 15%", False),
    ("10 and 18%", "₹10 crore, up 18%", True),
    ("10 and 19%", "₹10 crore, up 18%", False),
])
def test_is_grounded(value, text, expected):
    assert is_grounded(value, text) is expected


def test_numbers_in():
    assert numbers_in("₹10 crore, up 18% (1,850)") == {10.0, 18.0, 1850.0}


def test_words_keep_numbers_whole():
    assert words("Peak 1,850 and 2.3 lakh") == ["peak", "1850", "2.3", "lakh"]
    assert has_number("Q2") and not has_number("hello")


def test_title_line_is_used_as_title():
    chunks = make_chunks("NIMBUS REPORT\n\n1. Summary\n\nRevenue was 10 crore.\n", "report.txt")
    assert chunks[0].metadata["title"] == "NIMBUS REPORT"
    assert "NIMBUS REPORT" not in chunks[0].page_content.split("\n", 1)[1]  # not repeated as text


def test_first_line_that_is_a_sentence_is_kept():
    chunks = make_chunks("Revenue was 10 crore this quarter.\nChurn fell to 3.2 percent.\n", "notes.txt")
    assert chunks[0].metadata["title"] == "notes"  # the file name
    assert "Revenue was 10 crore" in chunks[0].page_content


def test_one_line_document_gives_one_chunk():
    assert len(make_chunks("Just one line, 42 percent.\n", "one.txt")) == 1


def test_sections_and_positions():
    text = "Title\n\n1. Revenue\n\nRevenue was 10 crore.\n\n2. Costs\n\nCosts were 7 crore.\n"
    chunks = make_chunks(text, "doc.txt")
    assert [c.metadata["section"] for c in chunks] == ["1. Revenue", "2. Costs"]
    for c in chunks:
        assert text[c.metadata["start"]:c.metadata["end"]] in c.page_content
