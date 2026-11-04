"""Spoken and written numbers (app/numbers.py)."""

import pytest

from app.numbers import is_hedged, same_value, values_in, words_to_digits


@pytest.mark.parametrize("spoken, expected", [
    ("Revenue in q two was one hundred forty eight point six crore.", "Revenue in Q2 was 148.6 crore."),
    ("We hired ninety six people.", "We hired 96 people."),
    ("Exports grew eighteen percent.", "Exports grew 18 percent."),
    ("It was two point four million packs", "It was 2.4 million packs"),
    ("We have fifty two thousand merchants, and three plants.", "We have 52000 merchants, and 3 plants."),
    ("one more thing", "one more thing"),  # "one" alone is not a number
    ("No numbers here.", "No numbers here."),
    ("Revenue was 148.6 crore.", "Revenue was 148.6 crore."),  # digits stay digits
    ("twenty five", "25"),
    ("two hundred and five stores", "205 stores"),
    ("twenty-four lakh packs", "24 lakh packs"),
    ("Ninety-six people.", "96 people."),
    ("a one-time upgrade", "a one-time upgrade"),
    ("one point eight five lakh outlets", "1.85 lakh outlets"),
])
def test_words_to_digits(spoken, expected):
    assert words_to_digits(spoken) == expected


def test_values_in_converts_units():
    assert values_in("₹1.2 crore, 5%, 1,486 million rupees and 24 lakh") == [
        (12_000_000.0, False), (5.0, True), (1_486_000_000.0, False), (2_400_000.0, False)]


@pytest.mark.parametrize("said, document, hedged, expected", [
    ("24 lakh packs", "2.4 million packs", False, True),
    ("1,486 million rupees", "₹148.6 crore", False, True),
    ("185,000 outlets", "1.85 lakh", False, True),
    ("148.6 crore", "₹148.6 crore", False, True),
    ("around 1,200", "1,240", True, True),     # "around": a 5% rounding is fine
    ("1,200", "1,240", False, False),          # said exactly: 3% off is a different number
    ("150 crore", "₹148.6 crore", True, True),
    ("13%", "₹13 crore", False, False),        # a percentage is not an amount
    ("131.2 crore", "₹148.6 crore", False, False),
    ("41%", "14%", False, False),
    ("", "₹10 crore", False, False),
])
def test_same_value(said, document, hedged, expected):
    assert same_value(said, document, hedged) is expected


@pytest.mark.parametrize("text, expected", [
    ("Revenue was roughly 150 crore", True), ("We have around 1,200 employees", True),
    ("We have over 54 crore in cash", True), ("Revenue was 150 crore", False), ("Throughout the year", False),
])
def test_is_hedged(text, expected):
    assert is_hedged(text) is expected


def test_periods_are_not_values():
    assert values_in("Revenue in Q2 FY2026 was ₹148.6 crore") == [(1_486_000_000.0, False)]


@pytest.mark.parametrize("text, expected", [
    ("Attrition rose to 14.5 percent", "up"), ("Inventory days are down to about 40", "down"),
    ("Revenue was 148.6 crore", ""), ("Gross margin improved to 38.5%", ""),
    ("Sales rose while costs fell", ""),
])
def test_direction_of(text, expected):
    from app.numbers import direction_of
    assert direction_of(text) == expected
