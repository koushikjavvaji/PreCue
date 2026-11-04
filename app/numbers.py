"""Understanding the numbers people say and documents print.

  "₹148.6 crore", "1,486 million rupees"  -> 1,486,000,000 (the same amount)
  "24 lakh", "2.4 million"                -> 2,400,000
  "38.5%", "38.5 percent"                 -> 38.5 percent
  "ninety six", "one hundred forty eight point six" -> 96, 148.6
  "q two"                                 -> "Q2"

Used in two places:
  - before a check: spoken number words are written as digits, so search and the LLM see "96"
  - after a check: a "wrong" verdict is dropped if the two values are really the same
    (24 lakh = 2.4 million) or a fair rounding ("about 1,200" for 1,240).
    This can only remove an alert, never add one.
"""

import re

SCALES = {
    "thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "million": 1e6, "mn": 1e6,
    "crore": 1e7, "crores": 1e7, "cr": 1e7, "billion": 1e9, "bn": 1e9,
}
PERCENT_WORDS = {"%", "percent", "per cent", "pc"}
HEDGES = {"about", "around", "roughly", "approximately", "nearly", "almost", "over", "under", "close to",
          "more than", "less than", "just over", "just under", "some", "approx"}
EXACT_TOLERANCE = 0.005  # 0.5%: "148.6" said as "148.5" is still the same number
HEDGED_TOLERANCE = 0.05  # 5%: "about 1,200" for 1,240

ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORD_SCALES = {"hundred": 100, "thousand": 1000}
NUMBER_WORDS = set(ONES) | set(TENS) | set(WORD_SCALES) | {"point", "and"}

# A number, then maybe a scale or percent: "₹1,486", "2.4 million", "38.5 %", "1.85 lakh".
# Not a digit inside a word: the "2" of "Q2" and the "2026" of "FY2026" are periods, not values.
VALUE = re.compile(
    r"(?<![\w.])(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|per ?cent|pc\b|thousand|lakhs?|lac\b|million|mn\b|crores?|cr\b|billion|bn\b|k\b)?",
    re.IGNORECASE,
)


def values_in(text):
    """Every value in a text, as (number in base units, is_percent). "₹1.2 crore and 5%" -> [(12000000.0, False), (5.0, True)]."""
    found = []
    for match in VALUE.finditer(text):
        number = float(match["number"].replace(",", ""))
        unit = (match["unit"] or "").lower().replace(" ", "")
        if unit in ("%", "percent", "pc"):
            found.append((number, True))
        else:
            found.append((number * SCALES.get(unit, 1), False))
    return found


def same_value(said, document_says, hedged=False):
    """True if the two texts state the same value, after converting units.

    hedged=True (the speaker said "about", "around"...): a 5% difference still counts as the same.
    """
    spoken, written = values_in(said), values_in(document_says)
    if not spoken or not written:
        return False
    tolerance = HEDGED_TOLERANCE if hedged else EXACT_TOLERANCE
    (a, a_percent), (b, b_percent) = spoken[0], written[0]
    if a_percent != b_percent and a_percent and b_percent:
        return False
    return b != 0 and abs(a - b) / abs(b) <= tolerance


def is_hedged(text):
    """True if the speaker said the number loosely: "about 150 crore", "around 1,200"."""
    text = " " + text.lower() + " "
    return any(f" {hedge} " in text for hedge in HEDGES)


def words_to_digits(text):
    """Write spoken number words as digits. "we hired ninety six people" -> "we hired 96 people".

    Also "one hundred forty eight point six crore" -> "148.6 crore" and "q two" -> "Q2".
    "one" on its own is left alone: "one more thing" is not a number.
    """
    text = re.sub(r"\bq\s+(one|two|three|four)\b", lambda m: "Q" + str(ONES[m[1].lower()]), text, flags=re.IGNORECASE)
    words = text.split(" ")
    out, run = [], []
    for word in words + [""]:  # "" at the end flushes the last run
        bare = word.lower().strip(".,;:!?")
        if bare in NUMBER_WORDS and not (bare in ("and", "point") and not run):
            run.append(word)
            if word != word.rstrip(".,;:!?"):  # a comma or full stop ends the number
                out.append(_run_to_digits(run))
                run = []
            continue
        if run:
            out.append(_run_to_digits(run))
            run = []
        out.append(word)
    return " ".join(out[:-1]) if out and out[-1] == "" else " ".join(out)


def _run_to_digits(run):
    """["one", "hundred", "forty", "eight", "point", "six"] -> "148.6" (keeps a final punctuation mark)."""
    ending = run[-1][len(run[-1].rstrip(".,;:!?")):]
    words = [w.lower().strip(".,;:!?") for w in run]
    while words and words[-1] in ("and", "point"):
        words.pop()
    if words == ["one"]:
        return " ".join(run)  # "one" alone stays a word
    whole, decimals = (words[: words.index("point")], words[words.index("point") + 1:]) if "point" in words else (words, [])
    total, current = 0, 0
    for word in whole:
        if word in ONES:
            current += ONES[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word == "thousand":
            total += (current or 1) * 1000
            current = 0
    number = str(total + current)
    digits = "".join(str(ONES[w]) for w in decimals if w in ONES and ONES[w] < 10)
    return (number + ("." + digits if digits else "")) + ending
