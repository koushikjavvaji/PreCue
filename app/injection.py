"""Spots text in a document that looks like instructions to an AI ("ignore all previous instructions").

A real report never talks to an AI. If one does, someone may have hidden instructions in it
to make PreCue say something false (prompt injection). PreCue's checks already stop such
text from reaching an alert or a whisper (see fact_checker.py), but you should know about it:
the document may have been tampered with.

This is a simple word check, not a guarantee: it can miss cleverly worded instructions.
"""

import re

PATTERNS = [
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier|your) (?:instructions|rules|prompts?)",
    r"disregard (?:all |any )?(?:the )?(?:previous|prior|above|your) (?:instructions|rules)",
    r"\b(?:ai|llm) assistants?\b",
    r"\bsystem (?:note|prompt|message)\b",
    r"\byou are (?:now )?(?:an? )?(?:ai|assistant|chatbot|language model)\b",
    r"\b(?:assistant|ai|llm)\s*[,:]\s*(?:when|always|never|reply|respond|answer|say)\b",
    r"\b(?:always|only) (?:answer|reply|respond|mark)\b",
    r"\"verdict\"\s*:",
]
SUSPICIOUS = re.compile("|".join(PATTERNS), re.IGNORECASE)


def suspicious_sections(chunks):
    """The sections whose text looks like instructions to an AI, each named once, in order."""
    found = []
    for chunk in chunks:
        text = chunk.metadata.get("original", chunk.page_content)
        section = chunk.metadata["section"] or "(start of the document)"
        if SUSPICIOUS.search(text) and section not in found:
            found.append(section)
    return found
