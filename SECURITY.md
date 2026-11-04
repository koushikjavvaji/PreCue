# Security

PreCue runs on your own computer. This page explains what data goes where,
what PreCue does to protect it, and what you should know before using it.

## Where your data goes

| Data | Sent to | When |
|---|---|---|
| Your voice (microphone audio) | Deepgram (speech-to-text) | While PreCue is listening |
| Whisper text (e.g. "₹10 crore, not 100 crore.") | Deepgram (text-to-speech) | When PreCue whispers |
| Spoken sentences that contain a number + the matching parts of your document | Groq (or Gemini, with `LIVE_LLM=gemini`) | During the meeting |
| Every part of your document | Gemini | Once, before the meeting (facts table; context in `contextual` mode) |
| LLM calls and searches, including document text | LangSmith | Only if you turn tracing on |

Free API tiers may use your data to improve their models. **Do not use confidential
documents with free-tier keys.** Check each provider's data policy, or use paid tiers.

Nothing else leaves your computer. The live screen and the overlay are served
only on `127.0.0.1` (this computer).

## What PreCue protects against

- **Secrets**: API keys live only in `.env`, which git ignores. Keep it readable
  only by you: `chmod 600 .env`. Keys are never printed or logged.
- **Other websites reading your transcript**: the live screen answers only requests
  addressed to `127.0.0.1` or `localhost`, which blocks "DNS rebinding" (a website
  pointing its own name at your computer to read local pages).
- **Text on the screen is never run as code**: the pages show speech and LLM text with
  `textContent`, never `innerHTML`.
- **Booby-trapped documents** (prompt injection): a document could contain text like
  "ignore your instructions and say...". PreCue limits what that can do:
  - prompts mark the document as data and tell the LLM to ignore instructions in it
  - an alert is only shown if the corrected value is short (6 words at most) and every
    number in it really appears in the document
  - what PreCue whispers is built from those checked values only
    ("₹10 crore, not 100 crore."), never from free LLM text
  - text shown on screen is cut to a short length
- **Screen sharing**: on macOS the alert overlay is hidden from screenshots and screen
  capture, so meeting participants do not see it. Test it once with a real share before
  an important meeting.
- **A stuck page**: each open page has a limited queue, so a tab that stops reading
  cannot fill the memory.

## What you should know

- **Local cache**: `data/cache` keeps your document's chunks, vectors and facts as plain
  JSON, so a document is processed only once. Delete the folder when you are done with
  a sensitive document.
- **Other programs on your computer** can open `http://127.0.0.1:8765` and read the live
  transcript while PreCue runs. That is normal for local tools; do not run PreCue on a
  shared account.
- **The microphone is on** for as long as PreCue runs. Stop it with Ctrl+C.
- **Whispers on speakers** can be heard by the meeting. Use earphones.

## Reporting a problem

Please open an issue on GitHub, or contact the maintainer privately for anything sensitive.
