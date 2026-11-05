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
- **Other websites using the app**: the app's page can save keys, upload a file and start the
  microphone. A website you visit cannot make those requests: the server only accepts them from
  its own page (the browser's `Origin` must be exactly `http://127.0.0.1:<port>`) and only as JSON
  or a file upload, never as a plain web form. The page never receives your keys back, only
  whether each one is set and works, and it can only start documents from its own list
  (never a file path).
- **Saved keys and uploads**: keys typed into the app go into `.env`, written so only your user
  account can read it. Uploaded documents are kept in `data/uploads` (never committed), also
  readable by you only, with their names cleaned (no folders, no odd characters). Remove one
  with the × next to it in the app; the samples cannot be removed. (Its cached text and facts stay
  in `data/cache` until you delete that folder.)
- **The website** (getprecue.vercel.app) is a static page plus one small function that reads the
  latest version from GitHub. It never receives your documents, keys or transcripts, and it sends
  strict security headers (no outside scripts, no framing, no camera or microphone).
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
  - when a document is loaded, PreCue warns you about sections that look like instructions
    to an AI ("ignore all previous instructions"), so you can check it was not tampered with
  - tested with `samples/injection_report.md`, a report full of such traps: every claim
    still got the right verdict, and no alert used the planted text
- **Screen sharing**: on macOS and Windows (10 version 2004 or newer) the alert overlay is
  hidden from screenshots and screen capture, so meeting participants do not see it.
  On Linux it is not hidden, so it is off unless you ask for it (`--overlay`). Test it once with a real share before
  an important meeting.
- **A stuck page**: each open page has a limited queue, so a tab that stops reading
  cannot fill the memory.
- **Huge or booby-trapped files**: files over 100 MB, PDFs over 1,000 pages, Word and
  PowerPoint files that unpack to over 500 MB (a "zip bomb") and documents with over
  5 million characters are refused before they can freeze the computer or use up the free tier.

## What you should know

- **Local cache**: `data/cache` keeps your document's text, chunks, vectors and facts as
  plain JSON, so a document is processed only once. The files can be read only by your
  user account (on macOS and Linux). Delete the folder when you are done with a
  sensitive document.
- **Other programs on your computer** can open `http://127.0.0.1:8765` and read the live
  transcript while PreCue runs. That is normal for local tools; do not run PreCue on a
  shared account.
- **The microphone is on** for as long as PreCue runs. Stop it with Ctrl+C.
- **Whispers on speakers** can be heard by the meeting. Use earphones.

## Reporting a problem

Please open an issue on GitHub, or contact the maintainer privately for anything sensitive.
