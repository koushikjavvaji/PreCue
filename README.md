# PreCue

[![tests](https://github.com/koushikjavvaji/PreCue/actions/workflows/tests.yml/badge.svg)](https://github.com/koushikjavvaji/PreCue/actions/workflows/tests.yml)

**A real-time AI voice copilot that listens, understands, verifies, and assists you during important conversations through intelligent visual alerts and private audio whispers.**

PreCue is not another voice chatbot. It is a real-time communication assistant that stays almost invisible until you need it.

## The problem

You have a 100–200 page document and need to explain it to your manager in 5–10 minutes. You have to:

- Remember key facts, statistics and numbers
- Summarize complex information in a few sentences
- Avoid saying anything incorrect
- Remember what to say next
- Speak confidently without constantly checking the document

PreCue acts as a real-time assistant while you speak.

## Features

### Real-time fact verification
If your document says revenue is ₹10 crore and you say ₹100 crore, PreCue detects the mismatch, shows an on-screen alert (what you said vs. what the document says) and whispers the correction through your earphones. If you speak correctly, it never interrupts.

### Continuation suggestions
If you pause or get stuck, PreCue uses the conversation context to retrieve relevant information from your documents and suggests the next point, shown on screen and optionally whispered.

### Non-intrusive assistance
A decision engine chooses one of four behaviors:

| Situation | Behavior |
|---|---|
| Speaking correctly | Stay silent |
| Incorrect statistic | Alert + whisper correction |
| Stuck or paused | Suggest next point |
| Continuing normally | Stay silent |

## How it works

Before the meeting, you upload documents (PDFs, reports, presentations). They are chunked, embedded and stored in a vector database. During the meeting, live speech is transcribed, relevant chunks are retrieved, and an LLM checks claims and generates suggestions, grounded in your documents rather than the model's general knowledge.

| Component | Responsibility |
|---|---|
| STT | Convert live speech to text |
| VAD | Detect when the user starts and stops speaking |
| RAG | Retrieve relevant document information |
| LLM | Understand context, verify claims, generate suggestions |
| Decision engine | Decide whether to stay silent, alert, or suggest |
| TTS | Convert suggestions into spoken audio |
| Audio output | Deliver private whispers through earphones |
| Frontend | Display live transcription and alerts |

## Engineering challenges

1. **Real-time speech understanding**: interpreting speech while it is still being spoken
2. **Grounded fact verification**: comparing spoken claims against source documents
3. **Context-aware assistance**: anticipating what the speaker needs next
4. **Interruption management**: knowing when to speak and when to stay silent
5. **Low-latency private audio**: whispering useful information without disrupting the conversation

## Status

Early stage. Built so far:

- **Live speech-to-text** with Deepgram (partial and final text, speech start and pause events)
- **Documents**: `.pdf`, `.docx` (Word), `.pptx` (PowerPoint), `.txt` and `.md`. PDFs keep their headings (found by font), tables, and reading order on two-column pages; page headers, footers and page numbers are removed. Scanned PDFs (pictures of pages) are not supported yet
- **Section-aware chunking**
- **Contextual retrieval** (an LLM writes a short context for each chunk) using Gemini
- **Embeddings** with a local model, in three modes: `plain`, `contextual` and `late` (late chunking, also for long documents)
- **Hybrid search**: vector search plus BM25 keyword search for questions with numbers, merged with Reciprocal Rank Fusion
- **Facts table**: every number-based fact in the document, extracted by Gemini and checked against the text
- **Saved index**: chunks and vectors are cached, so a document is only processed once
- **Evaluation**: a question set and a script that scores every mode and search method
- **LangSmith tracing** (optional)
- **Live fact-checking**: each spoken sentence with a number is checked against the document by a fast LLM (Groq)
- **Decision engine** (LangGraph): stay silent, alert on a wrong number, or suggest the next point when you pause; never repeats itself
- **Whispers**: alerts and suggestions spoken into your earphones (Deepgram Aura text-to-speech)
- **Live screen**: a local web page with the transcript, alerts and suggestions
- **Meeting overlay** (macOS and Windows): alerts pop up in a small floating window on top of every app, also full-screen Zoom and slideshows. It is click-through, never takes the focus, fades after a few seconds and is **hidden from screen sharing**: the people in the meeting never see it. Linux cannot hide a window from screen sharing, so there PreCue opens a browser tab unless you add `--overlay`
- **Security**: see [SECURITY.md](SECURITY.md) for where your data goes and what PreCue protects against
- **Built for long meetings**: keeps the connection alive through silence, reconnects after network drops, joins sentences split by a pause
- **Short phrases end on time**: after 0.8 s of silence PreCue asks Deepgram to finish the sentence, so "99 percent." is not glued to the next sentence
- **Automated tests**: 65 offline tests (no keys, no microphone), plus 13 real-window overlay tests, for the decision engine, fact-check safety rules, chunking, speech events, reconnects, the live screen, whispers and the overlay

Coming next: scanned PDFs (OCR), charts and images inside documents.

## Setup

Requires Python 3.11+ and a microphone. Tested on macOS.

```bash
git clone https://github.com/koushikjavvaji/PreCue.git
cd PreCue
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and add your keys:

| Key | Needed for | Where to get it |
|---|---|---|
| `DEEPGRAM_API_KEY` | Live speech-to-text | [deepgram.com](https://deepgram.com) (free credit on sign-up) |
| `GEMINI_API_KEY` | Contextual retrieval and the facts table | [aistudio.google.com](https://aistudio.google.com) (free tier) |
| `GROQ_API_KEY` | Live fact-checking | [console.groq.com](https://console.groq.com) (free tier) |

Never commit `.env`. It is already in `.gitignore`.

Optional settings in `.env`:

- `GEMINI_MODEL`: the Gemini model to use (default `gemini-3.5-flash-lite`)
- `GEMINI_REQUESTS_PER_MINUTE`: how fast to call Gemini (default `15`, the free-tier limit). Calls are spaced out automatically.
- `GROQ_MODEL` (default `openai/gpt-oss-20b`) and `GROQ_REQUESTS_PER_MINUTE` (default `30`)
- `LIVE_LLM`: `groq` (default) or `gemini`, if you have no Groq key. Gemini is fine for `check_claims`, but too slow for a live meeting: in our test a check took about 20 seconds, so most alerts came too late and were skipped.
- `WHISPER_DEVICE`: part of your earphones' name, e.g. `AirPods`. List devices with `python -m sounddevice`. Without it, whispers play on the default output.
- `WHISPER_VOLUME`: from `0` to `1` (default `0.6`)
- `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`: send traces of every LLM call and search to [LangSmith](https://smith.langchain.com). Off by default. Traces include document text, so use test documents.

## Usage

Run all commands from the project folder.

**Live copilot (the main app).** Prepares the document, starts listening and shows alerts in a small pop-up window in the top-right corner of your screen, on top of your meeting and hidden from screen sharing. The full page with the transcript is at the printed "Live screen" address. Use earphones: on laptop speakers the meeting could hear the whispers.

```bash
python -m scripts.live_copilot samples/sample_report.txt
python -m scripts.live_copilot samples/sample_report.txt --no-voice    # pop-ups only, no whispers
python -m scripts.live_copilot samples/sample_report.txt --no-overlay  # browser tab instead of pop-ups
python -m scripts.live_copilot my_report.pdf                           # any .pdf, .docx, .pptx, .txt or .md
```

| You say | PreCue does |
|---|---|
| A correct number | Stays silent |
| A wrong number | Alert on screen + whispers the correct value (once per mistake) |
| Nothing for 4 seconds after speaking | Suggests the next section you have not covered yet |
| Small talk | Stays silent (no LLM call) |

**Check claims by typing** (no microphone). Add `--eval` to measure accuracy on a set of right and wrong claims.

```bash
python -m scripts.check_claims samples/sample_report.txt
python -m scripts.check_claims samples/sample_report.txt --eval samples/eval_claims.json
```

On the sample, the checker gets 36-37 of the 37 claims right (correct, wrong and not in the document) with 0 false alerts, on both Groq and Gemini. A check takes 2-3 seconds on Groq. The Groq free tier allows 8,000 tokens per minute (about 4-6 checks) and 1,000 requests per day, so the eval, which sends all 37 claims back to back, slows down to about 10 seconds per claim; a real meeting does not. The last 12 claims were added after tuning, as a fair check. The sample is small, so test with your own documents too.

**Run the tests** (offline, a few seconds):

```bash
python -m pytest tests
```

The overlay tests open real windows and take screenshots, so they run only when asked (macOS, and your terminal needs Screen Recording permission). They check that alerts are visible to you, also over a full-screen app, and invisible to screenshots and to ScreenCaptureKit, the screen capture that Zoom, Teams and Chrome use:

```bash
PRECUE_UI_TESTS=1 python -m pytest tests
```

Before an important meeting, also try a real screen share once: share your whole screen in Zoom or Meet with a second device in the call, trigger an alert, and check that the second device does not see it.

**Live transcription.** Speak and watch the text appear. Press Ctrl+C to stop.

```bash
python -m scripts.live_transcribe
```

**See how a document is chunked.** Add `--context` to include the Gemini context for each chunk.

```bash
python -m scripts.show_chunks samples/sample_report.txt
python -m scripts.show_chunks samples/sample_report.txt --context
```

**Ask questions about a document** and see the best matching chunks. Add `--facts` to also show the facts table rows of each result (needs the Gemini key).

```bash
python -m scripts.search samples/sample_report.txt
python -m scripts.search samples/sample_report.txt --mode late --facts
```

| Mode | How chunks are embedded | Needs Gemini key |
|---|---|---|
| `plain` | Each chunk on its own | No |
| `contextual` | Gemini context line + chunk | Yes |
| `late` | Late chunking: whole document embedded once, then cut into chunks | No |

The first run downloads the embedding model (`nomic-ai/modernbert-embed-base`, about 600 MB). In late mode, a document longer than the model's 8,192-token limit is embedded in windows of neighbouring chunks.

**Measure search quality** for every mode and method (vector, keyword, hybrid):

```bash
python -m scripts.evaluate samples/sample_report.txt samples/eval_questions.json
```

Results on the sample document (32 questions, hybrid search):

| Mode | hit@1 | hit@3 | MRR |
|---|---|---|---|
| `plain` | 0.94 | 0.97 | 0.96 |
| `contextual` | 0.84 | 0.97 | 0.90 |
| `late` | 0.75 | 0.97 | 0.85 |

The sample has only 9 chunks, so treat these numbers as a smoke test, not a benchmark. That is why `plain` is the default for now.

## Notes

- Free API tiers may use your data to improve their models. Test with public or sample documents, not confidential ones.
- `samples/sample_report.txt` is a fictional document made for testing. The `.pdf`, `.docx` and `.pptx` versions are made from it by `python -m scripts.make_sample_documents`.
- Generated files (test documents, the cache) go in `data/`, which is not committed. Delete `data/cache` to rebuild everything.

## Project layout

```
app/
  stt.py            Deepgram live speech-to-text
  mic.py            Microphone input
  documents.py      Reads .pdf, .docx, .pptx, .txt and .md files
  chunker.py        Section-aware chunking
  llm.py            LLM setup (Gemini, Groq, rate limits, errors)
  contextualizer.py Contextual retrieval
  embedder.py       Embeddings (normal and late chunking)
  index.py          Builds and caches a document's chunks and vectors
  retriever.py      Hybrid search (vectors + BM25, merged with RRF)
  facts.py          Facts table
  cache.py          JSON cache in data/cache
  prepare.py        Everything done with a document before the meeting
  fact_checker.py   Checks one spoken claim against the document
  copilot.py        Decision engine (LangGraph): silent, alert or suggest
  voice.py          Whispers (text-to-speech to your earphones)
  screen.py         Live screen (local web server)
  overlay.py        Meeting overlay on macOS: floating pop-up window, hidden from screen sharing
  overlay_tk.py     Meeting overlay on Windows and Linux
  errors.py         PreCue's error types
web/
  index.html        The live screen page
  overlay.html      The pop-up page shown in the overlay
tests/              Offline tests (fake LLM, search and Deepgram)
scripts/
  live_copilot.py   The main app
  check_claims.py
  live_transcribe.py
  show_chunks.py
  search.py
  evaluate.py
  make_sample_documents.py
samples/
  sample_report.txt (.pdf, .docx, .pptx)
  eval_questions.json
  eval_claims.json
```

## License

[MIT](LICENSE)
