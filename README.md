# PreCue

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
- **Section-aware chunking** of `.txt` and `.md` documents
- **Contextual retrieval** (an LLM writes a short context for each chunk) using Gemini
- **Embeddings** with a local model, in three modes: `plain`, `contextual` and `late` (late chunking, also for long documents)
- **Hybrid search**: vector search plus BM25 keyword search for questions with numbers, merged with Reciprocal Rank Fusion
- **Facts table**: every number-based fact in the document, extracted by Gemini and checked against the text
- **Saved index**: chunks and vectors are cached, so a document is only processed once
- **Evaluation**: a question set and a script that scores every mode and search method
- **LangSmith tracing** (optional)

Coming next: live fact-checking (Groq + LangGraph), PDF support, text-to-speech whispers.

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

Never commit `.env`. It is already in `.gitignore`.

Optional settings in `.env`:

- `GEMINI_MODEL`: the Gemini model to use (default `gemini-3.5-flash-lite`)
- `GEMINI_REQUESTS_PER_MINUTE`: how fast to call Gemini (default `15`, the free-tier limit). Calls are spaced out automatically.
- `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`: send traces of every LLM call and search to [LangSmith](https://smith.langchain.com). Off by default. Traces include document text, so use test documents.

## Usage

Run all commands from the project folder.

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
- `samples/sample_report.txt` is a fictional document made for testing.
- Generated files (test documents, the cache) go in `data/`, which is not committed. Delete `data/cache` to rebuild everything.

## Project layout

```
app/
  stt.py            Deepgram live speech-to-text
  mic.py            Microphone input
  chunker.py        Section-aware chunking
  llm.py            LLM setup (Gemini, rate limit, errors)
  contextualizer.py Contextual retrieval
  embedder.py       Embeddings (normal and late chunking)
  index.py          Builds and caches a document's chunks and vectors
  retriever.py      Hybrid search (vectors + BM25, merged with RRF)
  facts.py          Facts table
  cache.py          JSON cache in data/cache
  errors.py         PreCue's error types
scripts/
  live_transcribe.py
  show_chunks.py
  search.py
  evaluate.py
samples/
  sample_report.txt
  eval_questions.json
```

## License

TBD
