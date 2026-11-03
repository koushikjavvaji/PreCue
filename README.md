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

Early stage. The architecture and stack are still being decided.

## License

TBD
