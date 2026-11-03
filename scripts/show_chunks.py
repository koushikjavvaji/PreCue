"""Load a text file, chunk it, and print the chunks so you can check them.

Run from the project folder:
  python -m scripts.show_chunks samples/sample_report.txt
  python -m scripts.show_chunks samples/sample_report.txt --context   (adds Gemini context)
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from app.chunker import load_text, make_chunks
from app.errors import PreCueError


def main():
    args = sys.argv[1:]
    with_context = "--context" in args
    files = [arg for arg in args if arg != "--context"]
    if len(files) != 1:
        raise SystemExit("Usage: python -m scripts.show_chunks <file.txt> [--context]")

    path = Path(files[0])
    text = load_text(path)
    chunks = make_chunks(text, source=path.name)

    if with_context:
        from app.contextualizer import add_context
        from app.llm import get_gemini

        load_dotenv()
        llm = get_gemini()
        start = time.time()
        chunks = add_context(chunks, text, llm, model_name=llm.model)
        print(f"Context added with {llm.model} in {time.time() - start:.1f}s")

    for chunk in chunks:
        info = chunk.metadata
        print(f"\n===== chunk {info['chunk']} | section: {info['section'] or '-'} "
              f"| {len(chunk.page_content)} chars =====")
        print(chunk.page_content)

    print(f"\nTotal: {len(chunks)} chunks from {len(text)} characters")


if __name__ == "__main__":
    try:
        main()
    except PreCueError as error:
        raise SystemExit(f"Error: {error}")
