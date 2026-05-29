import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from llm.ollama_client import OllamaClient


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")
    return text


def build_messages(source_text: str, source_lang: str, target_lang: str):
    system_prompt = f"""You are a professional literary translator.

Your task:
- Translate the text from {source_lang} to {target_lang}.
- Translate EVERY part of the source text.
- Preserve meaning, tone, narrative content, and paragraph structure.
- Keep names and proper nouns consistent.
- Do NOT summarize.
- Do NOT explain.
- Do NOT answer questions.
- Do NOT add commentary.
- Do NOT omit content.
- Output {target_lang} only.

If the input contains a title or chapter heading, translate it too.
Return only the translation of the text inside <SOURCE_TEXT> ... </SOURCE_TEXT>.
"""

    user_prompt = f"""Translate the following text from {source_lang} to {target_lang}.

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input/source.txt")
    parser.add_argument("--output", default="data/logs/single_agent_run.json")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="Chinese")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_text = load_text(input_path)
    messages = build_messages(source_text, args.source_lang, args.target_lang)

    client = OllamaClient(model=args.model)
    translation, raw_response = client.chat(
        messages=messages,
        temperature=args.temperature,
    )

    result = {
        "run_type": "single_agent_baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "input_path": str(input_path),
        "temperature": args.temperature,
        "messages": messages,
        "translation": translation.strip(),
        "raw_response": raw_response,
    }

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved JSON log to: {output_path}")
    print("\nTranslation preview:\n")
    print(result["translation"][:1000])


if __name__ == "__main__":
    main()
