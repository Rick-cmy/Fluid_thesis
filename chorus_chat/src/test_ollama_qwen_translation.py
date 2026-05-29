from chorus_mvp.llm import DEFAULT_OLLAMA_MODEL, call_ollama_chat_text


def build_messages(source_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": """You are a professional Swedish-to-English translator.

Rules:
- Translate every part of the source text.
- Preserve meaning, tone, terminology, and paragraph structure.
- Do not summarize.
- Do not explain.
- Do not add commentary.
- Return English only.
""",
        },
        {
            "role": "user",
            "content": f"""Translate the following Swedish text into English.

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>""",
        },
    ]


def main() -> None:
    source_text = "Jag skulle vilja boka ett möte nästa vecka."
    translation = call_ollama_chat_text(
        messages=build_messages(source_text),
        model=DEFAULT_OLLAMA_MODEL,
    )

    print("\n=== Local Ollama model ===")
    print(DEFAULT_OLLAMA_MODEL)

    print("\n=== Source ===")
    print(source_text)

    print("\n=== Translation ===")
    print(translation)


if __name__ == "__main__":
    main()
