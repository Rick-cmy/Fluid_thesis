from llm.ollama_client import OllamaClient

client = OllamaClient(model="llama3.2:1b")

text, raw = client.chat(
    [
        {"role": "system", "content": "You are a precise assistant."},
        {"role": "user", "content": "Reply with exactly OK"},
    ],
    temperature=0.0,
)

print(text.strip())
