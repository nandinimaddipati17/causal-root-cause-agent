import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "In one sentence, what is root cause analysis?"}
    ],
)

print(response["message"]["content"])