from llama_cpp import Llama

# 1) Initialize once at startup
llm = Llama(
    model_path="~/models/TinyLlama/tinyllama-4bit.bin",  # path to your quantized GGML model
    n_ctx=512,       # max context tokens
    n_threads=2,     # leverage both CPU cores
)

def generate_text(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """
    Generate text with TinyLlama via llama.cpp.

    Args:
        prompt: the user’s input string
        max_tokens: max tokens to generate
        temperature: randomness control (0.0–1.0)
        top_p: nucleus sampling threshold

    Returns:
        The generated text (stripped of leading/trailing whitespace).
    """
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["\n"]
    )
    # Extract the first choice’s text
    text = response.get("choices", [{}])[0].get("text", "")
    return text.strip()

# --- Example usage ---
if __name__ == "__main__":
    user_query = "Explain the benefits of lightweight LLMs on edge devices."
    ai_reply = generate_text(user_query, max_tokens=150)
    print("AI:", ai_reply)
