from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 1. Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ensure there is a pad_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. Create a chat pipeline
chat = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,      # only the assistant's reply
)

def get_element_cures(elements: str) -> list[str]:
    # 3. System prompt asks for a COMMA-SEPARATED list of cures
    system_prompt = (
        "You are a medical-chemistry assistant.\n"
        "When given two or more chemical elements, identify which diseases or medical conditions "
        "a drug containing those elements (along with its other components) can treat.\n"
        "Important:\n"
        "• Output ONLY a plain paragraph of conditions or treatments, one per line.\n"
        "• Do NOT mention element names, chemical terms, side-effects, explanations, or ask questions."
    )

    # user_prompt = f"{elements}\ngive me a paragraph of only the treatments, nothing more."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": elements},
    ]
    # 4. Generate
    response = chat(messages, max_new_tokens=512, temperature=0.9, top_p=0.9)
    text = response[0]["generated_text"].strip()
    
    # 5. Split on commas and strip whitespace
    return text

if __name__ == "__main__":
    cures = get_element_cures("Hydrogen + oxygen")
    print(cures)
    # e.g. ["Wound disinfection"]
