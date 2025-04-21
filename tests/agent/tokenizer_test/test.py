from transformers import AutoTokenizer

# Load tokenizer (Qwen or any other model with chat_template)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

# Test messages
messages = [
    {"role": "user", "content": "What is your name?"},
    {"role": "assistant", "content": "I am Qwen."},
    {"role": "user", "content": "What can you do?"},
    {"role": "assistant", "content": "I can answer your questions."}
]

# 1. Apply chat template for generation-time prompt of B
gen_prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
print(f"Generation time prompt after chat template: {repr(gen_prompt)}")
gen_prompt_ids = tokenizer(gen_prompt, add_special_tokens=False).input_ids

# 2. Apply chat template for full conversation [A, B, C, D]
full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(f"Full prompt after chat template: {repr(full_prompt)}")
full_prompt_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids

# 3. Compare
is_prefix = full_prompt_ids[:len(gen_prompt_ids)] == gen_prompt_ids

print("Generation-time prompt is prefix of full prompt:", is_prefix)
if not is_prefix:
    print("Mismatch at index:")
    for i, (a, b) in enumerate(zip(full_prompt_ids, gen_prompt_ids)):
        if a != b:
            print(f"  At token {i}: full={a}, gen={b}")
            break

print("\nDecoded generation-time prompt:")
print(repr(tokenizer.decode(gen_prompt_ids)))

print("\nDecoded full prompt prefix:")
print(repr(tokenizer.decode(full_prompt_ids[:len(gen_prompt_ids)])))
