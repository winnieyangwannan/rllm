from transformers import AutoTokenizer

# model = "Qwen/Qwen2.5-1.5B"
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

# Define multi-turn chat
messages = [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am Qwen."},
    {"role": "user", "content": "What can you do?"},
    {"role": "assistant", "content": "I can answer your questions."}
]

def _postprocess_model_chat_template(message_text):
    if any(substring in model.lower() for substring in ('qwen2', 'qwen2.5')):
        # from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json, a default system message is inserted. So we manually remove the first occurance of default system message.
        # This is currently assuming no tool call.
        target = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        if message_text.startswith(target):
            return message_text[len(target):]  # Remove only if it’s at the start
        target = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        if message_text.startswith(target):
            return message_text[len(target):]  # Remove only if it’s at the start
        return message_text

    return message_text

# Step 1: Tokenize the full chat at once
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
# full_text = _postprocess_model_chat_template(full_text)
full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

# Step 2: Tokenize individual messages with chat template, then concatenate
individual_ids = []
for i in range(len(messages)):
    partial = tokenizer.apply_chat_template([messages[i]], tokenize=False, add_generation_prompt=False)
    # partial = _postprocess_model_chat_template(partial)
    tokens = tokenizer(partial, add_special_tokens=False).input_ids
    individual_ids.extend(tokens)

# Step 3: Compare
is_equal = full_ids == individual_ids
print("Tokenizing individually and concatenating is equal to tokenizing full conversation:", is_equal)

if not is_equal:
    print("Mismatch details:")
    min_len = min(len(full_ids), len(individual_ids))
    for i in range(min_len):
        if full_ids[i] != individual_ids[i]:
            print(f"  At index {i}: full={full_ids[i]}, individual={individual_ids[i]}")
            break
    print(f"→ Lengths: full={len(full_ids)}, individual={len(individual_ids)}")

# Step 4: (Optional) Decode for sanity
print("\nDecoded full:")
print(repr(tokenizer.decode(full_ids)))

print("\nDecoded individual concat:")
print(repr(tokenizer.decode(individual_ids)))
