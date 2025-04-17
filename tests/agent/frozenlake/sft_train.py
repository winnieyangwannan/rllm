import json
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from typing import Dict, List, Any
import torch

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_messages(messages, tokenizer):
    """
    Creates masks for tokens to focus on assistant responses and tool calls.
    
    Args:
        messages: List of message dictionaries in the chat format
        tokenizer: The tokenizer used for the model
        
    Returns:
        tuple: (text, mask) where mask is 1 for assistant content/tool calls
               and 0 for system/user/tool outputs
    """    
    all_tokens = []
    all_masks = []

    for idx, msg in enumerate(messages):

        msg_text = tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=False
        )

        target = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        if msg_text.startswith(target):
            msg_text = msg_text[len(target):]

        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        all_tokens.extend(msg_tokens)
        all_masks.extend(msg_mask)

    return all_tokens, all_masks
    

class TrajectoryDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract messages from each example
        batch_messages = [feature['messages'] for feature in features]
        
        # Process each example
        batch_tokens = []
        batch_attention_masks = []
        batch_labels = []
        
        max_length = 0
        for messages in batch_messages:
            tokens, loss_mask = preprocess_messages(messages, self.tokenizer)
            batch_tokens.append(tokens)
            # Create attention mask (1 for all non-padding tokens)
            attention_mask = [1] * len(tokens)
            batch_attention_masks.append(attention_mask)
            # Use loss_mask to determine which tokens to include in loss calculation
            labels = [-100 if mask == 0 else token for token, mask in zip(tokens, loss_mask)]
            batch_labels.append(labels)
            max_length = max(max_length, len(tokens))
            
        # Pad all sequences to max_length
        for i in range(len(batch_tokens)):
            padding_length = max_length - len(batch_tokens[i])
            if padding_length > 0:
                batch_tokens[i].extend([self.tokenizer.pad_token_id] * padding_length)
                batch_attention_masks[i].extend([0] * padding_length)
                batch_labels[i].extend([-100] * padding_length)
        
        return {
            "input_ids": torch.tensor(batch_tokens),
            "attention_mask": torch.tensor(batch_attention_masks),
            "labels": torch.tensor(batch_labels)
        }

def prepare_training_dataset(data):
    # Convert the data to the format expected by datasets
    dataset_dict = {
        'messages': data,
    }
    return Dataset.from_dict(dataset_dict).shuffle(42)


def main():
    json_path = "./sft_trajectory.json"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_output_dir = "./sft_model_output/"
    os.makedirs(model_output_dir, exist_ok=True)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

   
    # Convert to Hugging Face dataset & tokenize
    raw_data = load_data(json_path)
    dataset = prepare_training_dataset(raw_data)
    data_collator = TrajectoryDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=5,
        learning_rate=1e-5,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_dir="./logs",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        save_steps=10,
        bf16=True,
        save_only_model=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(model_output_dir)
    print(f"Model saved to {model_output_dir}")

    # save tokenizer to the folder with finetuned model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save tokenizer in the same directory as the model
    tokenizer.save_pretrained(model_output_dir)

    print(f"Tokenizer saved to {model_output_dir}")

if __name__ == "__main__":
    main()
