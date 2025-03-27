from datasets import Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any
import torch

from transformers import Trainer, TrainingArguments

from rllm.tools import PythonInterpreter


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def preprocess_messages(messages, tokenizer, use_tools=True):
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

    skip_assistant_token = False
    for idx, msg in enumerate(messages):
        if idx == 0:
            add_generation_prompt = True
        else:
            add_generation_prompt = False

        if msg['role'] == "assistant":
            msg['skip_assistant_token'] = skip_assistant_token
            skip_assistant_token = True

        msg_text = tokenizer.apply_chat_template(
            [msg], tools=[PythonInterpreter.info] if use_tools else [], tokenize=False, add_generation_prompt=add_generation_prompt
        )
        
        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        all_tokens.extend(msg_tokens)
        all_masks.extend(msg_mask)

    # Print tokens where mask is 1 (assistant content/tool calls)
    # masked_tokens = [token for token, mask in zip(all_tokens, all_masks)]
    # Check for consecutive Assistant tokens
    # decoded = tokenizer.decode(all_tokens)
    # print("all tokens:", decoded)
    # import pdb; pdb.set_trace()

    return all_tokens, all_masks
    

def find_token_sequence(full_tokens, seq_tokens):
    """Helper function to find a sequence of tokens within a larger sequence"""
    n = len(full_tokens)
    m = len(seq_tokens)
    for i in range(n - m + 1):
        if full_tokens[i:i+m] == seq_tokens:
            return i
    return -1


class ToolCallDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract messages from each example
        batch_messages = [feature['messages'] for feature in features]
        batch_use_tools = [feature['use_tools'] for feature in features]
        
        # Process each example
        batch_tokens = []
        batch_attention_masks = []
        batch_labels = []
        
        max_length = 0
        for messages, use_tools in zip(batch_messages, batch_use_tools):
            tokens, loss_mask = preprocess_messages(messages, self.tokenizer, use_tools)
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
        'messages': [item['messages'] for item in data],
        'use_tools': [item['use_tools'] if 'use_tools' in item else True for item in data]
    }
    return Dataset.from_dict(dataset_dict).shuffle(42)


def main(args):
    # Load and prepare the dataset
    raw_data = load_jsonl(args.data_path)
    dataset = prepare_training_dataset(raw_data)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="no",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        save_steps=10,
        bf16=True,
        save_only_model=True,
        deepspeed=args.deepspeed,
        save_total_limit=2
        # fp16=True,
    )

    with open(args.chat_template, "r") as f:
        template = f.read()
        tokenizer.chat_template = template
        
    # Initialize data collator
    data_collator = ToolCallDataCollator(tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(args.model_output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model with tool call data')
    parser.add_argument('--data_path', type=str, 
                      default='./data/filtered_toolcall_claude_verified.jsonl',
                      help='Path to the JSONL data file')
    parser.add_argument('--model_path', type=str,
                      default='agentica-org/DeepScaleR-1.5B-Preview',
                      help='Path or name of the pretrained model to fine-tune')
    parser.add_argument('--output_dir', type=str,
                      default='./results',
                      help='Directory for training outputs and checkpoints')
    parser.add_argument('--model_output_dir', type=str,
                      default='./deepscaler-toolcall-claude-python',
                      help='Directory to save the final model')
    parser.add_argument('--chat_template', type=str,
                      default='../../rllm/templates/r1-toolcall-python.jinja',
                      help='Path to the chat template file')
    parser.add_argument('--deepspeed', type=str,
                      default='../config/ds_stage2.json',
                      help='Path to the deepspeed config file')
    args = parser.parse_args()
   
    main(args)