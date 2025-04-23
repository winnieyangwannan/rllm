from typing import List, Dict, Tuple, Union, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import logging
from rllm.parser.chat_template.parser import ChatTemplateParser

def get_recent_assistant_user_messages(chat_completions_messages):
    """
    Extracts the most recent assistant message and environment messages (user/tool) from a chat completions list.
    
    Args:
        chat_completions_messages (List[Dict]): List of message dictionaries from chat completions.
        
    Returns:
        Tuple[Dict, List[Dict]]: A tuple containing:
            - The most recent assistant message (or None if not found)
            - A list of environment messages (user/tool) that occurred after the last assistant message,
              in chronological order.
    """
    # Loop backwards to get the last assistant message and environment messages
    env_messages = []
    assistant_message = None
    seen_assistant_message = False
    for message in reversed(chat_completions_messages):
        role = message.get('role', None)
        if role == 'assistant':
            if assistant_message:
                break
            seen_assistant_message = True
            assistant_message = message
        elif role in ['user', 'tool'] and not seen_assistant_message:
            env_messages.append(message)
    # Reverse the env_messages to maintain chronological order
    env_messages = list(reversed(env_messages))
    
    return assistant_message, env_messages

def convert_messages_to_tokens_and_masks(messages: List[Dict[str, str]], tokenizer: AutoTokenizer, parser: ChatTemplateParser, contains_first_msg=False, contains_generation_msg=False):
    """
    Converts multiple messages to tokens and masks.
    contains_first_msg flag and contains_generaiton_msg flag are used to indicate whether the conversation is for beginning or contains the generation.
    The first and last message is assumed to be the special message respectively
    
    Args:
        messages (List[Dict]): The messages to convert.
        tokenizer: The tokenizer to use.
        contains_first_msg (bool): Whether the first message is a special message.
        contains_generation_msg (bool): Whether the last message is a special message.
        
    Returns:
        Tuple[List[int], List[int]]: A tuple containing all tokens and all masks.
    """
    all_msg_tokens = []
    all_msg_masks = []

    def _convert_message_to_tokens_and_masks(msg, first_msg=False, generation_msg=False):
        msg_text = parser.parse([msg], add_generation_prompt=generation_msg, is_first_msg=first_msg)
        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)

        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        #TODO: make this adhere to each tokenizer
        # need some additional masking like overlap token which should came from prompt's add_generation_prompt
        if mask_value == 1:
            # Mask out the assistant token at the beginning
            assistant_token = parser.assistant_token
            assistant_token_ids = tokenizer.encode(assistant_token, add_special_tokens=False)

            # Assert that the message start with the assistant token
            for i in range(min(len(assistant_token_ids), len(msg_tokens))):
                assert msg_tokens[i] == assistant_token_ids[i], f"Expected token {assistant_token_ids[i]} but got {msg_tokens[i]}"
            
            # Remove assistant token not from generation
            msg_mask = msg_mask[len(assistant_token_ids):]
            msg_tokens = msg_tokens[len(assistant_token_ids):]
            # NOTE: new template does not add eos so no check for that

        return msg_tokens, msg_mask

    for i, msg in enumerate(messages):
        msg_tokens, msg_mask = _convert_message_to_tokens_and_masks(
            msg, 
            first_msg=(contains_first_msg and i == 0), 
            generation_msg=(contains_generation_msg and i == len(messages) - 1)
        )
        all_msg_tokens.extend(msg_tokens)
        all_msg_masks.extend(msg_mask)

    return all_msg_tokens, all_msg_masks

