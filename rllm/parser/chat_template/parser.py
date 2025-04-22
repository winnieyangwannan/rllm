class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def parse(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    @classmethod
    def get_parser(cls, tokenizer):
        """Factory method to get the appropriate parser based on a string identifier.
        
        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            
        Returns:
            ChatTemplateParser: An instance of the requested parser
            
        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            if "deepseek" in model_name and "qwen" in model_name:
                parser = DeepseekQwenChatTemplateParser(tokenizer)
                print(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return parser
            elif "qwen" in model_name:
                parser = QwenChatTemplateParser(tokenizer)
                print(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return parser
        
        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer)
        print(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        return parser
    

class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ''
        self.user_token = '<｜User｜>'
        self.assistant_token = '<｜Assistant｜>'

    def parse(self, messages):
        result = ''
        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")
        return result

    def parse_system(self, message):
        return self.system_token + message['content']
    
    def parse_user(self, message):
        return self.user_token + message['content']
    
    def parse_assistant(self, message):
        return self.assistant_token + message['content'] + self.eos_token
    

class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = '<|im_start|>system\n'
        self.user_token = '<|im_start|>user\n'
        self.assistant_token = '<|im_start|>assistant\n'
        self.eot_token = '<|im_end|>\n'

    def parse(self, messages):
        result = ''
        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")
        return result

    def parse_system(self, message):
        return self.system_token + message['content'] + self.eot_token
    
    def parse_user(self, message):
        return self.user_token + message['content'] + self.eot_token
    
    def parse_assistant(self, message):
        return self.assistant_token + message['content'] + self.eot_token
