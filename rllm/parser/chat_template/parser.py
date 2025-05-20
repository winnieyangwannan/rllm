import json

class ChatTemplateParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = ""

    def parse(self, messages, add_generation_prompt=False, **kwargs):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    @classmethod
    def get_parser(cls, tokenizer, enable_thinking=False):
        """Factory method to get the appropriate parser based on a string identifier.
        
        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            enable_thinking: Whether generation prompt will enable thinking.
            
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
            elif "qwen" in model_name or 'r2egym' in model_name:
                parser = QwenChatTemplateParser(tokenizer, enable_thinking=enable_thinking)
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
        self.generation_prompt = self.assistant_token

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
        result = ''

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message['content']
    
    def parse_user(self, message):
        return self.user_token + message['content']
    
    def parse_assistant(self, message):
        return self.assistant_token + message['content'] #+ self.eos_token
    

class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, enable_thinking=True):
        super().__init__(tokenizer)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = '<|im_end|>\n'
        self.system_token = '<|im_start|>system\n'
        self.user_token = '<|im_start|>user\n'
        self.assistant_token = '<|im_start|>assistant\n'
        self.generation_prompt = self.assistant_token
        if enable_thinking:
            self.generation_prompt += '<think>\n'
        
        self.tool_start_token = "\n<tool_call>\n"
        self.tool_end_token = "\n</tool_call>"
        
        self.tool_response_start_token = '<tool_response>\n'
        self.tool_response_end_token = '\n</tool_response>'
        # enable_thinking only adds thinking for generation, not when transforming assistant messages
        if enable_thinking:
            print(f"Thinking is enabled, the required context will be larger") 

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
        result = ''

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + self.eot_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message['role'] == 'tool':
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message['content'] + self.eot_token
    
    def parse_user(self, message):
        return self.user_token + message['content'] + self.eot_token
    
    def parse_assistant(self, message):
        result = self.assistant_token + message['content']
        if 'tool_calls' in message:
            for tool_call in message['tool_calls']:
                tool_json = {
                    "name": tool_call['function']['name'],
                    "arguments": tool_call['function']['arguments']
                }
                arguments = tool_call['function']['arguments']
                if isinstance(arguments, str):
                    try:
                        # If arguments is already a JSON string, parse it to avoid double encoding
                        parsed_args = json.loads(arguments)
                        tool_json["arguments"] = parsed_args
                    except json.JSONDecodeError:
                        # If not valid JSON, keep as is
                        pass
                
                result += self.tool_start_token + json.dumps(tool_json, ensure_ascii=False) + self.tool_end_token
        result += self.eot_token
        return result
    
    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message['content'] + self.tool_response_end_token + self.eot_token