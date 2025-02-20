class Calculator:
    """A tool for evaluating mathematical expressions safely."""

    INFO = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions, prefer using this instead of calculating it yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }

    SYMBOL_REPLACEMENTS = {
        "^": "**",  # Power
        "×": "*",   # Multiplication 
        "÷": "/"    # Division
    }

    ALLOWED_CHARS = set("0123456789.+-*/() **")

    async def execute(self, *args):
        """Execute the calculator with given arguments."""
        return await self._evaluate_expression(*args)

    async def _evaluate_expression(self, expression: str) -> str:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Result as string, or error message if evaluation fails
        """
        try:
            # Replace mathematical symbols with Python operators
            for old, new in self.SYMBOL_REPLACEMENTS.items():
                expression = expression.replace(old, new)
            
            # Validate characters
            if not all(c in self.ALLOWED_CHARS for c in expression):
                return "Error: Invalid characters in expression"
            
            # Evaluate with empty namespace for safety
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
            
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"