"""Simple calculator tool for mathematical operations.

This tool provides basic mathematical operations that can be used by the StrandsAgent.

Note: strands_tools do have a calculator tool that you can import, this is a demo of how you can build ur own strands tool by @tool decogration
"""

import re
from strands import tool
from typing import Union, Optional


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression as a string (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation as a string
        
    Examples:
        - "2 + 3" -> "5"
        - "10 / 2" -> "5.0"
        - "2^3" -> "8"
        - "sqrt(16)" -> "4.0"
    """
    try:
        # Clean the expression
        expression = expression.strip()
        
        # Replace common mathematical notations
        expression = expression.replace('^', '**')  # Power
        expression = expression.replace('×', '*')   # Multiplication
        expression = expression.replace('÷', '/')   # Division
        
        # Handle square root
        sqrt_match = re.search(r'sqrt\(([^)]+)\)', expression)
        if sqrt_match:
            inner_expr = sqrt_match.group(1)
            inner_result = eval(inner_expr)
            if inner_result < 0:
                return "Error: Cannot take square root of negative number"
            result = inner_result ** 0.5
            return str(result)
        
        # Handle basic operations
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        
        # Check for division by zero
        if '/0' in expression.replace('/0.', ''):
            return "Error: Division by zero"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Format the result
        if isinstance(result, int):
            return str(result)
        elif isinstance(result, float):
            # Round to reasonable precision
            if result.is_integer():
                return str(int(result))
            else:
                return f"{result:.6f}".rstrip('0').rstrip('.')
        else:
            return str(result)
            
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid mathematical expression"
    except NameError:
        return "Error: Expression contains invalid names or functions"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def simple_calc(operation: str, a: Union[int, float], b: Union[int, float]) -> str:
    """
    Perform a simple mathematical operation between two numbers.
    
    Args:
        operation: The operation to perform ('+', '-', '*', '/', '^')
        a: First number
        b: Second number
        
    Returns:
        The result of the operation as a string
    """
    try:
        if operation == '+':
            result = a + b
        elif operation == '-':
            result = a - b
        elif operation == '*':
            result = a * b
        elif operation == '/':
            if b == 0:
                return "Error: Division by zero"
            result = a / b
        elif operation == '^':
            result = a ** b
        else:
            return f"Error: Unknown operation '{operation}'. Supported: +, -, *, /, ^"
        
        # Format the result
        if isinstance(result, int):
            return str(result)
        elif isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            else:
                return f"{result:.6f}".rstrip('0').rstrip('.')
        else:
            return str(result)
            
    except Exception as e:
        return f"Error: {str(e)}"


# For backward compatibility, create an alias
calculator_tool = calculator
