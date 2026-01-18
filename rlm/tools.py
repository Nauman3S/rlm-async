"""Tool definitions for RLM function calling"""

from typing import Any

# Tool definitions in OpenAI function calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "peek",
            "description": "View a character range of the context. Use this to inspect specific portions of the text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "integer",
                        "description": "Start character index (0-indexed)"
                    },
                    "end": {
                        "type": "integer",
                        "description": "End character index (exclusive)"
                    }
                },
                "required": ["start", "end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "peek_lines",
            "description": "View a range of lines from the context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "integer",
                        "description": "Start line number (0-indexed)"
                    },
                    "end": {
                        "type": "integer",
                        "description": "End line number (exclusive)"
                    }
                },
                "required": ["start", "end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the context with a regex pattern. Returns matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 50)",
                        "default": 50
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "chunk",
            "description": "Get chunk boundaries for splitting context into pieces for parallel processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_size": {
                        "type": "integer",
                        "description": "Approximate size of each chunk in characters"
                    }
                },
                "required": ["chunk_size"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sub_llm",
            "description": "Recursively call the LLM on a slice of the context. Use this to process large contexts in parallel chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or instruction for the sub-LLM"
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start character index of context slice"
                    },
                    "end": {
                        "type": "integer",
                        "description": "End character index of context slice"
                    }
                },
                "required": ["query", "start", "end"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "exec_python",
            "description": "Execute Python code. Has access to 'context' (str), 'lines' (list), 'variables' (dict), and 're' module.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_variable",
            "description": "Store an intermediate result in a named variable for later use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Variable name"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to store"
                    }
                },
                "required": ["name", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_variable",
            "description": "Retrieve a previously stored variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Variable name to retrieve"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final",
            "description": "Return the final answer to the user's question. Call this when you have determined the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer"
                    }
                },
                "required": ["answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_var",
            "description": "Return a stored variable as the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Variable name containing the final answer"
                    }
                },
                "required": ["name"]
            }
        }
    }
]


def execute_tool(sandbox: Any, tool_name: str, args: dict[str, Any]) -> str:
    """
    Execute a tool call on the sandbox.

    Args:
        sandbox: Sandbox instance
        tool_name: Name of the tool to execute
        args: Tool arguments

    Returns:
        String result of the tool execution
    """
    if tool_name == "peek":
        return sandbox.peek(args["start"], args["end"])

    elif tool_name == "peek_lines":
        return sandbox.peek_lines(args["start"], args["end"])

    elif tool_name == "grep":
        results = sandbox.grep(args["pattern"], args.get("max_results", 50))
        if not results:
            return "No matches found"
        return "\n".join(f"{line_num}: {line}" for line_num, line in results)

    elif tool_name == "chunk":
        chunks = sandbox.chunk(args["chunk_size"])
        return str(chunks)

    elif tool_name == "exec_python":
        return sandbox.exec_python(args["code"])

    elif tool_name == "set_variable":
        sandbox.set_var(args["name"], args["value"])
        return f"Variable '{args['name']}' set"

    elif tool_name == "get_variable":
        value = sandbox.get_var(args["name"])
        if value is None:
            return f"Variable '{args['name']}' not found"
        return str(value)

    elif tool_name == "final":
        return args["answer"]

    elif tool_name == "final_var":
        value = sandbox.get_var(args["name"])
        if value is None:
            return f"Variable '{args['name']}' not found"
        return str(value)

    else:
        return f"Unknown tool: {tool_name}"
