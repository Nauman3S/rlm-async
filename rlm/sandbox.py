"""REPL sandbox environment for RLM"""

import re
import io
import sys
from typing import Any


class Sandbox:
    """
    REPL environment that stores context and provides tools for inspection.
    The context is never sent directly to the LLM - only accessed via tools.
    """

    def __init__(self, context: str):
        """
        Initialize sandbox with context.

        Args:
            context: The full text context to be processed
        """
        self.context = context
        self.variables: dict[str, Any] = {}
        self._lines: list[str] | None = None

    @property
    def lines(self) -> list[str]:
        """Lazy-load lines from context"""
        if self._lines is None:
            self._lines = self.context.split("\n")
        return self._lines

    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the context without exposing the content.

        Returns:
            Dict with context_length and line_count
        """
        return {
            "context_length": len(self.context),
            "line_count": len(self.lines),
        }

    def peek(self, start: int, end: int) -> str:
        """
        View a character slice of the context.

        Args:
            start: Start character index
            end: End character index

        Returns:
            The substring from start to end
        """
        return self.context[start:end]

    def peek_lines(self, start: int, end: int) -> str:
        """
        View a line range of the context.

        Args:
            start: Start line number (0-indexed)
            end: End line number (exclusive)

        Returns:
            Lines joined with newlines
        """
        return "\n".join(self.lines[start:end])

    def grep(self, pattern: str, max_results: int = 50) -> list[tuple[int, str]]:
        """
        Search context with regex pattern.

        Args:
            pattern: Regex pattern to search for
            max_results: Maximum number of results to return

        Returns:
            List of (line_number, line_content) tuples
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [(0, f"Invalid regex: {e}")]

        results = []
        for i, line in enumerate(self.lines):
            if regex.search(line):
                results.append((i, line))
                if len(results) >= max_results:
                    break
        return results

    def chunk(self, chunk_size: int) -> list[tuple[int, int]]:
        """
        Get chunk boundaries for parallel processing.

        Args:
            chunk_size: Approximate size of each chunk in characters

        Returns:
            List of (start, end) character indices
        """
        chunks = []
        total = len(self.context)
        start = 0
        while start < total:
            end = min(start + chunk_size, total)
            # Try to break at newline
            if end < total:
                newline_pos = self.context.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos + 1
            chunks.append((start, end))
            start = end
        return chunks

    def exec_python(self, code: str) -> str:
        """
        Execute Python code with access to context and variables.

        Args:
            code: Python code to execute

        Returns:
            Captured stdout or repr of last expression
        """
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        local_vars = {
            "context": self.context,
            "lines": self.lines,
            "variables": self.variables,
            "re": re,
        }

        try:
            # Try to exec, if it's an expression, eval it
            try:
                result = eval(code, {"__builtins__": __builtins__}, local_vars)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                exec(code, {"__builtins__": __builtins__}, local_vars)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
        finally:
            sys.stdout = old_stdout

        return captured.getvalue().strip()

    def set_var(self, name: str, value: Any) -> None:
        """Store a variable"""
        self.variables[name] = value

    def get_var(self, name: str) -> Any:
        """Retrieve a variable"""
        return self.variables.get(name)

    def list_vars(self) -> list[str]:
        """List all stored variable names"""
        return list(self.variables.keys())
