"""Main RLM orchestrator with async parallel sub_llm support"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .client import get_async_client, DEFAULT_MODEL
from .sandbox import Sandbox
from .tools import TOOLS, execute_tool
from .prompts import format_prompt


@dataclass
class TraceNode:
    """A node in the execution trace tree"""
    name: str
    is_async: bool
    depth: int
    children: list = field(default_factory=list)
    result_preview: str = ""
    duration_ms: float = 0
    parallel_group: int = -1  # -1 means not in a parallel group

    def add_child(self, child: "TraceNode"):
        self.children.append(child)


@dataclass
class Stats:
    """Track RLM execution statistics"""
    iterations: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    sub_llm_calls: int = 0
    max_depth: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    start_time: float = field(default_factory=time.time)
    tool_breakdown: dict = field(default_factory=dict)
    trace_root: TraceNode = field(default_factory=lambda: TraceNode("RLM Query", True, 0))
    _current_trace: TraceNode = None

    def __post_init__(self):
        self._current_trace = self.trace_root

    def record_tool(self, name: str):
        self.tool_calls += 1
        self.tool_breakdown[name] = self.tool_breakdown.get(name, 0) + 1

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def render_trace(self) -> str:
        """Render the execution trace as a text tree"""
        lines = ["\n┌─ Execution Trace ─────────────────────────────"]
        self._render_node(self.trace_root, lines, "", True)
        lines.append("└" + "─" * 47)
        return "\n".join(lines)

    def _render_node(self, node: TraceNode, lines: list, prefix: str, is_last: bool):
        # Determine connector
        if prefix == "":
            connector = "│ "
        else:
            connector = "└─ " if is_last else "├─ "

        # Format node
        async_tag = "\033[36m[async]\033[0m" if node.is_async else "\033[33m[sync]\033[0m"
        duration = f" ({node.duration_ms:.0f}ms)" if node.duration_ms > 0 else ""
        result = f" → {node.result_preview}" if node.result_preview else ""

        line = f"│ {prefix}{connector}{async_tag} {node.name}{duration}{result}"
        lines.append(line)

        # Render children
        child_prefix = prefix + ("   " if is_last else "│  ")

        # Group parallel children
        parallel_groups = {}
        for child in node.children:
            if child.parallel_group >= 0:
                if child.parallel_group not in parallel_groups:
                    parallel_groups[child.parallel_group] = []
                parallel_groups[child.parallel_group].append(child)

        # Render children, marking parallel groups
        i = 0
        while i < len(node.children):
            child = node.children[i]
            is_child_last = (i == len(node.children) - 1)

            if child.parallel_group >= 0 and child.parallel_group in parallel_groups:
                group = parallel_groups.pop(child.parallel_group)
                if len(group) > 1:
                    lines.append(f"│ {child_prefix}┌─ \033[35m[parallel x{len(group)}]\033[0m")
                    for j, g_child in enumerate(group):
                        self._render_node(g_child, lines, child_prefix + "│  ", j == len(group) - 1)
                    lines.append(f"│ {child_prefix}└─────────────────")
                    i += len(group)
                    continue

            self._render_node(child, lines, child_prefix, is_child_last)
            i += 1

    def summary(self) -> str:
        lines = [
            "",
            "═" * 50,
            f"  LLM calls: {self.llm_calls}  │  Iterations: {self.iterations}  │  Tools: {self.tool_calls}",
        ]

        # Recursion info with explanation
        if self.sub_llm_calls > 0:
            lines.append(f"  sub_llm calls: {self.sub_llm_calls}  │  Max depth: {self.max_depth}")
        else:
            lines.append(f"  sub_llm calls: 0 (no recursive decomposition used)")

        lines.append(f"  Tokens: {self.tokens_in:,} in / {self.tokens_out:,} out")
        lines.append(f"  Time: {self.elapsed():.2f}s")

        if self.tool_breakdown:
            breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(self.tool_breakdown.items()))
            lines.append(f"  Tools: {breakdown}")
        lines.append("═" * 50)
        return "\n".join(lines)


ROOT_MODEL = "openai/gpt-4.1"  # Smarter model for planning/coordination
WORKER_MODEL = "openai/gpt-4.1-mini"  # Cheaper model for sub_llm execution


class RLM:
    """
    Recursive Language Model orchestrator.

    Enables LLMs to process arbitrarily large contexts by storing the context
    in a sandbox and letting the LLM use tools to inspect and process it.

    Uses a smarter root_model for planning and a cheaper worker_model for sub_llm calls.
    """

    def __init__(
        self,
        root_model: str = ROOT_MODEL,
        worker_model: str = WORKER_MODEL,
        max_iterations: int = 30,
        _is_worker: bool = False,
    ):
        self.root_model = root_model
        self.worker_model = worker_model
        self.max_iterations = max_iterations
        self._is_worker = _is_worker
        self._client = None

    @property
    def model(self) -> str:
        """Return appropriate model based on whether this is root or worker"""
        return self.worker_model if self._is_worker else self.root_model

    @property
    def client(self):
        if self._client is None:
            self._client = get_async_client()
        return self._client

    async def query(
        self,
        question: str,
        context: str,
        verbose: bool = False,
        _depth: int = 0,
        _stats: Stats | None = None,
        _parent_trace: TraceNode | None = None,
    ) -> str:
        """
        Process a question against arbitrarily large context.

        Args:
            question: The question or instruction to process
            context: The full text context (can be very large)
            verbose: If True, print tool calls and stats
            _depth: Internal - current recursion depth
            _stats: Internal - shared stats object
            _parent_trace: Internal - parent trace node

        Returns:
            The final answer string
        """
        is_root = _stats is None
        stats = _stats or Stats()
        stats.max_depth = max(stats.max_depth, _depth)

        # Create trace node for this query
        query_label = f"query(depth={_depth}, {len(context):,} chars)"
        query_trace = TraceNode(query_label, True, _depth)
        if _parent_trace:
            _parent_trace.add_child(query_trace)
        elif is_root:
            stats.trace_root = query_trace

        indent = "  " * _depth
        sandbox = Sandbox(context)
        system_prompt = format_prompt(sandbox)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        if verbose and _depth > 0:
            model_tag = f"\033[35m{self.model.split('/')[-1]}\033[0m"
            print(f"{indent}\033[36m[async]\033[0m sub_llm started (depth={_depth}, {len(context):,} chars) [{model_tag}]")

        for iteration in range(self.max_iterations):
            stats.iterations += 1
            stats.llm_calls += 1

            iter_trace = TraceNode(f"LLM call #{iteration+1}", True, _depth)
            query_trace.add_child(iter_trace)

            if verbose:
                model_short = self.model.split('/')[-1]
                role = "worker" if self._is_worker else "root"
                print(f"{indent}\033[36m[async]\033[0m iter {iteration+1}: calling \033[35m{model_short}\033[0m ({role})...")

            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
            )
            iter_trace.duration_ms = (time.time() - start_time) * 1000

            if hasattr(response, 'usage') and response.usage:
                stats.tokens_in += response.usage.prompt_tokens or 0
                stats.tokens_out += response.usage.completion_tokens or 0

            message = response.choices[0].message

            if not message.tool_calls:
                if verbose and is_root:
                    print(stats.render_trace())
                    print(stats.summary())
                return message.content or "No response generated"

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            # Execute tool calls
            results = await self._execute_tools_parallel(
                sandbox, message.tool_calls, question, verbose, _depth, stats, iter_trace
            )

            for tc, result in zip(message.tool_calls, results):
                stats.record_tool(tc.function.name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

                if tc.function.name in ("final", "final_var"):
                    if verbose and is_root:
                        print(stats.render_trace())
                        print(stats.summary())
                    return result

        if verbose and is_root:
            print(stats.render_trace())
            print(stats.summary())
        return f"Max iterations ({self.max_iterations}) reached"

    async def _execute_tools_parallel(
        self,
        sandbox: Sandbox,
        tool_calls: list,
        original_query: str,
        verbose: bool,
        depth: int,
        stats: Stats,
        parent_trace: TraceNode,
    ) -> list[str]:
        """Execute tool calls, running sub_llm calls in parallel."""
        tasks = []
        trace_nodes = []
        indent = "  " * depth
        parallel_group = id(tool_calls)  # Unique group ID for parallel calls

        # Identify if we have multiple async calls
        async_count = sum(1 for tc in tool_calls if tc.function.name == "sub_llm")
        use_parallel_marker = async_count > 1

        for i, tc in enumerate(tool_calls):
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            is_async = (name == "sub_llm")

            # Create trace node
            args_preview = json.dumps(args, ensure_ascii=False)
            if len(args_preview) > 40:
                args_preview = args_preview[:40] + "..."
            trace_node = TraceNode(
                f"{name}({args_preview})",
                is_async,
                depth,
                parallel_group=parallel_group if use_parallel_marker and is_async else -1
            )
            trace_nodes.append(trace_node)
            parent_trace.add_child(trace_node)

            if verbose:
                tag = "\033[36m[async]\033[0m" if is_async else "\033[33m[sync]\033[0m"
                print(f"{indent}  {tag} {name}({args_preview})")

            if is_async:
                stats.sub_llm_calls += 1
                task = self._execute_sub_llm(sandbox, args, verbose, depth, stats, trace_node)
            else:
                task = self._execute_sync_tool(sandbox, name, args, trace_node)

            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Update trace nodes with results
        for trace_node, result in zip(trace_nodes, results):
            preview = result[:50].replace('\n', ' ')
            trace_node.result_preview = preview + ("..." if len(result) > 50 else "")
            if verbose:
                tag = "\033[36m[async]\033[0m" if trace_node.is_async else "\033[33m[sync]\033[0m"
                print(f"{indent}    {tag} → {preview}...")

        return results

    async def _execute_sync_tool(
        self,
        sandbox: Sandbox,
        name: str,
        args: dict,
        trace_node: TraceNode,
    ) -> str:
        """Execute a sync tool and track timing."""
        start = time.time()
        result = await asyncio.to_thread(execute_tool, sandbox, name, args)
        trace_node.duration_ms = (time.time() - start) * 1000
        return result

    async def _execute_sub_llm(
        self,
        sandbox: Sandbox,
        args: dict[str, Any],
        verbose: bool,
        depth: int,
        stats: Stats,
        trace_node: TraceNode,
    ) -> str:
        """Execute a recursive sub_llm call on a context slice."""
        query = args.get("query", "")
        start = args.get("start", 0)
        end = args.get("end", len(sandbox.context))
        context_slice = sandbox.context[start:end]

        # Create worker RLM - uses worker_model for sub_llm calls
        child_rlm = RLM(
            root_model=self.root_model,
            worker_model=self.worker_model,
            max_iterations=max(5, self.max_iterations // 2),
            _is_worker=True,  # Mark as worker to use worker_model
        )
        child_rlm._client = self.client

        start_time = time.time()
        try:
            result = await child_rlm.query(
                query, context_slice,
                verbose=verbose,
                _depth=depth + 1,
                _stats=stats,
                _parent_trace=trace_node,
            )
            trace_node.duration_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            trace_node.duration_ms = (time.time() - start_time) * 1000
            return f"sub_llm error: {e}"


def run_sync(
    question: str,
    context: str,
    root_model: str = ROOT_MODEL,
    worker_model: str = WORKER_MODEL,
    verbose: bool = False,
) -> str:
    """Synchronous wrapper for RLM.query()."""
    rlm = RLM(root_model=root_model, worker_model=worker_model)
    return asyncio.run(rlm.query(question, context, verbose=verbose))
