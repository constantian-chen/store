"""
ReAct Math Solver with Dynamic Tool Synthesis + Sandbox Execution.

Key design:
  1. LLM generates Python functions per-problem (dynamic tool creation)
  2. Functions are loaded into a per-problem dspy.PythonInterpreter sandbox
  3. Wrapper functions with preserved signatures are passed to ReAct as tools
  4. All code execution happens inside Deno+Pyodide WASM sandbox
"""

import ast
import atexit
import inspect
import json
import math
import os
import random
import sys
import threading
from datetime import datetime
from fractions import Fraction
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Callable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / ".dspy_cache_dynamic_syn"
CACHE_DIR = Path(
    os.environ.get("DYNAMIC_SYN_DSPY_CACHE_DIR", DEFAULT_CACHE_DIR)
).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DSPY_CACHEDIR"] = str(CACHE_DIR)

import dspy
from dspy import GEPA
from omni_split_loader import init_omni_math_dataset
from reflection_lm import build_reflection_lm

dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_cache_dir=str(CACHE_DIR),
    disk_size_limit_bytes=int(os.environ.get("DYNAMIC_SYN_DSPY_CACHE_LIMIT_BYTES", 20 * 1024**3)),
    memory_max_entries=int(os.environ.get("DYNAMIC_SYN_DSPY_MEMORY_MAX_ENTRIES", 10000)),
)

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None
    print("Warning: math_verify is not installed; metric-based evaluation will fail fast.")


if not hasattr(dspy, "PythonInterpreter"):
    raise RuntimeError(
        "This script requires a dspy build that provides dspy.PythonInterpreter."
    )

TOOL_TIMEOUT_SECONDS = 60


_ACTIVE_SESSIONS = set()
_ACTIVE_SESSIONS_LOCK = threading.Lock()


def _register_session(session) -> None:
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSIONS.add(session)


def _unregister_session(session) -> None:
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSIONS.discard(session)


def shutdown_all_sandboxes() -> None:
    with _ACTIVE_SESSIONS_LOCK:
        sessions = list(_ACTIVE_SESSIONS)

    for session in sessions:
        try:
            session.shutdown()
        except Exception:
            pass


atexit.register(shutdown_all_sandboxes)


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

class MultipleToolCreatorSignature(dspy.Signature):
    """
    Solve the given competition-style math problem.

    Requirements:
    - Read the problem carefully and identify the requested final answer.
    - Use exact arithmetic whenever possible.
    - For fractions, reduce to lowest terms.
    - For modular/remainder questions, return the requested residue in the required format.
    - For counting, probability, algebra, and number theory problems, check constraints and edge cases.
    - The final answer must be returned as a plain string, with no units or explanation unless explicitly requested.
    - Do not include LaTeX wrappers such as \\boxed{}, \\( ... \\), or $...$ unless the problem explicitly requires a symbolic expression.

    Dynamic-tool policy:
    - Generate Python functions that can compute or verify the answer in a Pyodide sandbox.
    - Available libraries include the Python standard library and sympy.
    - Output only valid Python code: top-level imports plus one to three function definitions.
    - Every function must have typed parameters, a concise docstring, and an explicit -> str return annotation.
    - Do not include markdown fences, prose, examples, prints, or top-level function calls.
    - Prefer a simple solve/main function when the problem can be solved directly.

    Do not rely on memorized examples or problem-specific answers.
    """

    problem: str = dspy.InputField(
        desc="Mathematical problem statement used as context for tool generation"
    )
    python_codes = dspy.OutputField(
        prefix="```python",
        desc=(
            "Python code containing one to three generic Python function definitions, "
            "separated by two blank lines. "
            "Each function must have explicit type-annotated parameters and return str. "
            "Each function must include a comprehensive docstring. "
            "Each function must return a string with actual computation results. "
            "Do not include markdown fences or prose."
        ),
    )


class ToolRepairSignature(dspy.Signature):
    """Repair generated Python tool code for a competition-style math problem.

    Requirements:
    - Preserve the intended mathematical computation whenever possible.
    - Return only valid Python code.
    - The output must contain only top-level imports and one to three function definitions.
    - Do not include markdown fences, explanations, examples, prints, or top-level calls.
    - Each function must have typed parameters, a concise docstring, and an explicit -> str return annotation.
    - The returned string should be the final answer or a useful verification result.
    """

    problem: str = dspy.InputField(
        desc="Mathematical problem statement used as context for tool generation"
    )
    broken_code: str = dspy.InputField(
        desc="The invalid Python tool code that failed local validation"
    )
    error_message: str = dspy.InputField(
        desc="The exact syntax or validation error produced by local checks"
    )
    repaired_code: str = dspy.OutputField(
        desc=(
            "Only valid Python source code containing one to three top-level "
            "function definitions. Do not include markdown fences or prose."
        )
    )


class FactCheckSignature(dspy.Signature):
    """Solve the given competition-style math problem.

    Requirements:
    - Read the problem carefully and identify the requested final answer.
    - Use exact arithmetic whenever possible.
    - For fractions, reduce to lowest terms.
    - For modular/remainder questions, return the requested residue in the required format.
    - For counting, probability, algebra, and number theory problems, check constraints and edge cases.
    - The answer field must contain only the final answer, with no units or explanation.
    - Do not include LaTeX wrappers such as \\boxed{}, \\( ... \\), or $...$ unless the problem explicitly requires a symbolic expression.

    Tool-use policy:
    - Use the provided per-problem tools to compute intermediate results or verify key steps.
    - Use at least one tool call before giving the final answer.
    - If a tool returns an error, correct the arguments or choose a more appropriate tool.
    """

    problem: str = dspy.InputField(
        desc="Mathematical problem statement"
    )
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning and tool-use summary"
    )
    answer: str = dspy.OutputField(desc="Final answer only")


# ---------------------------------------------------------------------------
# AST-based function extraction and sandbox wrapping
# ---------------------------------------------------------------------------


class SandboxSession:
    """
    Manage a per-problem PythonInterpreter and allow best-effort recovery after
    a timed-out tool call by swapping in a freshly initialized interpreter.
    """

    def __init__(self, preamble: str, user_code: str):
        self.preamble = preamble
        self.user_code = user_code
        self.lock = threading.Lock()
        self._dead = threading.Event()   # set by shutdown(); polled in _execute_in_sandbox
        self._pending_interp: "dspy.PythonInterpreter | None" = None
        self._closed = False
        self._interpreter = self._new_interpreter()
        _register_session(self)

    def _new_interpreter(self) -> dspy.PythonInterpreter:
        """
        Start a fresh Deno/Pyodide process and execute the preamble.

        This is intentionally called on the same thread that will later execute
        tool calls. PythonInterpreter is thread-owned after first use.
        """
        interp = dspy.PythonInterpreter()
        # Stash it so a concurrent kill can reach the subprocess even if we
        # are still blocked inside execute().
        self._pending_interp = interp
        try:
            interp.execute(self.preamble + "\n" + self.user_code)
        finally:
            self._pending_interp = None
        return interp

    @property
    def interpreter(self) -> dspy.PythonInterpreter:
        return self._interpreter

    def rebuild_after_timeout(self) -> None:
        """
        Swap in a fresh interpreter after abandoning a timed-out execution.

        The old Deno process must already be killed by the caller before this is
        invoked. Rebuild happens on the current thread to preserve interpreter
        thread ownership.
        """
        # _kill_interpreter_process was already called by the caller; calling it
        # again here is harmless (process is dead / reference is None).
        _kill_interpreter_process(self._interpreter)
        self._interpreter = None  # mark dead before the rebuild attempt

        try:
            self._interpreter = self._new_interpreter()
        except Exception as exc:
            _kill_interpreter_process(self._pending_interp)
            raise RuntimeError(f"sandbox rebuild failed: {exc}") from exc

    def shutdown(self) -> None:
        # Signal _dead first so any polling _execute_in_sandbox wakes up within
        # 0.5 s without waiting for the Deno pipe to close on its own.
        if self._closed:
            return
        self._closed = True
        self._dead.set()
        _kill_interpreter_process(self._pending_interp)
        self._pending_interp = None
        _kill_interpreter_process(self._interpreter)
        self._interpreter = None
        _unregister_session(self)


def _clean_code_block(code_string: str) -> str:
    """Strip markdown fences if present."""
    code = code_string.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _validate_tool_tree(tree: ast.Module) -> str | None:
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return "No function definitions found in generated code."

    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return (
                "Generated code must not contain bare top-level function calls; "
                f"found a call to '{ast.dump(node.value.func)[:60]}'."
            )

    for fn in functions:
        for child in ast.walk(fn):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == "input":
                    return f"Function {fn.name} calls input(), which is not allowed."

    return None


def _kill_interpreter_process(interpreter: dspy.PythonInterpreter | None) -> None:
    """
    Best-effort hard stop for the sandbox process.

    The upstream shutdown() path can block forever on deno_process.wait(), so
    timeout recovery uses direct process termination instead.
    """
    if interpreter is None:
        return

    proc = getattr(interpreter, "deno_process", None)
    try:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
    finally:
        if proc is not None:
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = getattr(proc, stream_name, None)
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        try:
            interpreter.deno_process = None
        except Exception:
            pass
        try:
            interpreter._owner_thread = None
        except Exception:
            pass


def _should_reset_sandbox_exception(exc: Exception) -> bool:
    message = str(exc)
    return isinstance(exc, (MemoryError, BrokenPipeError)) or any(
        marker in message
        for marker in (
            "Deno exited",
            "BrokenPipe",
            "No response",
            "Response ID mismatch",
            "Too many non-JSON lines",
            "MemoryError",
        )
    )


def _extract_function_info(source: str) -> List[dict]:
    """
    Parse source code with AST, extract each function's:
      - name
      - parameters with type annotations
      - docstring
      - source code
    """
    tree = ast.parse(source)
    functions = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        def _annotation_text(annotation_node) -> str | None:
            if annotation_node is None:
                return None
            text = ast.get_source_segment(source, annotation_node)
            return text if text is not None else "str"

        def _safe_default(default_node):
            if default_node is None:
                return inspect._empty
            try:
                return ast.literal_eval(default_node)
            except Exception:
                return inspect._empty

        params = []

        positional = list(node.args.posonlyargs) + list(node.args.args)
        positional_defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)

        for idx, arg in enumerate(node.args.posonlyargs):
            params.append({
                "name": arg.arg,
                "kind": "POSITIONAL_ONLY",
                "annotation": _annotation_text(arg.annotation),
                "default": _safe_default(positional_defaults[idx]),
            })

        posonly_count = len(node.args.posonlyargs)
        for idx, arg in enumerate(node.args.args, start=posonly_count):
            params.append({
                "name": arg.arg,
                "kind": "POSITIONAL_OR_KEYWORD",
                "annotation": _annotation_text(arg.annotation),
                "default": _safe_default(positional_defaults[idx]),
            })

        if node.args.vararg is not None:
            params.append({
                "name": node.args.vararg.arg,
                "kind": "VAR_POSITIONAL",
                "annotation": _annotation_text(node.args.vararg.annotation),
                "default": inspect._empty,
            })

        for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
            params.append({
                "name": arg.arg,
                "kind": "KEYWORD_ONLY",
                "annotation": _annotation_text(arg.annotation),
                "default": _safe_default(default_node),
            })

        if node.args.kwarg is not None:
            params.append({
                "name": node.args.kwarg.arg,
                "kind": "VAR_KEYWORD",
                "annotation": _annotation_text(node.args.kwarg.annotation),
                "default": inspect._empty,
            })

        # Extract docstring
        docstring = ast.get_docstring(node) or f"Dynamically generated function: {node.name}"

        # Extract source code for this function
        func_lines = source.split("\n")[node.lineno - 1 : node.end_lineno]
        func_source = "\n".join(func_lines)

        functions.append({
            "name": node.name,
            "params": params,
            "docstring": docstring,
            "source": func_source,
        })

    return functions


def _make_sandbox_wrapper(func_info: dict, session: SandboxSession) -> Callable:
    """
    Create a wrapper function that:
      - Has the same name and parameter names as the original
      - Serializes arguments in the host process
      - Executes the real synthesized function inside the sandbox interpreter
    """
    func_name = func_info["name"]
    params = func_info["params"]
    docstring = func_info["docstring"]

    signature_params = []
    annotations = {}

    for param in params:
        annotation = param.get("annotation")
        if annotation is not None:
            annotations[param["name"]] = annotation

        signature_params.append(
            inspect.Parameter(
                name=param["name"],
                kind=getattr(inspect.Parameter, param["kind"]),
                default=param.get("default", inspect._empty),
                annotation=annotation if annotation is not None else inspect._empty,
            )
        )

    signature = inspect.Signature(signature_params, return_annotation=str)

    def wrapper_func(*args, **kwargs) -> str:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        payload = json.dumps(
            {
                "args": list(bound.args),
                "kwargs": bound.kwargs,
            }
        )
        call_code = (
            "import json\n"
            f"_payload = json.loads({payload!r})\n"
            "_args = _payload['args']\n"
            "_kwargs = _payload['kwargs']\n"
            f"result = {func_name}(*_args, **_kwargs)\n"
            "print(result)"
        )
        try:
            return _execute_in_sandbox(session, call_code, TOOL_TIMEOUT_SECONDS)
        except Exception as exc:
            return f"Execution error: {type(exc).__name__}: {exc}"

    wrapper_func.__name__ = func_name
    wrapper_func.__doc__ = docstring
    wrapper_func.__signature__ = signature
    wrapper_func.__annotations__ = annotations | {"return": str}
    return wrapper_func


def create_sandboxed_tools(
    code_string: str,
) -> Tuple[List[Callable], SandboxSession | None, str | None, List[str]]:
    """
    Main entry point: parse generated code, create sandbox, return wrapped tools.

    Returns:
        tools: list of callable wrapper functions
        session: the SandboxSession instance (caller must shutdown later)
        error: error message if failed, None if success
        diagnostics: non-fatal warnings from wrapping usable generated code
    """
    diagnostics = []
    code = _clean_code_block(code_string)

    # 1. Parse AST to extract function info
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [], None, f"SyntaxError in generated code: {e}", diagnostics

    validation_error = _validate_tool_tree(tree)
    if validation_error:
        return [], None, validation_error, diagnostics

    func_infos = _extract_function_info(code)

    # 2. Create a fresh interpreter for this problem.
    preamble = (
        "import sympy\n"
        "import math\n"
        "from fractions import Fraction\n"
        "from itertools import combinations, permutations, product\n"
    )
    try:
        session = SandboxSession(preamble=preamble, user_code=code)
    except Exception as e:
        return [], None, f"Failed to load functions into sandbox: {e}", diagnostics

    # 4. Create wrappers with preserved signatures
    tools = []
    for info in func_infos:
        if not info["docstring"] or info["docstring"].startswith("Dynamically generated"):
            warning = f"Warning: {info['name']} has no docstring; using generic description."
            diagnostics.append(warning)
        try:
            wrapper = _make_sandbox_wrapper(info, session)
            tools.append(wrapper)
        except Exception as e:
            warning = f"Warning: failed to wrap {info['name']}: {e}"
            diagnostics.append(warning)

    if not tools:
        session.shutdown()
        return [], None, "All function wrappers failed.", diagnostics

    return tools, session, None, diagnostics


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def metric_fnv4(example, pred, trace=None):
    if parse is None or verify is None:
        raise RuntimeError(
            "metric_fnv4 requires math_verify to be installed, but it could not be imported."
        )
    try:
        def wrap(s):
            s = str(s).strip()
            return s if s.startswith("$") else f"${s}$"

        gold = parse(wrap(example.answer), parsing_timeout=None)
        predicted = parse(wrap(pred.answer), parsing_timeout=None)
        return int(verify(gold, predicted, timeout_seconds=None))
    except Exception:
        return 0


def _wrap_math_answer(value) -> str:
    value = str(value).strip()
    if not value.startswith("$"):
        value = f"${value}$"
    return value


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Optimization metric for GEPA with math_verify-based feedback."""
    correct_answer = str(example["answer"]).strip()
    written_solution = str(example.get("solution", "")).strip()
    predicted_answer = str(getattr(prediction, "answer", "")).strip()
    tool_debug = str(getattr(prediction, "tool_debug", "")).strip()

    def add_tool_feedback(feedback: str) -> str:
        if not tool_debug:
            return feedback
        return (
            feedback
            + "\n\nTool generation diagnostics from this rollout:\n"
            + tool_debug[:8000]  # cap tool debug feedback to avoid overwhelming the model
            + "\nFuture tool code should contain only top-level imports and one to three function definitions. "
            "Every function needs typed parameters, a docstring, and an explicit -> str return annotation. "
            "Repair outputs must be valid Python code only, with no markdown fences or prose."
        )

    try:
        gold = parse(_wrap_math_answer(correct_answer), parsing_timeout=None)
        predicted = parse(_wrap_math_answer(predicted_answer), parsing_timeout=None)
        score = int(verify(gold, predicted, timeout_seconds=None))
    except Exception:
        feedback_text = (
            "Your final answer could not be parsed or verified by math_verify. "
            f"You responded with '{predicted_answer[:500]}'. Please return only the final mathematical answer "
            "in a clean format with no extra explanation."
        )
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += (
                " Here's the full step-by-step solution:\n"
                f"{written_solution}\n\n"
                "Study the solution carefully and extract reusable strategies for similar problems."
            )
        return dspy.Prediction(score=0, feedback=add_tool_feedback(feedback_text))

    if score == 1:
        feedback_text = (
            f"Your answer is mathematically correct under symbolic verification. "
            f"The correct answer is '{correct_answer}'."
        )
    else:
        feedback_text = (
            f"Your answer is mathematically incorrect under symbolic verification. "
            f"The correct answer is '{correct_answer}'. "
            "Check algebraic manipulation, numeric simplification, and whether your final answer format is clean."
        )
        if written_solution:
            feedback_text += (
                " Here's the full step-by-step solution:\n"
                f"{written_solution}\n\n"
                "Study the solution carefully and extract reusable strategies for similar problems."
            )

    return dspy.Prediction(score=score, feedback=add_tool_feedback(feedback_text))


# ---------------------------------------------------------------------------
# Solver Module
# ---------------------------------------------------------------------------

class ReActMathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tool_creator = dspy.ChainOfThought(MultipleToolCreatorSignature)
        self.tool_repair = dspy.ChainOfThought(ToolRepairSignature)

    def forward(self, problem: str) -> dspy.Prediction:
        session = None
        tool_debug: list[str] = []
        try:
            # 1. Generate tool code
            tool_result = self.tool_creator(problem=problem)
            python_codes = tool_result.python_codes

            # 2. Create sandboxed tools (per-problem interpreter)
            tools, session, err, diagnostics = create_sandboxed_tools(python_codes)
            tool_debug.extend(diagnostics)

            if err:
                tool_debug.append(f"Tool creation failed: {err}")
                if session is not None:
                    try:
                        session.shutdown()
                    except Exception:
                        pass
                    session = None

                try:
                    repair_result = self.tool_repair(
                        problem=problem,
                        broken_code=python_codes,
                        error_message=err,
                    )
                    repaired_code = repair_result.repaired_code
                    tools, session, err, diagnostics = create_sandboxed_tools(repaired_code)
                    tool_debug.extend(diagnostics)
                    if err:
                        tool_debug.append(f"Tool repair failed: {err}")
                    else:
                        tool_debug.append("Tool repair succeeded after initial creation failure.")
                except Exception as repair_exc:
                    err = f"Tool repair exception: {repair_exc}"
                    tool_debug.append(err)

            if err:
                tool_debug.append("Fallback python_eval tool was used because synthesized tools were unavailable.")
                # Fallback: use a basic python_eval tool with the same math-oriented
                # preamble as the synthesized-tool sandbox.
                preamble = (
                    "import sympy\n"
                    "import math\n"
                    "from fractions import Fraction\n"
                    "from itertools import combinations, permutations, product\n"
                )
                session = SandboxSession(preamble=preamble, user_code="")

                def python_eval(code: str) -> str:
                    """Execute Python code for mathematical calculations."""
                    try:
                        return _execute_in_sandbox(session, code, TOOL_TIMEOUT_SECONDS)
                    except Exception as exc:
                        return f"Execution error: {type(exc).__name__}: {exc}"

                tools = [python_eval]

            # 3. Solve with ReAct in the current thread. The sandbox is also
            #    created and called in this thread because PythonInterpreter is
            #    thread-owned after first use.
            react_agent = dspy.ReAct(
                signature=FactCheckSignature,
                tools=tools,
                max_iters=6,
            )
            prediction = react_agent(problem=problem)
            trajectory = getattr(prediction, "trajectory", None)
            if trajectory:
                for key, value in trajectory.items():
                    value_text = str(value)
                    if any(
                        marker in value_text
                        for marker in (
                            "Execution error",
                            "TimeoutError",
                            "sandbox execution exceeded",
                            "Deno exited",
                            "BrokenPipe",
                            "MemoryError",
                        )
                    ):
                        tool_debug.append(f"ReAct tool error [{key}]: {value_text[:500]}")
            return dspy.Prediction(
                answer=prediction.answer,
                reasoning=getattr(prediction, "reasoning", ""),
                trajectory=trajectory,
                tool_debug="\n".join(tool_debug),
            )

        except Exception as e:
            tool_debug.append(f"Forward error: {type(e).__name__}: {e}")
            return dspy.Prediction(
                answer=0,
                tool_debug="\n".join(tool_debug),
            )

        finally:
            # Always clean up the interpreter
            if session is not None:
                try:
                    session.shutdown()
                except Exception:
                    pass


LOG_DIR = SCRIPT_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)
GEPA_RUN_DIR = SCRIPT_DIR / "gepa_runs"
GEPA_RUN_DIR.mkdir(exist_ok=True)

_RESOURCE_MONITOR_STOP = threading.Event()


def _read_proc_status(pid: int) -> dict[str, str]:
    status_path = Path("/proc") / str(pid) / "status"
    if not status_path.exists():
        return {}

    wanted = {"VmRSS", "VmHWM", "VmSize", "Threads"}
    result = {}
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key in wanted:
                result[key] = value.strip()
    except Exception:
        return {}
    return result


def _active_sandbox_processes() -> list[tuple[int, dict[str, str]]]:
    with _ACTIVE_SESSIONS_LOCK:
        sessions = list(_ACTIVE_SESSIONS)

    children = []
    for session in sessions:
        for interpreter in (getattr(session, "_pending_interp", None), getattr(session, "_interpreter", None)):
            proc = getattr(interpreter, "deno_process", None)
            if proc is not None and proc.poll() is None:
                children.append((proc.pid, _read_proc_status(proc.pid)))
    return children


def log_resource_snapshot(label: str, log_path: Path | None = None) -> None:
    if os.name != "posix":
        return

    main_status = _read_proc_status(os.getpid())
    children = _active_sandbox_processes()
    line = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "pid": os.getpid(),
        "main": main_status,
        "active_sandboxes": len(children),
        "sandbox_children": [{"pid": pid, **status} for pid, status in children],
    }
    text = json.dumps(line, ensure_ascii=False)
    if log_path is not None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass


def start_resource_monitor(log_path: Path, interval_seconds: int = 30) -> threading.Thread | None:
    if os.name != "posix":
        return None

    def monitor() -> None:
        while not _RESOURCE_MONITOR_STOP.wait(interval_seconds):
            log_resource_snapshot("periodic", log_path)

    thread = threading.Thread(target=monitor, name="resource-monitor", daemon=True)
    thread.start()
    return thread


def _execute_in_sandbox(session: SandboxSession, code: str, timeout_seconds: int) -> str:
    """
    Run sandbox code in the current thread.

    PythonInterpreter owns the thread that first uses it, so this function does
    not offload execution to a ThreadPoolExecutor. Timeout enforcement comes
    from the PythonInterpreter source itself. If execution times out or the Deno
    process fails, rebuild the sandbox for the next tool call.
    """
    with session.lock:
        interpreter = session.interpreter
        if interpreter is None:
            raise RuntimeError("Sandbox session has been shut down")

        try:
            if session._dead.is_set():
                raise RuntimeError("Sandbox session was shut down externally")
            return str(interpreter.execute(code))
        except TimeoutError as exc:
            _kill_interpreter_process(interpreter)
            try:
                session.rebuild_after_timeout()
            except Exception as rebuild_err:
                print(f"  Warning: sandbox rebuild failed: {rebuild_err}")
            raise TimeoutError(
                f"sandbox execution exceeded {timeout_seconds}s; sandbox rebuilt"
            ) from exc
        except Exception as exc:
            if _should_reset_sandbox_exception(exc):
                _kill_interpreter_process(interpreter)
                try:
                    session.rebuild_after_timeout()
                except Exception as rebuild_err:
                    print(f"  Warning: sandbox rebuild failed: {rebuild_err}")
            raise


def run_one(program, ex, idx: int, total: int) -> dict:
    entry = {
        "id": idx,
        "problem": ex.problem,
        "expected": ex.answer,
        "predicted": None,
        "status": None,
        "metric": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"[{idx}/{total}] {ex.problem}")
    print(f"Expected: {ex.answer}")
    print(f"{'=' * 70}")
    log_resource_snapshot(f"eval_start_{idx}")

    try:
        pred = program(problem=ex.problem)
        predicted = str(pred.answer).strip()
        score = metric_fnv4(ex, pred)

        entry["predicted"] = predicted
        entry["metric"] = score
        entry["status"] = "correct" if score == 1 else "wrong"

        trajectory = getattr(pred, "trajectory", None)
        if trajectory:
            entry["trajectory"] = {k: str(v) for k, v in trajectory.items()}
        tool_debug = getattr(pred, "tool_debug", None)
        if tool_debug:
            entry["tool_debug"] = str(tool_debug)

        print(f"Predicted: {predicted}")
        print(f"Result:    {'CORRECT' if score == 1 else 'WRONG'}")
    except Exception as e:
        entry["predicted"] = f"ERROR: {e}"
        entry["status"] = "error"
        print(f"EXCEPTION: {e}")
    finally:
        log_resource_snapshot(f"eval_end_{idx}")

    return entry


def evaluate_with_logging(program, examples, log_path: Path) -> dict:
    results = []
    correct = []
    total = len(examples)

    print("\n===== ReAct + Synthesized Tools =====")
    print(f"Log -> {log_path}")

    for i, ex in enumerate(examples, 1):
        entry = run_one(program, ex, i, total)
        results.append(entry)
        if entry["status"] == "correct":
            correct.append(i)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "correct": correct,
                        "accuracy": f"{len(correct)}/{len(results)}",
                        "last_completed": i,
                    },
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return {
        "correct_count": len(correct),
        "total": total,
        "accuracy": f"{len(correct)}/{total}",
        "log_path": str(log_path),
    }


def build_optimizer(reflection_lm: dspy.LM) -> GEPA:
    track_best_outputs = os.environ.get("GEPA_TRACK_BEST_OUTPUTS", "").lower() in {"1", "true", "yes"}
    return GEPA(
        metric=metric_with_feedback,
        max_metric_calls=6900,
        num_threads=4,
        track_stats=True,
        track_best_outputs=track_best_outputs,
        log_dir=str(GEPA_RUN_DIR),
        reflection_minibatch_size=3,
        reflection_lm=reflection_lm,
    )


def init_dataset():
    """Load the frozen Omni-MATH-Rule split used across omini experiments."""
    return init_omni_math_dataset()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    resource_log_path = LOG_DIR / f"resource_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    resource_monitor = start_resource_monitor(resource_log_path)
    log_resource_snapshot("startup", resource_log_path)
    try:
        train_examples, val_examples, test_examples = init_dataset()

        student_lm = dspy.LM(
            model="openai//disk/scratch/s2799944/Qwen3.5-9B",
            api_base="http://localhost:8001/v1",
            api_key="EMPTY",
            max_tokens=16384,
            cache=True,
            num_retries=0,
            timeout=600,
        )

        teacher_lm = build_reflection_lm()

        dspy.configure(
            lm=student_lm,
        )

        mode = sys.argv[1] if len(sys.argv) > 1 else "train"
        math_solver = ReActMathSolver()

        if mode == "test":
            example_index = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            if not 1 <= example_index <= len(test_examples):
                raise ValueError(f"test index must be between 1 and {len(test_examples)}")

            example = test_examples[example_index - 1]
            print(f"Running single-problem test on Omni-MATH example #{example_index}")
            print(f"Problem: {example.problem}")
            print(f"Expected: {example.answer}")

            prediction = math_solver(problem=example.problem)
            print(f"Predicted: {prediction.answer}")
            print("Reasoning:")
            print(prediction.reasoning)

            trajectory = getattr(prediction, "trajectory", None)
            if trajectory:
                print("Trajectory:")
                print(json.dumps({k: str(v) for k, v in trajectory.items()}, ensure_ascii=False, indent=2))

            if parse is not None and verify is not None:
                score = metric_fnv4(example, prediction)
                print(f"Metric: {score}")
        else:
            log_path = LOG_DIR / f"9breact_synthesize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            optimizer = build_optimizer(teacher_lm)
            log_resource_snapshot("before_gepa_compile", resource_log_path)
            optimized_solver = optimizer.compile(
                math_solver,
                trainset=train_examples,
                valset=val_examples,
            )
            log_resource_snapshot("after_gepa_compile", resource_log_path)
            optimized_result = evaluate_with_logging(optimized_solver, test_examples, log_path)
            print("Optimized result:", optimized_result)
            optimized_solver.save(str(SCRIPT_DIR / "optimized_9breact_synthesize.json"))
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down active sandboxes...")
        shutdown_all_sandboxes()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    finally:
        _RESOURCE_MONITOR_STOP.set()
        log_resource_snapshot("shutdown", resource_log_path)
        shutdown_all_sandboxes()
