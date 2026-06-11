"""
ReAct Math Solver with Dynamic Tool Synthesis — GEPA-Optimizable Variant.
Self-contained: no imports from sibling files.

Key change vs. plain dspy.ReAct + dynamic synthesis:
  The reasoner and extractor are persistent attributes of the solver, so
  `program.named_parameters()` finds them and GEPA can rewrite their
  instructions. Tools are passed in at forward() time, not baked into
  __init__-time prompts, so per-problem tool synthesis still works.

GEPA will discover and optimise three prompts:
    ReActMathSolver
    ├── tool_creator               (tool synthesis)
    ├── react.reasoner             (next-step thought + tool choice)
    └── react.extractor            (final-answer extraction)
"""

import ast
import atexit
import gc
import inspect
import json
import math
import multiprocessing as mp
import os
import random
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / ".dspy_cache_dynamic_syn_persist"
CACHE_DIR = Path(
    os.environ.get("DYNAMIC_SYN_PERSIST_DSPY_CACHE_DIR", DEFAULT_CACHE_DIR)
).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DSPY_CACHEDIR"] = str(CACHE_DIR)

import dspy
import sympy as sp
from dspy import GEPA
from omni_split_loader import init_omni_math_dataset
from reflection_lm import build_reflection_lm
from sympy import Symbol, simplify, solve
from sympy.parsing.latex import parse_latex

dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_cache_dir=str(CACHE_DIR),
    disk_size_limit_bytes=int(os.environ.get("DYNAMIC_SYN_PERSIST_DSPY_CACHE_LIMIT_BYTES", 20 * 1024**3)),
    memory_max_entries=int(os.environ.get("DYNAMIC_SYN_PERSIST_DSPY_MEMORY_MAX_ENTRIES", 10000)),
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
GEPA_EXPERIMENT_SEED = int(os.environ.get("GEPA_EXPERIMENT_SEED", "42"))
random.seed(GEPA_EXPERIMENT_SEED)
DEFAULT_GEPA_NUM_THREADS = int(os.environ.get("GEPA_NUM_THREADS", "4"))
GC_EVERY_FORWARD = os.environ.get("GEPA_GC_EVERY_FORWARD", "1").lower() in {
    "1",
    "true",
    "yes",
}
NOFILE_TARGET = int(os.environ.get("GEPA_NOFILE_TARGET", "65535"))


def raise_file_descriptor_limit() -> None:
    if os.name != "posix":
        return
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(soft, NOFILE_TARGET), hard)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"Raised RLIMIT_NOFILE from {soft} to {target}.", flush=True)
        else:
            print(f"RLIMIT_NOFILE soft={soft}, hard={hard}.", flush=True)
    except Exception as exc:
        print(f"Warning: could not adjust RLIMIT_NOFILE: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Sandbox session bookkeeping (multi-process safe shutdown)
# ---------------------------------------------------------------------------

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
# Primitive tools copied from utils/equationSolver.py fallback implementation
# ---------------------------------------------------------------------------

def _run_with_timeout(func, args=(), timeout: int = 30):
    """Run a function in a subprocess and return None on timeout/error."""
    with mp.Pool(processes=1) as pool:
        async_result = pool.apply_async(func, args)
        try:
            return async_result.get(timeout)
        except mp.TimeoutError:
            return None


def _parse_latex_expr(expr: str):
    """Clean a LaTeX math string and parse it into a SymPy expression."""
    expr = expr.strip()
    if len(expr) >= 2 and expr[0] == "$" and expr[-1] == "$":
        expr = expr[1:-1].strip()
    expr = expr.replace("\\%", "/ 100")
    return parse_latex(expr)


def _sympy_simplify_and_eval(expr):
    """Simplify symbolically, then try numeric evaluation."""
    simplified = simplify(expr)
    numeric = sp.N(simplified)
    return simplified, numeric


def calculator(expr_latex: str, timeout: int = 30) -> str:
    """
    Evaluate a LaTeX arithmetic expression exactly with SymPy.

    Example input: "$\\frac{1}{\\log_{15} 2 + 1}$" or "$2^2 + 2 + 1$".
    Returns a string containing the simplified exact value and, when useful,
    a numeric check.
    """
    try:
        expr = _parse_latex_expr(expr_latex)
    except Exception as exc:
        return f"Calculator parse error: {exc}"

    result = _run_with_timeout(_sympy_simplify_and_eval, args=(expr,), timeout=timeout)
    if result is None:
        return "Calculator timeout or execution error."

    simplified, numeric = result
    try:
        numeric_float = float(numeric)
        if math.isinf(numeric_float):
            return f"{expr_latex} is too large."
        if str(simplified) == str(numeric):
            return f"{expr_latex} = {simplified}"
        return f"{expr_latex} = {simplified} = {numeric}"
    except Exception:
        return f"{expr_latex} = {simplified}"


def _solve_equations_worker(equations, variables):
    """Worker for solving equations."""
    return solve(equations, variables, dict=True)


def equation_solver(
    variables_latex: str,
    equations_latex: str,
    timeout: int = 30,
) -> str:
    """
    Solve LaTeX equations for the requested variables using SymPy.

    Example variables_latex: "$a$, $b$" or "a, b".
    Example equations_latex: "$5*a-3*b = 32$, $3*a+5*b = -8$".
    Returns solutions such as "a = 4, b = -4".
    """
    variables = []
    for var in variables_latex.split(","):
        var = var.strip()
        if len(var) >= 2 and var[0] == "$" and var[-1] == "$":
            var = var[1:-1].strip()
        if var:
            variables.append(Symbol(var))

    if not variables:
        return "Equation solver error: no variables provided."

    raw_equations = [eq.strip() for eq in equations_latex.split(",") if eq.strip()]
    parsed_equations = []

    for eq in raw_equations:
        try:
            if len(eq) >= 2 and eq[0] == "$" and eq[-1] == "$":
                eq = eq[1:-1].strip()
            eq = eq.replace("\\%", "/ 100")
            if "=" in eq:
                lhs, rhs = eq.split("=", 1)
                eq = f"{lhs} - ({rhs})"
            parsed_equations.append(simplify(_parse_latex_expr(eq)))
        except Exception as exc:
            return f"Equation solver parse error for `{eq}`: {exc}"

    if not parsed_equations:
        return "Equation solver error: no valid equations parsed."

    result = _run_with_timeout(
        _solve_equations_worker,
        args=(parsed_equations, variables),
        timeout=timeout,
    )
    if result is None:
        return "Equation solver timeout or execution error."
    if not result:
        return "Equation solver found no solution."

    formatted_solutions = []
    for sol in result:
        parts = []
        for var in variables:
            if var in sol:
                parts.append(f"{var} = {sol[var]}")
        if parts:
            formatted_solutions.append(", ".join(parts))

    if not formatted_solutions:
        return f"Equation solver raw result: {result}"
    return " ; ".join(formatted_solutions)


PRIMITIVE_TOOL_SOURCE = "builtin fallback in react_syn_persistent_equation_primitives.py"
PRIMITIVE_TOOL_IMPORT_ERROR = ""

try:
    from utils.equationSolver import calculator as _primitive_calculator
    from utils.equationSolver import equation_solver as _primitive_equation_solver

    calculator = _primitive_calculator
    equation_solver = _primitive_equation_solver
    PRIMITIVE_TOOL_SOURCE = "utils.equationSolver"
except Exception as exc:
    PRIMITIVE_TOOL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


PRIMITIVE_TOOLS: list[Callable] = [calculator, equation_solver]


# ---------------------------------------------------------------------------
# Signatures: tool synthesis
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
    - Two primitive tools are already available and must not be synthesized again:
      calculator(expr_latex: str, timeout: int = 30) -> str
      equation_solver(variables_latex: str, equations_latex: str, timeout: int = 30) -> str
    - Generate only additional Python functions that compute or verify the answer in a Pyodide sandbox.
    - Do not duplicate, wrap, reimplement, or rename the primitive calculator/equation_solver tools.
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


# ---------------------------------------------------------------------------
# Signatures: persistent ReAct loop (this is what makes GEPA see ReAct)
# ---------------------------------------------------------------------------

class ReActStepSignature(dspy.Signature):
    """Decide the next step in a tool-using reasoning loop.

    Requirements:
    - Read the problem carefully and identify the requested final answer.
    - Use exact arithmetic whenever possible.
    - For fractions, reduce to lowest terms.
    - For modular/remainder questions, return the requested residue in the required format.
    - For counting, probability, algebra, and number theory problems, check constraints and edge cases.

    Tool-use policy:
    - Read the available tool names, signatures, and docstrings literally.
    - Choose exactly one next action: call one tool with matching JSON arguments, or emit next_tool_name='finish'.
    - If a tool has no parameters, use an empty JSON object: {}.
    - If a tool call fails, correct the arguments or choose a more appropriate tool.
    - A general-purpose 'python_eval' tool is available: it runs arbitrary
      Python (sympy, math, Fraction, itertools preloaded). Use it to compute
      directly, or to re-check / iterate when another tool returns a
      suspicious value such as 0, an empty result, or something inconsistent
      with the problem. Do not blindly trust a single tool call; verify before
      finishing.
    - Finish only when the trajectory contains a credible, verified final answer in the requested format.
    """

    problem: str = dspy.InputField(
        desc="Mathematical problem statement"
    )
    tools_description: str = dspy.InputField(
        desc=(
            "List of tools available for this problem, with their names, "
            "signatures, and docstrings. The special tool 'finish' (no args) "
            "ends the episode."
        )
    )
    trajectory: str = dspy.InputField(
        desc="Previous thoughts, tool calls, and observations in this episode"
    )

    next_thought: str = dspy.OutputField(
        desc="Reasoning about the current state and the next action to take"
    )
    next_tool_name: str = dspy.OutputField(
        desc="Name of the tool to call next, or 'finish' to stop"
    )
    next_tool_args: dict[str, Any] = dspy.OutputField(
        desc="JSON object of arguments for the selected tool; {} for finish"
    )


class ExtractAnswerSignature(dspy.Signature):
    """Extract the final answer from a completed reasoning trajectory.

    Requirements:
    - The answer field must contain only the final answer, with no units or explanation.
    - Do not include LaTeX wrappers such as \\boxed{}, \\( ... \\), or $...$ unless the problem explicitly requires a symbolic expression.
    - Preserve required formatting, such as reduced fractions or requested modular/remainder form.
    """

    problem: str = dspy.InputField(
        desc="Mathematical problem statement"
    )
    trajectory: str = dspy.InputField(
        desc="Full trajectory of thoughts, tool calls, and observations"
    )

    reasoning: str = dspy.OutputField(
        desc="Concise summary of how the answer follows from the trajectory"
    )
    answer: str = dspy.OutputField(
        desc="Final mathematical answer only (no units, no explanation)"
    )


# ---------------------------------------------------------------------------
# Sandbox session
# ---------------------------------------------------------------------------

class SandboxSession:
    """Manage a per-problem PythonInterpreter with timeout-safe rebuild."""

    def __init__(self, preamble: str, user_code: str):
        self.preamble = preamble
        self.user_code = user_code
        self.lock = threading.Lock()
        self._dead = threading.Event()
        self._pending_interp: "dspy.PythonInterpreter | None" = None
        self._closed = False
        self._interpreter = self._new_interpreter()
        _register_session(self)

    def _new_interpreter(self) -> dspy.PythonInterpreter:
        interp = dspy.PythonInterpreter()
        self._pending_interp = interp
        try:
            interp.execute(self.preamble + "\n" + self.user_code)
        except BaseException:
            _kill_interpreter_process(interp)
            raise
        finally:
            self._pending_interp = None
        return interp

    @property
    def interpreter(self) -> dspy.PythonInterpreter:
        return self._interpreter

    def rebuild_after_timeout(self) -> None:
        _kill_interpreter_process(self._interpreter)
        self._interpreter = None
        try:
            self._interpreter = self._new_interpreter()
        except Exception as exc:
            _kill_interpreter_process(self._pending_interp)
            raise RuntimeError(f"sandbox rebuild failed: {exc}") from exc

    def shutdown(self) -> None:
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
    code = code_string.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _validate_tool_tree(tree: ast.Module) -> "str | None":
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


def _kill_interpreter_process(interpreter: "dspy.PythonInterpreter | None") -> None:
    if interpreter is None:
        return
    proc = getattr(interpreter, "deno_process", None)
    try:
        if proc is not None and proc.poll() is None:
            # Close stdin first so deno can exit cleanly if it's responsive.
            stdin = getattr(proc, "stdin", None)
            try:
                if stdin is not None:
                    stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
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
    tree = ast.parse(source)
    functions = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        def _annotation_text(annotation_node) -> "str | None":
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

        docstring = ast.get_docstring(node) or f"Dynamically generated function: {node.name}"
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
) -> Tuple[List[Callable], "SandboxSession | None", "str | None", List[str]]:
    diagnostics: list[str] = []
    code = _clean_code_block(code_string)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [], None, f"SyntaxError in generated code: {e}", diagnostics

    validation_error = _validate_tool_tree(tree)
    if validation_error:
        return [], None, validation_error, diagnostics

    func_infos = _extract_function_info(code)

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

    tools = []
    for info in func_infos:
        if not info["docstring"] or info["docstring"].startswith("Dynamically generated"):
            diagnostics.append(
                f"Warning: {info['name']} has no docstring; using generic description."
            )
        try:
            wrapper = _make_sandbox_wrapper(info, session)
            tools.append(wrapper)
        except Exception as e:
            diagnostics.append(f"Warning: failed to wrap {info['name']}: {e}")

    if not tools:
        session.shutdown()
        return [], None, "All function wrappers failed.", diagnostics

    return tools, session, None, diagnostics


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _wrap_math_answer(value) -> str:
    value = str(value).strip()
    if not value.startswith("$"):
        value = f"${value}$"
    return value


def metric_fnv4(example, pred, trace=None):
    if parse is None or verify is None:
        raise RuntimeError(
            "metric_fnv4 requires math_verify to be installed, but it could not be imported."
        )
    try:
        gold = parse(_wrap_math_answer(example.answer), parsing_timeout=None)
        predicted = parse(_wrap_math_answer(pred.answer), parsing_timeout=None)
        return int(verify(gold, predicted, timeout_seconds=None))
    except Exception:
        return 0


PRIMITIVE_TOOL_PROMPT_PATCH = """

Primitive tools already available:
- calculator(expr_latex: str, timeout: int = 30) -> str evaluates LaTeX arithmetic expressions with SymPy.
- equation_solver(variables_latex: str, equations_latex: str, timeout: int = 30) -> str solves LaTeX equations with SymPy.

When synthesizing tools, create only additional problem-specific tools beyond these two primitives.
Do not duplicate, wrap, reimplement, or rename calculator/equation_solver.
During ReAct, use these primitive tools directly when arithmetic simplification or equation solving is enough.
A general-purpose python_eval tool is also available during ReAct: it executes
arbitrary Python (sympy, math, Fraction, itertools preloaded) for iterative
computation and verification. Prefer re-checking a suspicious tool output
(e.g. 0 or empty) with python_eval over trusting it blindly.
"""


def apply_primitive_tool_prompt_patch(program: dspy.Module) -> None:
    """Patch loaded optimized prompts so dynamic synthesis excludes primitives."""
    for name, predictor in program.named_predictors():
        signature = getattr(predictor, "signature", None)
        instructions = getattr(signature, "instructions", None)
        if not instructions or "Primitive tools already available:" in instructions:
            continue
        if name in {"tool_creator.predict", "react.reasoner"}:
            try:
                signature.instructions = instructions.rstrip() + PRIMITIVE_TOOL_PROMPT_PATCH
            except Exception as exc:
                print(f"Warning: could not patch prompt for {name}: {exc}", flush=True)


def _stringify_for_feedback(value: Any, limit: int | None = None) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(value)
    if limit is not None and len(text) > limit:
        return text[:limit] + "\n... [truncated]"
    return text


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric. math_verify-based score + textual feedback.

    Feedback is held at the same "answer-level" as the python_eval baseline
    (interpreter.py): score + correct answer + solution, with no trajectory,
    tool reasoning, or tool_synthesis_status echoed. This keeps the shared
    ReAct predictors (react.reasoner / react.extractor.predict) on an
    optimization signal identical to the baseline, so a performance comparison
    isolates the synthesized-tool variable rather than feedback richness.

    The ONLY exception: the synthesis-specific predictor 'tool_creator.predict'
    additionally sees its own generated tool code, otherwise GEPA cannot tell
    what code it produced and can only guess from the final answer.
    """
    if parse is None or verify is None:
        raise RuntimeError("metric_with_feedback requires math_verify to be installed.")

    correct_answer = str(example["answer"]).strip()
    written_solution = str(example.get("solution", "")).strip()
    predicted_answer = str(getattr(prediction, "answer", "")).strip()

    def add_tool_creator_code(feedback: str) -> str:
        # Only the synthesis-specific predictor sees the generated code; the
        # shared react.reasoner / extractor receive the bare feedback unchanged
        # so their signal matches the interpreter.py baseline byte-for-byte.
        if pred_name != "tool_creator.predict":
            return feedback
        generated_code = str(getattr(prediction, "generated_tool_code", "")).strip()
        if not generated_code:
            return feedback
        return feedback + "\n\nGenerated tool code from this rollout:\n" + generated_code

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
        return dspy.Prediction(score=0, feedback=add_tool_creator_code(feedback_text))

    if score == 1:
        feedback_text = (
            "Your answer is mathematically correct under symbolic verification. "
            f"The correct answer is '{correct_answer}'."
        )
    else:
        feedback_text = (
            "Your answer is mathematically incorrect under symbolic verification. "
            f"The correct answer is '{correct_answer}'. "
            "Check algebraic manipulation, numeric simplification, and whether your final answer format is clean."
        )
        if written_solution:
            feedback_text += (
                " Here's the full step-by-step solution:\n"
                f"{written_solution}\n\n"
                "Study the solution carefully and extract reusable strategies for similar problems."
            )

    return dspy.Prediction(score=score, feedback=add_tool_creator_code(feedback_text))


# ---------------------------------------------------------------------------
# Persistent ReAct (the actual fix)
# ---------------------------------------------------------------------------

class PersistentReAct(dspy.Module):
    """ReAct loop whose decision and extraction Predicts are persistent.

    Tools are supplied at forward() time, so the per-step Predict sees them
    via an InputField rather than a baked-in instruction. `self.reasoner` and
    `self.extractor` live on `self`, so an optimiser walking
    `named_parameters()` finds them and can rewrite their instructions.
    """

    def __init__(self, max_iters: int = 6):
        super().__init__()
        self.max_iters = max_iters
        self.reasoner = dspy.Predict(ReActStepSignature)
        self.extractor = dspy.ChainOfThought(ExtractAnswerSignature)

    @staticmethod
    def _describe_tools(tools: list[Callable]) -> str:
        lines = []
        for tool in tools:
            try:
                sig = str(inspect.signature(tool))
            except (TypeError, ValueError):
                sig = "(...)"
            doc = (getattr(tool, "__doc__", "") or "").strip()
            name = getattr(tool, "__name__", repr(tool))
            lines.append(f"- {name}{sig}\n    {doc}")
        lines.append(
            "- finish()\n    Marks the task as complete. "
            "Call this once you have a verified answer."
        )
        return "\n".join(lines)

    @staticmethod
    def _format_trajectory(trajectory: dict[str, Any]) -> str:
        if not trajectory:
            return "(no steps yet)"
        return "\n".join(f"{k}: {v}" for k, v in trajectory.items())

    @staticmethod
    def _coerce_args(raw_args: Any) -> dict[str, Any]:
        if raw_args is None or raw_args == "":
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def forward(self, problem: str, tools: list[Callable]) -> dspy.Prediction:
        tools_dict = {tool.__name__: tool for tool in tools}
        tools_desc = self._describe_tools(tools)

        trajectory: dict[str, Any] = {}

        for idx in range(self.max_iters):
            step = self.reasoner(
                problem=problem,
                tools_description=tools_desc,
                trajectory=self._format_trajectory(trajectory),
            )

            tool_name = (step.next_tool_name or "").strip()
            tool_args = self._coerce_args(step.next_tool_args)

            trajectory[f"thought_{idx}"] = step.next_thought
            trajectory[f"tool_name_{idx}"] = tool_name
            trajectory[f"tool_args_{idx}"] = tool_args

            if tool_name == "finish":
                trajectory[f"observation_{idx}"] = "Completed."
                break

            if tool_name not in tools_dict:
                trajectory[f"observation_{idx}"] = (
                    f"Error: tool '{tool_name}' not found. "
                    f"Available: {list(tools_dict.keys()) + ['finish']}"
                )
                continue

            try:
                observation = tools_dict[tool_name](**tool_args)
            except Exception as exc:
                observation = (
                    f"Execution error in {tool_name}: "
                    f"{type(exc).__name__}: {exc}"
                )
            trajectory[f"observation_{idx}"] = observation

        final = self.extractor(
            problem=problem,
            trajectory=self._format_trajectory(trajectory),
        )
        return dspy.Prediction(
            reasoning=final.reasoning,
            answer=final.answer,
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Solver module
# ---------------------------------------------------------------------------

class ReActMathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tool_creator = dspy.ChainOfThought(MultipleToolCreatorSignature)
        # Persistent ReAct — discoverable by GEPA via named_predictors().
        self.react = PersistentReAct(max_iters=6)

    def forward(self, problem: str) -> dspy.Prediction:
        session = None
        tool_debug: list[str] = []
        python_codes = ""
        tool_creator_reasoning = ""
        synthesized_tool_names: list[str] = []
        fallback_used = False
        creation_error_message = ""
        try:
            tool_result = self.tool_creator(problem=problem)
            python_codes = tool_result.python_codes
            tool_creator_reasoning = str(getattr(tool_result, "reasoning", ""))

            tools, session, err, diagnostics = create_sandboxed_tools(python_codes)
            tool_debug.extend(diagnostics)
            synthesized_tool_names = [getattr(tool, "__name__", repr(tool)) for tool in tools]
            creation_error_message = err or ""

            if err:
                # No repair stage: a broken one-shot tool is rarely worth
                # rescuing. Fall straight back to the python_eval interpreter
                # path, which is the stronger baseline.
                fallback_used = True
                tool_debug.append(f"Tool creation failed: {err}")
                tool_debug.append("Falling back to python_eval-only tooling.")
                if session is not None:
                    try:
                        session.shutdown()
                    except Exception:
                        pass
                    session = None
                tools = []
                synthesized_tool_names = []

            # python_eval is ALWAYS available as a co-tool, alongside any
            # synthesized tools and the primitives. This lets the reasoner
            # iteratively write/fix code in-loop (the interpreter behaviour
            # that beats one-shot synthesis), instead of being stuck with a
            # single solve() call.
            if session is None:
                preamble = (
                    "import sympy\n"
                    "import math\n"
                    "from fractions import Fraction\n"
                    "from itertools import combinations, permutations, product\n"
                )
                session = SandboxSession(preamble=preamble, user_code="")

            def python_eval(code: str) -> str:
                """Execute arbitrary Python in the sandbox and return stdout.

                sympy, math, Fraction, and itertools (combinations,
                permutations, product) are preloaded. Use this to compute a
                result directly, or to re-check / verify when another tool
                returns a suspicious value such as 0, an empty result, or
                something inconsistent with the problem.
                """
                try:
                    return _execute_in_sandbox(
                        session, code, TOOL_TIMEOUT_SECONDS
                    )
                except Exception as exc:
                    return f"Execution error: {type(exc).__name__}: {exc}"

            available_tools = tools + [python_eval] + PRIMITIVE_TOOLS
            prediction = self.react(problem=problem, tools=available_tools)

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
                        tool_debug.append(
                            f"ReAct tool error [{key}]: {value_text[:500]}"
                        )

            tool_synthesis_status = {
                "primitive_tool_source": PRIMITIVE_TOOL_SOURCE,
                "primitive_tool_import_error": PRIMITIVE_TOOL_IMPORT_ERROR,
                "primitive_tool_names": [tool.__name__ for tool in PRIMITIVE_TOOLS],
                "tool_creation_attempted": True,
                "tool_creation_succeeded": bool(synthesized_tool_names)
                and not creation_error_message,
                "tool_creation_error": creation_error_message,
                "fallback_python_eval_used": fallback_used,
                "synthesized_tool_names": synthesized_tool_names,
                "available_tool_names": [tool.__name__ for tool in available_tools],
                "generated_tool_code_present": bool(str(python_codes).strip()),
                "diagnostics": list(tool_debug),
            }

            return dspy.Prediction(
                answer=prediction.answer,
                reasoning=getattr(prediction, "reasoning", ""),
                trajectory=trajectory,
                full_trajectory_text=PersistentReAct._format_trajectory(trajectory or {}),
                tool_debug="\n".join(tool_debug),
                synthesized_tool_names=synthesized_tool_names,
                primitive_tool_names=[tool.__name__ for tool in PRIMITIVE_TOOLS],
                primitive_tool_source=PRIMITIVE_TOOL_SOURCE,
                primitive_tool_import_error=PRIMITIVE_TOOL_IMPORT_ERROR,
                available_tool_names=[tool.__name__ for tool in available_tools],
                tool_synthesis_status=tool_synthesis_status,
                generated_tool_code=python_codes,
                generated_tool_reasoning=tool_creator_reasoning,
            )

        except Exception as e:
            tb = traceback.format_exc()
            tool_debug.append(f"Forward error: {type(e).__name__}: {e}\n{tb}")
            # Print immediately so the full traceback (file + line of the real
            # culprit) lands in stdout/.out right at the failing problem,
            # instead of only the one-line message buried in the JSON log.
            print(
                f"!!! Forward exception: {type(e).__name__}: {e}\n{tb}",
                flush=True,
            )
            tool_synthesis_status = {
                "primitive_tool_source": PRIMITIVE_TOOL_SOURCE,
                "primitive_tool_import_error": PRIMITIVE_TOOL_IMPORT_ERROR,
                "primitive_tool_names": [tool.__name__ for tool in PRIMITIVE_TOOLS],
                "tool_creation_attempted": bool(str(python_codes).strip()),
                "tool_creation_succeeded": False,
                "tool_creation_error": creation_error_message,
                "fallback_python_eval_used": fallback_used,
                "synthesized_tool_names": synthesized_tool_names,
                "available_tool_names": [
                    *synthesized_tool_names,
                    "python_eval",
                    *[tool.__name__ for tool in PRIMITIVE_TOOLS],
                ],
                "generated_tool_code_present": bool(str(python_codes).strip()),
                "diagnostics": list(tool_debug),
            }
            # No hardcoded sentinel answer: a crash should not masquerade as a
            # computed "0" (which is sometimes the real answer and pollutes the
            # metric / GEPA feedback). Return an empty answer so the failure is
            # honest.
            return dspy.Prediction(
                answer="",
                tool_debug="\n".join(tool_debug),
                synthesized_tool_names=synthesized_tool_names,
                primitive_tool_names=[tool.__name__ for tool in PRIMITIVE_TOOLS],
                primitive_tool_source=PRIMITIVE_TOOL_SOURCE,
                primitive_tool_import_error=PRIMITIVE_TOOL_IMPORT_ERROR,
                tool_synthesis_status=tool_synthesis_status,
                generated_tool_code=python_codes,
                generated_tool_reasoning=tool_creator_reasoning,
            )
        finally:
            if session is not None:
                try:
                    session.shutdown()
                except Exception:
                    pass
            if GC_EVERY_FORWARD:
                gc.collect()


# ---------------------------------------------------------------------------
# Logging / sandbox execution helpers
# ---------------------------------------------------------------------------

LOG_DIR = SCRIPT_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)
GEPA_RUN_DIR = SCRIPT_DIR / "gepa_runs_persistent_equation_primitives"
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


def _open_fd_count(pid: int | None = None) -> "int | None":
    if os.name != "posix":
        return None
    fd_dir = Path("/proc") / str(pid or os.getpid()) / "fd"
    try:
        return len(list(fd_dir.iterdir()))
    except Exception:
        return None


def _active_sandbox_processes() -> list[tuple[int, dict[str, str]]]:
    with _ACTIVE_SESSIONS_LOCK:
        sessions = list(_ACTIVE_SESSIONS)
    children = []
    for session in sessions:
        for interpreter in (
            getattr(session, "_pending_interp", None),
            getattr(session, "_interpreter", None),
        ):
            proc = getattr(interpreter, "deno_process", None)
            if proc is not None and proc.poll() is None:
                child_status = _read_proc_status(proc.pid)
                child_status["OpenFDs"] = str(_open_fd_count(proc.pid))
                children.append((proc.pid, child_status))
    return children


def log_resource_snapshot(label: str, log_path: "Path | None" = None) -> None:
    if os.name != "posix":
        return
    main_status = _read_proc_status(os.getpid())
    children = _active_sandbox_processes()
    line = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "pid": os.getpid(),
        "main": main_status,
        "open_fds": _open_fd_count(),
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


def start_resource_monitor(log_path: Path, interval_seconds: int = 30) -> "threading.Thread | None":
    if os.name != "posix":
        return None

    def monitor() -> None:
        while not _RESOURCE_MONITOR_STOP.wait(interval_seconds):
            log_resource_snapshot("periodic", log_path)

    thread = threading.Thread(target=monitor, name="resource-monitor", daemon=True)
    thread.start()
    return thread


def _execute_in_sandbox(session: SandboxSession, code: str, timeout_seconds: int) -> str:
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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
        for attr in (
            "synthesized_tool_names",
            "primitive_tool_names",
            "primitive_tool_source",
            "primitive_tool_import_error",
            "available_tool_names",
            "tool_synthesis_status",
            "full_trajectory_text",
            "generated_tool_code",
            "generated_tool_reasoning",
        ):
            value = getattr(pred, attr, None)
            if value:
                entry[attr] = value

        if entry.get("synthesized_tool_names") or entry.get("primitive_tool_names"):
            print("\n--- Tools ---")
            print(f"  synthesized: {entry.get('synthesized_tool_names', [])}")
            print(f"  primitive:   {entry.get('primitive_tool_names', [])}")
            print(f"  source:      {entry.get('primitive_tool_source', '')}")
        if entry.get("tool_synthesis_status"):
            print("\n--- Tool Synthesis Status ---")
            print(_stringify_for_feedback(entry["tool_synthesis_status"], limit=2500))
        if entry.get("generated_tool_reasoning"):
            print("\n--- Tool Creator Reasoning ---")
            print(str(entry["generated_tool_reasoning"])[:1200])
        if entry.get("generated_tool_code"):
            print("\n--- Synthesized Tool Code ---")
            print(str(entry["generated_tool_code"])[:2000])
        if trajectory:
            print("\n--- Trajectory ---")
            for key in sorted(entry["trajectory"]):
                print(f"  {key}: {entry['trajectory'][key][:500]}")

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

    print("\n===== ReAct + Synthesized Tools (Persistent) =====")
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
        auto="light",
        num_threads=DEFAULT_GEPA_NUM_THREADS,
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
    print(f"GEPA experiment seed: {GEPA_EXPERIMENT_SEED}", flush=True)
    raise_file_descriptor_limit()
    resource_log_path = LOG_DIR / f"resource_persistent_equation_primitives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    resource_monitor = start_resource_monitor(resource_log_path)
    log_resource_snapshot("startup", resource_log_path)
    try:
        train_examples, val_examples, test_examples = init_dataset()
        print(
            "Dataset split: "
            f"train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)}"
        )

        mode = sys.argv[1] if len(sys.argv) > 1 else "train"

        student_lm = dspy.LM(
            model="openai//disk/scratch/s2799944/Qwen3.5-9B",
            api_base="http://localhost:8001/v1",
            api_key="EMPTY",
            max_tokens=16384,
            cache=True,
            num_retries=0,
            timeout=600,
        )
        dspy.configure(lm=student_lm)

        math_solver = ReActMathSolver()

        # Sanity check: GEPA should now see 4 sub-predicts, not 2.
        named = [n for n, _ in math_solver.named_predictors()]
        print(f"Discovered predictors ({len(named)}): {named}")

        if mode == "test":
            example_index = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            if not 1 <= example_index <= len(test_examples):
                raise ValueError(
                    f"test index must be between 1 and {len(test_examples)}"
                )
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
                print(json.dumps(
                    {k: str(v) for k, v in trajectory.items()},
                    ensure_ascii=False,
                    indent=2,
                ))
            score = metric_fnv4(example, prediction)
            print(f"Metric: {score}")
        else:
            log_path = LOG_DIR / f"react_syn_persistent_equation_primitives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            teacher_lm = build_reflection_lm()
            optimizer = build_optimizer(teacher_lm)
            log_resource_snapshot("before_gepa_compile", resource_log_path)
            optimized_solver = optimizer.compile(
                math_solver,
                trainset=train_examples,
                valset=val_examples,
            )
            log_resource_snapshot("after_gepa_compile", resource_log_path)
            optimized_result = evaluate_with_logging(
                optimized_solver, test_examples, log_path
            )
            print("Optimized result:", optimized_result)
            optimized_solver.save(
                str(SCRIPT_DIR / "optimized_react_syn_persistent_equation_primitives.json")
            )
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
