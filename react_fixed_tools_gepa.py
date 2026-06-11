"""
ReAct Math Solver with a fixed math toolset + GEPA optimization.

This variant removes dynamic tool synthesis. GEPA can optimize the registered
ReAct module directly because the fixed toolset is known at construction time.
"""

import json
import math
import os
import random
import sys
import threading
from datetime import datetime
from fractions import Fraction
from math import comb, factorial, perm
from pathlib import Path

import sympy as sp

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / ".dspy_cache_fixed_tools"
CACHE_DIR = Path(
    os.environ.get("FIXED_TOOLS_DSPY_CACHE_DIR", DEFAULT_CACHE_DIR)
).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DSPY_CACHEDIR"] = str(CACHE_DIR)

import dspy
from dspy import GEPA
from omni_split_loader import init_omni_math_dataset
from reflection_lm import build_reflection_lm

GEPA_EXPERIMENT_SEED = int(os.environ.get("GEPA_EXPERIMENT_SEED", "42"))
random.seed(GEPA_EXPERIMENT_SEED)

dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_cache_dir=str(CACHE_DIR),
    disk_size_limit_bytes=int(os.environ.get("FIXED_TOOLS_DSPY_CACHE_LIMIT_BYTES", 20 * 1024**3)),
    memory_max_entries=int(os.environ.get("FIXED_TOOLS_DSPY_MEMORY_MAX_ENTRIES", 10000)),
)

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None
    print("Warning: math_verify is not installed; metric-based evaluation will fail fast.")


# ---------------------------------------------------------------------------
# Fixed Math Tools
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression with exact rational arithmetic."""
    try:
        result = sp.sympify(expression, rational=True)
        return str(sp.nsimplify(result))
    except Exception as e:
        return f"Error: {e}"


def solve_equation(equation: str, variable: str = "x") -> str:
    """Solve one equation for one variable. Use 'lhs = rhs' equation format."""
    try:
        var = sp.Symbol(variable)
        lhs, rhs = equation.split("=")
        eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
        return str(sp.solve(eq, var))
    except Exception as e:
        return f"Error: {e}"


def solve_system(equations: str, variables: str) -> str:
    """Solve equations separated by semicolons for comma-separated variables."""
    try:
        var_list = [sp.Symbol(v.strip()) for v in variables.split(",")]
        eq_list = []
        for eq_str in equations.split(";"):
            lhs, rhs = eq_str.split("=")
            eq_list.append(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)))
        return str(sp.solve(eq_list, var_list, dict=True))
    except Exception as e:
        return f"Error: {e}"


def algebraic_manipulate(expression: str, operation: str) -> str:
    """Apply one of factor, expand, or simplify to an algebraic expression."""
    try:
        expr = sp.sympify(expression)
        ops = {"factor": sp.factor, "expand": sp.expand, "simplify": sp.simplify}
        if operation not in ops:
            return f"Error: unknown op {operation}"
        return str(ops[operation](expr))
    except Exception as e:
        return f"Error: {e}"


def number_theory(n: int, operation: str) -> str:
    """Run factorize, divisors, is_prime, or totient on an integer."""
    try:
        if operation == "factorize":
            return str(sp.factorint(n))
        if operation == "divisors":
            return str(sp.divisors(n))
        if operation == "is_prime":
            return str(sp.isprime(n))
        if operation == "totient":
            return str(sp.totient(n))
        return f"Error: unknown op {operation}"
    except Exception as e:
        return f"Error: {e}"


def combinatorics(operation: str, n: int, k: int = 0) -> str:
    """Run binomial, permutation, or factorial computations."""
    try:
        if operation == "binomial":
            return str(comb(n, k))
        if operation == "permutation":
            return str(perm(n, k))
        if operation == "factorial":
            return str(factorial(n))
        return f"Error: unknown op {operation}"
    except Exception as e:
        return f"Error: {e}"


def polynomial_roots(coefficients: str) -> str:
    """Find roots of a polynomial from comma-separated high-to-low coefficients."""
    try:
        coeffs = [sp.Rational(c.strip()) for c in coefficients.split(",")]
        x = sp.Symbol("x")
        poly = sum(c * x**i for i, c in enumerate(reversed(coeffs)))
        return str(sp.solve(poly, x))
    except Exception as e:
        return f"Error: {e}"


def modular_arithmetic(a: int, b: int, mod: int, operation: str) -> str:
    """Run add, mul, pow, or inverse modulo mod."""
    try:
        if operation == "add":
            return str((a + b) % mod)
        if operation == "mul":
            return str((a * b) % mod)
        if operation == "pow":
            return str(pow(a, b, mod))
        if operation == "inverse":
            return str(pow(a, -1, mod))
        return f"Error: unknown op {operation}"
    except Exception as e:
        return f"Error: {e}"


FIXED_MATH_TOOLS = [
    calculator,
    solve_equation,
    solve_system,
    algebraic_manipulate,
    number_theory,
    combinatorics,
    polynomial_roots,
    modular_arithmetic,
]


# ---------------------------------------------------------------------------
# Signature and Metrics
# ---------------------------------------------------------------------------

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
    - Use the provided fixed math tools for exact arithmetic, algebra, equations, combinatorics, number theory, polynomial roots, and modular arithmetic.
    - Use at least one tool call before giving the final answer.
    - If a tool returns an error, correct the arguments or choose a more appropriate tool.
    """

    problem: str = dspy.InputField(desc="Mathematical problem statement")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning and tool-use summary")
    answer: str = dspy.OutputField(desc="Final answer only")


def metric_fnv4(example, pred, trace=None):
    if parse is None or verify is None:
        raise RuntimeError("metric_fnv4 requires math_verify to be installed.")
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
    """GEPA feedback metric with fixed-tool execution diagnostics."""
    correct_answer = str(example["answer"]).strip()
    written_solution = str(example.get("solution", "")).strip()
    predicted_answer = str(getattr(prediction, "answer", "")).strip()
    tool_debug = str(getattr(prediction, "tool_debug", "")).strip()

    def add_tool_feedback(feedback: str) -> str:
        if not tool_debug:
            return feedback
        return (
            feedback
            + "\n\nFixed tool diagnostics from this rollout:\n"
            + tool_debug[:8000]
            + "\nUse the fixed tools with valid arguments. If a tool returns Error:, correct the arguments "
            "or use a different tool. The answer field must contain only the final answer."
        )

    try:
        gold = parse(_wrap_math_answer(correct_answer), parsing_timeout=None)
        predicted = parse(_wrap_math_answer(predicted_answer), parsing_timeout=None)
        score = int(verify(gold, predicted, timeout_seconds=None))
    except Exception:
        feedback_text = (
            "Your final answer could not be parsed or verified by math_verify. "
            f"You responded with '{predicted_answer[:500]}'. Return only the final mathematical answer."
        )
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f"\nReference solution:\n{written_solution}"
        return dspy.Prediction(score=0, feedback=add_tool_feedback(feedback_text))

    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = (
            f"Your answer is incorrect. The correct answer is '{correct_answer}'. "
            "Check the setup, tool arguments, arithmetic, and final answer formatting."
        )
        if written_solution:
            feedback_text += f"\nReference solution:\n{written_solution}"

    return dspy.Prediction(score=score, feedback=add_tool_feedback(feedback_text))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class FixedToolReActMathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            signature=FactCheckSignature,
            tools=FIXED_MATH_TOOLS,
            max_iters=int(os.environ.get("FIXED_REACT_MAX_ITERS", 6)),
        )

    def forward(self, problem: str) -> dspy.Prediction:
        tool_debug: list[str] = []
        try:
            prediction = self.react(problem=problem)
            trajectory = getattr(prediction, "trajectory", None)
            if trajectory:
                for key, value in trajectory.items():
                    value_text = str(value)
                    if "Error:" in value_text or "Exception" in value_text or "Traceback" in value_text:
                        tool_debug.append(f"Fixed tool error [{key}]: {value_text[:500]}")

            return dspy.Prediction(
                answer=getattr(prediction, "answer", ""),
                reasoning=getattr(prediction, "reasoning", ""),
                trajectory=trajectory,
                tool_debug="\n".join(tool_debug),
            )
        except Exception as e:
            return dspy.Prediction(
                answer=0,
                tool_debug=f"Forward error: {type(e).__name__}: {e}",
            )


# ---------------------------------------------------------------------------
# Logging and Training
# ---------------------------------------------------------------------------

LOG_DIR = SCRIPT_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)
FIXED_GEPA_RUN_DIR = SCRIPT_DIR / "fixed_gepa_runs"
FIXED_GEPA_RUN_DIR.mkdir(exist_ok=True)

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


def log_resource_snapshot(label: str, log_path: Path | None = None) -> None:
    if os.name != "posix":
        return

    line = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "pid": os.getpid(),
        "main": _read_proc_status(os.getpid()),
    }
    if log_path is not None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
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

    return entry


def evaluate_with_logging(program, examples, log_path: Path) -> dict:
    results = []
    correct = []
    total = len(examples)

    print("\n===== ReAct + Fixed Math Tools =====")
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
        num_threads=int(os.environ.get("GEPA_NUM_THREADS", 4)),
        track_stats=True,
        track_best_outputs=track_best_outputs,
        log_dir=str(FIXED_GEPA_RUN_DIR),
        reflection_minibatch_size=int(os.environ.get("GEPA_REFLECTION_MINIBATCH_SIZE", 3)),
        reflection_lm=reflection_lm,
    )


def init_dataset():
    """Load the frozen Omni-MATH-Rule split used across omini experiments."""
    return init_omni_math_dataset()


if __name__ == "__main__":
    print(f"GEPA experiment seed: {GEPA_EXPERIMENT_SEED}", flush=True)
    resource_log_path = LOG_DIR / f"resource_fixed_gepa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    resource_monitor = start_resource_monitor(resource_log_path)
    log_resource_snapshot("startup", resource_log_path)
    try:
        train_examples, val_examples, test_examples = init_dataset()

        student_lm = dspy.LM(
            model=os.environ.get("STUDENT_MODEL", "openai//disk/scratch/s2799944/Qwen3.5-9B"),
            api_base=os.environ.get("STUDENT_API_BASE", "http://localhost:8001/v1"),
            api_key=os.environ.get("STUDENT_API_KEY", "EMPTY"),
            max_tokens=int(os.environ.get("STUDENT_MAX_TOKENS", 16384)),
            cache=True,
            num_retries=0,
            timeout=int(os.environ.get("STUDENT_TIMEOUT", 600)),
        )

        teacher_lm = build_reflection_lm()

        dspy.configure(lm=student_lm)

        mode = sys.argv[1] if len(sys.argv) > 1 else "train"
        math_solver = FixedToolReActMathSolver()

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
            log_path = LOG_DIR / f"9breact_fixed_tools_gepa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            optimizer = build_optimizer(teacher_lm)
            log_resource_snapshot("before_fixed_gepa_compile", resource_log_path)
            optimized_solver = optimizer.compile(
                math_solver,
                trainset=train_examples,
                valset=val_examples,
            )
            log_resource_snapshot("after_fixed_gepa_compile", resource_log_path)
            optimized_result = evaluate_with_logging(optimized_solver, test_examples, log_path)
            print("Optimized result:", optimized_result)
            optimized_solver.save(str(SCRIPT_DIR / "optimized_9breact_fixed_tools_gepa.json"))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    finally:
        _RESOURCE_MONITOR_STOP.set()
        log_resource_snapshot("shutdown", resource_log_path)
