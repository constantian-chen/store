# -*- coding: utf-8 -*-
"""ReAct math solver with a single python_eval tool, optimized by GEPA."""

import argparse
import atexit
import contextlib
import json
import os
import sys
import threading
import weakref
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / ".dspy_cache_python_eval"
CACHE_DIR = Path(
    os.environ.get("PYTHON_EVAL_DSPY_CACHE_DIR", DEFAULT_CACHE_DIR)
).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DSPY_CACHEDIR"] = str(CACHE_DIR)

import dspy
from dspy import GEPA
from omni_split_loader import init_omni_math_dataset
from reflection_lm import build_reflection_lm

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None
    print("Warning: math_verify is not installed; metric-based evaluation will fail fast.")

dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_cache_dir=str(CACHE_DIR),
    disk_size_limit_bytes=int(os.environ.get("PYTHON_EVAL_DSPY_CACHE_LIMIT_BYTES", 20 * 1024**3)),
    memory_max_entries=int(os.environ.get("PYTHON_EVAL_DSPY_MEMORY_MAX_ENTRIES", 10000)),
)

LOG_DIR = SCRIPT_DIR / "log"
LOG_DIR.mkdir(exist_ok=True)
GEPA_RUN_DIR = SCRIPT_DIR / "python_eval_gepa_runs"
GEPA_RUN_DIR.mkdir(exist_ok=True)
_INTERPRETER_STATE = threading.local()


def build_student_lm() -> dspy.LM:
    """Student model used for task execution."""
    return dspy.LM(
        model="openai//disk/scratch/s2799944/Qwen3.5-9B",
        api_base="http://localhost:8001/v1",
        api_key="EMPTY",
        max_tokens=16384,
        cache=True,
        num_retries=0,
        timeout=600,
    )


def configure_lm() -> None:
    """Configure DSPy with the local student model."""
    dspy.configure(lm=build_student_lm())


def enable_mlflow_autolog() -> None:
    """Optional helper for MLflow DSPy autologging."""
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("DSPy")
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces=True,
    )


def init_dataset():
    """Load the frozen Omni-MATH-Rule split used across omini experiments."""
    return init_omni_math_dataset()


_ALL_INTERPRETERS: "weakref.WeakSet[dspy.PythonInterpreter]" = weakref.WeakSet()
_ALL_INTERPRETERS_LOCK = threading.Lock()


def _register_interpreter(interp: "dspy.PythonInterpreter") -> None:
    with _ALL_INTERPRETERS_LOCK:
        _ALL_INTERPRETERS.add(interp)


def _kill_interpreter_process(interpreter: "dspy.PythonInterpreter | None") -> None:
    """Force-kill an interpreter's deno subprocess and close its pipes."""
    if interpreter is None:
        return
    proc = getattr(interpreter, "deno_process", None)
    if proc is not None and proc.poll() is None:
        stdin = getattr(proc, "stdin", None)
        if stdin is not None:
            with contextlib.suppress(Exception):
                stdin.close()
        try:
            proc.wait(timeout=2)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
    if proc is not None:
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(proc, stream_name, None)
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
    with contextlib.suppress(Exception):
        interpreter.deno_process = None


def _reset_current_interpreter() -> None:
    interp = getattr(_INTERPRETER_STATE, "interp", None)
    if interp is not None:
        with contextlib.suppress(Exception):
            interp.shutdown()
        # Belt-and-braces: even if shutdown raised or left the subprocess alive.
        _kill_interpreter_process(interp)
    _INTERPRETER_STATE.interp = None


def _new_interpreter() -> dspy.PythonInterpreter:
    interp = dspy.PythonInterpreter()
    try:
        interp.start()
    except BaseException:
        _kill_interpreter_process(interp)
        raise
    _register_interpreter(interp)
    return interp


def _shutdown_all_interpreters() -> None:
    with _ALL_INTERPRETERS_LOCK:
        interps = list(_ALL_INTERPRETERS)
    for interp in interps:
        with contextlib.suppress(Exception):
            interp.shutdown()
        _kill_interpreter_process(interp)


atexit.register(_shutdown_all_interpreters)


def _should_reset_interpreter(exc: Exception) -> bool:
    message = str(exc)
    return isinstance(exc, (TimeoutError, MemoryError, BrokenPipeError)) or any(
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


@contextlib.contextmanager
def problem_interpreter_context():
    """Provide one PythonInterpreter for a single problem/program call."""
    previous_interp = getattr(_INTERPRETER_STATE, "interp", None)
    previous_problem_context = getattr(_INTERPRETER_STATE, "in_problem_context", False)
    interp = _new_interpreter()
    _INTERPRETER_STATE.interp = interp
    _INTERPRETER_STATE.in_problem_context = True
    try:
        yield interp
    finally:
        current_interp = getattr(_INTERPRETER_STATE, "interp", None)
        if current_interp is not None and current_interp is not interp:
            with contextlib.suppress(Exception):
                current_interp.shutdown()
        if current_interp is interp:
            with contextlib.suppress(Exception):
                interp.shutdown()
        _INTERPRETER_STATE.interp = previous_interp
        _INTERPRETER_STATE.in_problem_context = previous_problem_context


def python_eval(code: str) -> str:
    """Execute Python code using dspy sandbox."""
    interp = getattr(_INTERPRETER_STATE, "interp", None)
    created_for_call = False
    if interp is None:
        interp = _new_interpreter()
        _INTERPRETER_STATE.interp = interp
        created_for_call = not getattr(_INTERPRETER_STATE, "in_problem_context", False)

    try:
        return str(interp(code))
    except TimeoutError as e:
        _reset_current_interpreter()
        return f"Error: {e}"
    except Exception as e:
        if _should_reset_interpreter(e):
            _reset_current_interpreter()
        return f"Execution error: {type(e).__name__}: {e}"
    finally:
        if created_for_call:
            _reset_current_interpreter()


python_eval_tool = dspy.Tool(
    func=python_eval,
    name="python_eval",
    desc=(
        "Execute Python code in a secure sandbox and return the output. "
        "You may import math, sympy, itertools, fractions, etc. "
        "Use print() to output results. "
        "Do NOT include markdown fences or explanations in the code string."
    ),
)


class MathSolver(dspy.Signature):
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
    - Use python_eval for nontrivial arithmetic, symbolic manipulation, exhaustive checks, or verification.
    - Write each python_eval call as self-contained code; the sandbox is shared within a problem but may reset after failures.
    - Use print() to output intermediate or final computed values.
    - If a call errors, read the message and fix the exact issue before retrying.
    - When finishing, call the finish tool with no arguments: finish(). Its argument object must be exactly {}.
    - The final answer belongs only in the answer field, not in tool arguments. Do not pass answer, result, or any other key to finish.
    - Finish as soon as you have a verified final answer.
    """

    problem: str = dspy.InputField(desc="Mathematical problem statement")
    answer: str = dspy.OutputField(desc="Final answer only")


class ProblemScopedReAct(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(MathSolver, tools=[python_eval_tool], max_iters=6)

    def forward(self, problem: str):
        with problem_interpreter_context():
            return self.react(problem=problem)


def _wrap_math_answer(value) -> str:
    value = str(value).strip()
    if not value.startswith("$"):
        value = f"${value}$"
    return value


def metric_fnv4(example, pred, trace=None) -> int:
    if parse is None or verify is None:
        raise RuntimeError("metric_fnv4 requires math_verify to be installed.")
    try:
        gold = parse(_wrap_math_answer(example.answer), parsing_timeout=None)
        predicted = parse(_wrap_math_answer(pred.answer), parsing_timeout=None)
        return int(verify(gold, predicted, timeout_seconds=None))
    except Exception:
        return 0


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Optimization metric for GEPA with math_verify-based feedback."""
    if parse is None or verify is None:
        raise RuntimeError("metric_with_feedback requires math_verify to be installed.")

    correct_answer = str(example["answer"]).strip()
    written_solution = str(example.get("solution", "")).strip()
    predicted_answer = str(getattr(prediction, "answer", "")).strip()

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
        return dspy.Prediction(score=0, feedback=feedback_text)

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

    return dspy.Prediction(score=score, feedback=feedback_text)


def run_one(program, ex, idx: int, total: int) -> dict:
    entry = {
        "id": idx,
        "problem": ex.problem,
        "expected": ex.answer,
        "predicted": None,
        "status": None,
        "metric": 0,
        "trajectory": None,
    }

    print(f"\n{'=' * 70}")
    print(f"[{idx}/{total}] {ex.problem}")
    print(f"Expected: {ex.answer}")
    print(f"{'=' * 70}")

    try:
        pred = program(problem=ex.problem)
        predicted = str(getattr(pred, "answer", "")).strip()
        raw_traj = dict(getattr(pred, "trajectory", {}) or {})
        trajectory = {k: str(v) for k, v in raw_traj.items()}
        score = metric_fnv4(ex, pred)

        entry["predicted"] = predicted
        entry["trajectory"] = trajectory
        entry["metric"] = score
        entry["status"] = "correct" if score == 1 else "wrong"

        if trajectory:
            print("\n--- Trajectory ---")
            for k in sorted(trajectory):
                print(f"  {k}: {trajectory[k][:300]}")
        print(f"\nPredicted: {predicted}")
        print(f"Result:    {'CORRECT' if score == 1 else 'WRONG'}")

    except Exception as exc:
        err = str(exc)
        entry["predicted"] = f"ERROR: {err}"
        entry["status"] = "error"
        print(f"EXCEPTION: {err}")

    return entry


def evaluate_with_logging(program, examples, log_path: Path, start_idx: int = 1) -> dict:
    results = []
    correct = []
    total = len(examples)

    print("\n=== Optimized ReAct + python_eval evaluation ===")
    print(f"Log -> {log_path}")

    for i in range(start_idx, total + 1):
        entry = run_one(program, examples[i - 1], i, total)
        results.append(entry)
        if entry["status"] == "correct":
            correct.append(i)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "start_problem": start_idx,
                        "last_completed": i,
                        "correct": correct,
                        "accuracy": f"{len(correct)}/{len(results)}",
                    },
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return {
        "correct_count": len(correct),
        "total": len(results),
        "accuracy": len(correct) / len(results) if results else 0.0,
        "log_path": str(log_path),
    }


def build_program():
    return ProblemScopedReAct()


def build_optimizer():
    return GEPA(
        metric=metric_with_feedback,
        max_metric_calls=6900,
        num_threads=4,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=build_reflection_lm(),
        log_dir=str(GEPA_RUN_DIR),
    )


def print_optimized_instructions(program) -> None:
    print("\n=== Optimized prompt instructions ===")
    if hasattr(program, "named_predictors"):
        found = False
        for name, predictor in program.named_predictors():
            signature = getattr(predictor, "signature", None)
            instructions = getattr(signature, "instructions", None)
            if instructions:
                found = True
                print(f"\n--- {name} ---")
                print(instructions)
        if found:
            return

    signatures = []
    for value in vars(program).values():
        signature = getattr(getattr(value, "predict", None), "signature", None)
        instructions = getattr(signature, "instructions", None)
        if instructions:
            signatures.append(str(instructions))

    if signatures:
        for idx, instructions in enumerate(signatures, 1):
            print(f"\n--- Signature {idx} ---")
            print(instructions)
    else:
        print("No optimized signature instructions found on the compiled program.")


def main(
    use_mlflow: bool = False,
    start_problem: int = 1,
    evaluate: bool = True,
    smoke_test_problem: int | None = None,
):
    configure_lm()
    if use_mlflow:
        enable_mlflow_autolog()

    train_set, val_set, test_set = init_dataset()
    print("Dataset sizes:", len(train_set), len(val_set), len(test_set))
    print("\nExample problem:\n")
    print(train_set[0]["problem"])
    print("\nExample answer:\n")
    print(train_set[0]["answer"])

    if not (1 <= start_problem <= len(test_set)):
        raise ValueError(f"--problem must be between 1 and {len(test_set)}")

    program = build_program()

    if smoke_test_problem is not None:
        if not (1 <= smoke_test_problem <= len(test_set)):
            raise ValueError(f"--smoke-test must be between 1 and {len(test_set)}")
        print("Running unoptimized smoke test only; GEPA compile is skipped.")
        entry = run_one(
            program,
            test_set[smoke_test_problem - 1],
            smoke_test_problem,
            len(test_set),
        )
        print("Smoke test result:", entry)
        return {
            "program": program,
            "optimized_program": None,
            "optimized_results": {
                "correct_count": 1 if entry["status"] == "correct" else 0,
                "total": 1,
                "accuracy": 1.0 if entry["status"] == "correct" else 0.0,
                "entry": entry,
            },
        }

    optimizer = build_optimizer()
    optimized_program = optimizer.compile(
        program,
        trainset=train_set,
        valset=val_set,
    )

    print_optimized_instructions(optimized_program)

    optimized_results = None
    if evaluate:
        log_path = LOG_DIR / f"gepa_python_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimized_results = evaluate_with_logging(
            optimized_program,
            test_set,
            log_path,
            start_idx=start_problem,
        )
        print("Optimized result:", optimized_results)

    optimized_program.save(str(SCRIPT_DIR / "optimized_gepa_python_eval.json"))

    return {
        "program": program,
        "optimized_program": optimized_program,
        "optimized_results": optimized_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=int,
        default=1,
        help="1-indexed Omni-MATH test problem number to start evaluation from.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Compile with GEPA and save the optimized program without running the test set.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow DSPy autologging.",
    )
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=None,
        help="Run one unoptimized Omni-MATH test problem and skip GEPA compile.",
    )
    args = parser.parse_args()

    try:
        main(
            use_mlflow=args.mlflow,
            start_problem=args.problem,
            evaluate=not args.no_eval,
            smoke_test_problem=args.smoke_test,
        )
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down active interpreters...")
        _shutdown_all_interpreters()
        sys.exit(130)
