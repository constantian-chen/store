import os
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = SCRIPT_DIR / ".dspy_cache_cot"
CACHE_DIR = Path(
    os.environ.get("COT_DSPY_CACHE_DIR", DEFAULT_CACHE_DIR)
).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DSPY_CACHEDIR"] = str(CACHE_DIR)

import dspy
from dspy import GEPA
from math_verify import parse, verify
from omni_split_loader import init_omni_math_dataset
from reflection_lm import build_reflection_lm

GEPA_EXPERIMENT_SEED = int(os.environ.get("GEPA_EXPERIMENT_SEED", "42"))
random.seed(GEPA_EXPERIMENT_SEED)

dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_cache_dir=str(CACHE_DIR),
    disk_size_limit_bytes=int(os.environ.get("COT_DSPY_CACHE_LIMIT_BYTES", 20 * 1024**3)),
    memory_max_entries=int(os.environ.get("COT_DSPY_MEMORY_MAX_ENTRIES", 10000)),
)


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
    dspy.configure(
        lm=build_student_lm(),
    )


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


class GenerateResponse(dspy.Signature):
    """Solve the given competition-style math problem.

    Requirements:
    - Read the problem carefully and identify the requested final answer.
    - Use exact arithmetic whenever possible.
    - For fractions, reduce to lowest terms.
    - For modular/remainder questions, return the requested residue in the required format.
    - For counting, probability, algebra, and number theory problems, check constraints and edge cases.
    - The answer field must contain only the final answer, with no units or explanation.
    - Do not include LaTeX wrappers such as \\boxed{}, \\( ... \\), or $...$ unless the problem explicitly requires a symbolic expression.
    """

    problem: str = dspy.InputField(desc="Mathematical problem statement")
    answer: str = dspy.OutputField(desc="Final answer only")


def metric_fnv4(example, pred, trace=None):
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

    return dspy.Prediction(score=score, feedback=feedback_text)


def evaluate_one_by_one(program, examples):
    results = []
    correct = 0
    total = len(examples)

    print("\n=== Optimized evaluation (one by one) ===")
    for idx, example in enumerate(examples, 1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{total}]")
        print(example.problem)
        print(f"Expected: {example.answer}")

        try:
            prediction = program(problem=example.problem)
            predicted_answer = str(getattr(prediction, "answer", "")).strip()
            score = metric_fnv4(example, prediction)
            status = "CORRECT" if score == 1 else "WRONG"
            if score == 1:
                correct += 1

            print(f"Predicted: {predicted_answer}")
            print(f"Result:    {status}")
        except Exception as exc:
            predicted_answer = f"ERROR: {exc}"
            score = 0
            status = "ERROR"
            print(f"Predicted: {predicted_answer}")
            print(f"Result:    {status}")

        results.append(
            {
                "index": idx,
                "problem": example.problem,
                "expected": example.answer,
                "predicted": predicted_answer,
                "score": score,
                "status": status,
            }
        )

        print(f"Running accuracy: {correct}/{idx}")

    summary = {
        "correct_count": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "results": results,
    }

    print(f"\nFinal accuracy: {correct}/{total} = {summary['accuracy']:.4f}")
    return summary


def build_optimizer():
    return GEPA(
        metric=metric_with_feedback,
        max_metric_calls=6900,
        num_threads=4,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=build_reflection_lm(),
    )


def main(use_mlflow: bool = False):
    print(f"GEPA experiment seed: {GEPA_EXPERIMENT_SEED}", flush=True)
    configure_lm()
    if use_mlflow:
        enable_mlflow_autolog()

    train_set, val_set, test_set = init_dataset()

    print("Dataset sizes:", len(train_set), len(val_set), len(test_set))
    print("\nExample problem:\n")
    print(train_set[0]["problem"])
    print("\nExample answer:\n")
    print(train_set[0]["answer"])

    program = dspy.ChainOfThought(GenerateResponse)

    optimizer = build_optimizer()
    optimized_program = optimizer.compile(
        program,
        trainset=train_set,
        valset=val_set,
    )

    print("\n=== Optimized prompt instructions ===")
    print(optimized_program.predict.signature.instructions)

    optimized_results = evaluate_one_by_one(optimized_program, test_set)

    return {
        "program": program,
        "optimized_program": optimized_program,
        "optimized_results": optimized_results,
    }


if __name__ == "__main__":
    main(use_mlflow=False)
