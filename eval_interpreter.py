# -*- coding: utf-8 -*-
"""
Eval-only variant of interpreter.py.

No GEPA training. Loads the GEPA-optimized prompts from
`optimized_gepa_python_eval.json` and runs them on the frozen Omni-MATH test
split, with the student model swapped to the 27B served by vLLM on :8001.

The vLLM server is launched (separately) with:
    vllm serve /disk/scratch/s2799944/Qwen3.5-27B \
      --host 0.0.0.0 --port 8001 --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.8 --max-model-len 65536 \
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --language-model-only

enable_thinking=false is handled server-side, so nothing extra is needed here.

Usage:
    python eval_interpreter.py                # full test set
    python eval_interpreter.py --problem 50   # start at test #50
    python eval_interpreter.py --test 3       # single test problem #3
"""

import argparse
import json
import sys
from datetime import datetime

import dspy

import interpreter as base

# Only change vs. training: 9B -> 27B student model.
STUDENT_MODEL = "openai//disk/scratch/s2799944/Qwen3.5-27B"
API_BASE = "http://localhost:8001/v1"
OPTIMIZED_PATH = base.SCRIPT_DIR / "optimized_gepa_python_eval.json"


def build_student_lm() -> dspy.LM:
    return dspy.LM(
        model=STUDENT_MODEL,
        api_base=API_BASE,
        api_key="EMPTY",
        max_tokens=16384,
        cache=True,
        num_retries=0,
        timeout=600,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=int,
        default=1,
        help="1-indexed Omni-MATH test problem to start evaluation from.",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help="Run a single 1-indexed test problem and exit.",
    )
    args = parser.parse_args()

    dspy.configure(lm=build_student_lm())

    train_set, val_set, test_set = base.init_dataset()
    print("Dataset sizes:", len(train_set), len(val_set), len(test_set))

    program = base.build_program()
    if not OPTIMIZED_PATH.exists():
        raise FileNotFoundError(f"Optimized prompts not found: {OPTIMIZED_PATH}")
    program.load(str(OPTIMIZED_PATH))
    print(f"Loaded optimized prompts <- {OPTIMIZED_PATH}")

    if args.test is not None:
        if not 1 <= args.test <= len(test_set):
            raise ValueError(f"--test must be between 1 and {len(test_set)}")
        entry = base.run_one(program, test_set[args.test - 1], args.test, len(test_set))
        print("Single-problem result:", json.dumps(entry, ensure_ascii=False, indent=2))
        return

    if not 1 <= args.problem <= len(test_set):
        raise ValueError(f"--problem must be between 1 and {len(test_set)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base.LOG_DIR / f"eval_gepa_python_eval_27b_{stamp}.json"

    result = base.evaluate_with_logging(
        program, test_set, log_path, start_idx=args.problem
    )
    print("Eval result (27B, optimized prompts):", result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down active interpreters...")
        base._shutdown_all_interpreters()
        sys.exit(130)
    finally:
        base._shutdown_all_interpreters()
