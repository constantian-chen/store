"""Smoke-test what GEPA feedback can see for synthesized tools.

This does not run a full GEPA compile. It directly calls the metric with a
fake prediction and prints the feedback text for two predictor names:

- tool_creator.predict should receive generated tool code + ReAct trajectory.
- react.reasoner should receive only the bare answer-level feedback.
"""

from __future__ import annotations

import sys
import types


# The experiment module imports these project modules at import time. Stub them
# so this feedback test can run on a lightweight local environment.
omni = types.ModuleType("omni_split_loader")
omni.init_omni_math_dataset = lambda: ([], [], [])
sys.modules.setdefault("omni_split_loader", omni)

reflection = types.ModuleType("reflection_lm")
reflection.build_reflection_lm = lambda: None
sys.modules.setdefault("reflection_lm", reflection)

import dspy
import react_syn_persistent_equation_primitives_traj_src as experiment


def _fake_parse(value, parsing_timeout=None):
    return str(value)


def _fake_verify(gold, predicted, timeout_seconds=None):
    return gold == predicted


def main() -> None:
    # Avoid needing math_verify for this smoke test.
    experiment.parse = _fake_parse
    experiment.verify = _fake_verify

    example = {
        "answer": "42",
        "solution": "The reference solution computes the value as 42.",
    }
    prediction = dspy.Prediction(
        answer="41",
        generated_tool_code=(
            "def solve() -> str:\n"
            "    \"\"\"Incorrectly returns a hardcoded answer.\"\"\"\n"
            "    return \"41\"\n"
        ),
        full_trajectory_text=(
            "thought_0: I will call the synthesized solve tool.\n"
            "tool_name_0: solve\n"
            "tool_args_0: {}\n"
            "observation_0: 41\n"
            "thought_1: The tool returned 41, so I will finish.\n"
            "tool_name_1: finish\n"
            "observation_1: Completed."
        ),
        tool_synthesis_status={
            "tool_creation_succeeded": True,
            "synthesized_tool_names": ["solve"],
            "available_tool_names": ["solve", "calculator", "equation_solver"],
        },
    )

    for pred_name in ("tool_creator.predict", "react.reasoner"):
        result = experiment.metric_with_feedback(
            example,
            prediction,
            pred_name=pred_name,
        )
        print("\n" + "=" * 80)
        print(f"pred_name={pred_name}")
        print(f"score={result.score}")
        print("-" * 80)
        print(result.feedback)


if __name__ == "__main__":
    main()
