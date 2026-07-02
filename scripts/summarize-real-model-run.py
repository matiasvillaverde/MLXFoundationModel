#!/usr/bin/env python3
"""Build a machine-readable summary for serialized real-model runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ATTENTIONLESS_ARCHITECTURES = {"mamba", "mamba2", "rwkv7"}
NATIVE_TOOL_ARCHITECTURES = {
    "cohere",
    "cohere2",
    "deepseek_v4",
    "function_gemma",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    "gemma3n",
    "gemma4",
    "gemma4_text",
    "glm",
    "glm4",
    "glm4_moe",
    "glm4_moe_lite",
    "glm_moe_dsa",
    "gpt_oss",
    "kimi_k2",
    "kimi_k25",
    "longcat_flash",
    "longcat_flash_ngram",
    "minimax",
    "minimax_m3",
    "mistral",
    "mistral3",
    "mixtral",
    "qwen",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_moe",
    "qwen3_next",
}
BASE_REQUIRED_FEATURES = (
    "generation",
    "sampling_logits",
    "stream_lifecycle",
    "model_load_progress",
    "memory_guard",
    "observability",
    "greedy_constrained_decode",
    "stop_sequence",
    "continuous_batch_prompt_cache",
    "persistent_prompt_cache",
    "json_schema_constraints",
    "json_constraints",
    "finite_choice_constraints",
    "token_grammar_constraints",
    "regex_constraints",
)
ATTENTION_REQUIRED_FEATURES = ("runtime_kv_cache",)
NATIVE_TOOL_REQUIRED_FEATURES = (
    "tool_call_rendering",
    "tool_stream_translation",
    "tool_schema_normalization",
    "native_tool_constraints",
    "native_tool_stream_translation",
)
SESSION_REQUIRED_FEATURES = (
    "session_style_request",
    "rendered_text_streaming",
)
STRESS_REQUIRED_FEATURES = ("stress_generation",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a real-model validation log into benchmark summary JSON."
    )
    parser.add_argument("--log", required=True, type=Path, help="Validation log path.")
    parser.add_argument("--summary", required=True, type=Path, help="Summary output path.")
    parser.add_argument("--result", required=True, choices=("passed", "failed"))
    parser.add_argument("--started-utc", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--host-memory-gb", required=True)
    parser.add_argument("--selected-count", required=True)
    parser.add_argument("--swift-test-invocation-count", required=True)
    return parser.parse_args()


def parse_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_records: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    current_test: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\n")
            append_benchmark_record(line, line_number, benchmark_records, parse_errors)

            if line.startswith("-> "):
                if current_test is not None:
                    tests.append(current_test)
                current_test = {
                    "label": line.removeprefix("-> "),
                    "status": "unknown",
                    "duration_seconds": None,
                }
                continue

            if line.startswith("TEST_META_JSON "):
                apply_test_metadata(line, line_number, current_test, parse_errors)
                continue

            if current_test is None:
                continue

            status = parse_status(line)
            if status is not None:
                current_test.update(status)

            duration_seconds = parse_duration(line, line_number, parse_errors)
            if duration_seconds is not None:
                current_test["duration_seconds"] = duration_seconds

    if current_test is not None:
        tests.append(current_test)

    return benchmark_records, tests, parse_errors


def append_benchmark_record(
    line: str,
    line_number: int,
    records: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
) -> None:
    if line.startswith("BENCH_JSON "):
        kind = "BENCH_JSON"
        payload = line.removeprefix("BENCH_JSON ")
    elif line.startswith("STRESS_JSON "):
        kind = "STRESS_JSON"
        payload = line.removeprefix("STRESS_JSON ")
    else:
        return

    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        parse_errors.append(parse_error(line_number, kind, str(error)))
        return

    if isinstance(value, dict):
        records.append(value)


def apply_test_metadata(
    line: str,
    line_number: int,
    current_test: dict[str, Any] | None,
    parse_errors: list[dict[str, Any]],
) -> None:
    if current_test is None:
        parse_errors.append(parse_error(line_number, "TEST_META_JSON", "metadata without active test"))
        return

    payload = line.removeprefix("TEST_META_JSON ")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        parse_errors.append(parse_error(line_number, "TEST_META_JSON", str(error)))
        return

    if not isinstance(value, dict):
        parse_errors.append(parse_error(line_number, "TEST_META_JSON", "metadata is not an object"))
        return

    for key in ("feature_key", "model_id", "architecture"):
        current_test[key] = optional_string(value.get(key))
    current_test["feature_keys"] = feature_keys(value)
    current_test["tags"] = string_list(value.get("tags"))


def parse_status(line: str) -> dict[str, str] | None:
    stripped = line.strip()
    if stripped == "passed":
        return {"status": "passed"}
    if stripped.startswith("skipped: "):
        return {"status": "skipped", "reason": stripped.removeprefix("skipped: ")}
    if stripped.startswith("failed with exit status "):
        return {
            "status": "failed",
            "exit_status": stripped.removeprefix("failed with exit status "),
        }
    return None


def parse_duration(
    line: str,
    line_number: int,
    parse_errors: list[dict[str, Any]],
) -> float | None:
    stripped = line.strip()
    if not stripped.startswith("duration_seconds: "):
        return None
    value = stripped.removeprefix("duration_seconds: ")
    try:
        return float(value)
    except ValueError:
        parse_errors.append(
            parse_error(
                line_number,
                "duration_seconds",
                f"Invalid duration_seconds value: {value}",
            )
        )
        return None


def summarize_coverage(
    tests: list[dict[str, Any]],
    selected_count: int | None,
) -> dict[str, Any]:
    selected_models = selected_models_from_tests(tests)
    rows = []

    for model_id, model in sorted(selected_models.items()):
        for feature_key in required_features(model):
            row = coverage_row(model_id, model, feature_key, tests)
            rows.append(row)
    count_status = selected_model_count_status(selected_count, selected_models)
    if count_status is not None:
        rows.append(feature_count_mismatch_row(count_status))

    failed_rows = [row for row in rows if row["status"] != "passed"]
    return {
        "schema_version": 1,
        "status": coverage_summary_status(selected_models, count_status),
        "passed": not failed_rows,
        "selected_model_metadata_count": len(selected_models),
        "rows": rows,
        "failed_count": len(failed_rows),
    }


def summarize_benchmark_coverage(
    tests: list[dict[str, Any]],
    benchmark_records: list[dict[str, Any]],
    selected_count: int | None,
) -> dict[str, Any]:
    selected_models = selected_models_from_tests(tests)
    rows = [
        benchmark_coverage_row(
            model_id=model_id,
            architecture=optional_string(model.get("architecture")),
            benchmark_records=benchmark_records,
        )
        for model_id, model in sorted(selected_models.items())
    ]
    count_status = selected_model_count_status(selected_count, selected_models)
    if count_status is not None:
        rows.append(benchmark_count_mismatch_row(count_status))

    failed_rows = [row for row in rows if row["status"] != "passed"]
    return {
        "schema_version": 1,
        "status": coverage_summary_status(selected_models, count_status),
        "passed": not failed_rows,
        "selected_model_metadata_count": len(selected_models),
        "rows": rows,
        "failed_count": len(failed_rows),
    }


def benchmark_coverage_row(
    model_id: str | None,
    architecture: str | None,
    benchmark_records: list[dict[str, Any]],
    status: str | None = None,
) -> dict[str, Any]:
    records = [
        record
        for record in benchmark_records
        if record.get("kind") == "bench"
        and record.get("model") == model_id
        and (architecture is None or record.get("architecture") == architecture)
    ]
    return {
        "model_id": model_id,
        "architecture": architecture,
        "status": status or ("passed" if records else "missing"),
        "benchmark_count": len(records),
    }


def selected_model_count_status(
    selected_count: int | None,
    selected_models: dict[str, dict[str, Any]],
) -> dict[str, int | str] | None:
    if selected_count is None:
        return None
    observed_count = len(selected_models)
    if selected_count == observed_count:
        return None
    return {
        "status": "selected_model_count_mismatch",
        "selected_model_count": selected_count,
        "selected_model_metadata_count": observed_count,
    }


def feature_count_mismatch_row(count_status: dict[str, int | str]) -> dict[str, Any]:
    return {
        "model_id": None,
        "architecture": None,
        "feature_key": "generation",
        "status": count_status["status"],
        "label": None,
        "test_status": None,
        "duration_seconds": None,
        "selected_model_count": count_status["selected_model_count"],
        "selected_model_metadata_count": count_status["selected_model_metadata_count"],
    }


def benchmark_count_mismatch_row(count_status: dict[str, int | str]) -> dict[str, Any]:
    return {
        "model_id": None,
        "architecture": None,
        "status": count_status["status"],
        "benchmark_count": 0,
        "selected_model_count": count_status["selected_model_count"],
        "selected_model_metadata_count": count_status["selected_model_metadata_count"],
    }


def coverage_summary_status(
    selected_models: dict[str, dict[str, Any]],
    count_status: dict[str, int | str] | None,
) -> str:
    if count_status is not None:
        return str(count_status["status"])
    return "available" if selected_models else "not_available"


def selected_models_from_tests(tests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for test in tests:
        if test.get("feature_key") != "generation":
            continue
        model_id = test.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        models[model_id] = {
            "architecture": optional_string(test.get("architecture")),
            "tags": string_list(test.get("tags")),
        }
    return models


def required_features(model: dict[str, Any]) -> list[str]:
    features = list(BASE_REQUIRED_FEATURES)
    architecture = model.get("architecture")
    if architecture not in ATTENTIONLESS_ARCHITECTURES:
        features.extend(ATTENTION_REQUIRED_FEATURES)
    if architecture in NATIVE_TOOL_ARCHITECTURES:
        features.extend(NATIVE_TOOL_REQUIRED_FEATURES)

    tags = set(string_list(model.get("tags")))
    if "native-template-only" not in tags:
        features.extend(SESSION_REQUIRED_FEATURES)
    if "stress" in tags:
        features.extend(STRESS_REQUIRED_FEATURES)
    return features


def coverage_row(
    model_id: str,
    model: dict[str, Any],
    feature_key: str,
    tests: list[dict[str, Any]],
) -> dict[str, Any]:
    matches = [
        test
        for test in tests
        if test.get("model_id") == model_id and feature_matches(test, feature_key)
    ]
    passed = next((test for test in matches if test.get("status") == "passed"), None)
    selected = passed or (matches[0] if matches else None)
    return {
        "model_id": model_id,
        "architecture": model.get("architecture"),
        "feature_key": feature_key,
        "status": coverage_status(matches, passed),
        "label": selected.get("label") if selected else None,
        "test_status": selected.get("status") if selected else None,
        "duration_seconds": selected.get("duration_seconds") if selected else None,
    }


def coverage_status(matches: list[dict[str, Any]], passed: dict[str, Any] | None) -> str:
    if passed is not None:
        return "passed"
    if not matches:
        return "missing"
    return "not_passed"


def feature_matches(test: dict[str, Any], feature_key: str) -> bool:
    if test.get("feature_key") == feature_key:
        return True
    return feature_key in string_list(test.get("feature_keys"))


def status_counts(tests: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "passed": sum(1 for test in tests if test["status"] == "passed"),
        "skipped": sum(1 for test in tests if test["status"] == "skipped"),
        "failed": sum(1 for test in tests if test["status"] == "failed"),
        "unknown": sum(1 for test in tests if test["status"] == "unknown"),
    }


def parse_error(line: int, kind: str, message: str) -> dict[str, Any]:
    return {"line": line, "kind": kind, "message": message}


def optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    if isinstance(value, str) and value:
        return [item for item in value.split(",") if item]
    return []


def feature_keys(value: dict[str, Any]) -> list[str]:
    keys = string_list(value.get("feature_keys"))
    if keys:
        return keys
    feature_key = optional_string(value.get("feature_key"))
    return [feature_key] if feature_key else []


def integer(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def main() -> int:
    args = parse_args()
    benchmark_records, tests, parse_errors = parse_log(args.log)
    selected_count = integer(args.selected_count)
    summary = {
        "schema_version": 1,
        "result": args.result,
        "started_utc": args.started_utc,
        "ended_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "catalog": args.catalog,
        "models": args.models,
        "scope": args.scope,
        "host_ram_gib": integer(args.host_memory_gb),
        "selected_model_count": selected_count,
        "swift_test_invocation_count": integer(args.swift_test_invocation_count),
        "benchmark_log": str(args.log),
        "benchmark_records": benchmark_records,
        "benchmark_record_count": len(benchmark_records),
        "tests": tests,
        "test_status_counts": status_counts(tests),
        "feature_coverage": summarize_coverage(tests, selected_count),
        "benchmark_coverage": summarize_benchmark_coverage(
            tests,
            benchmark_records,
            selected_count,
        ),
        "benchmark_parse_errors": parse_errors,
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
