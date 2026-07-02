#!/usr/bin/env python3
"""Compare MLX real-model benchmark summary JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_METRICS = ("decode_tps", "total_tps", "prompt_tps", "e2e_tps")
DEFAULT_TEST_DURATION_MAX_RATIO = 1.50


@dataclass(frozen=True)
class BenchmarkKey:
    kind: str
    model: str
    architecture: str

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.model}:{self.architecture}"


@dataclass
class BenchmarkAggregate:
    count: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class CoverageKey:
    model: str
    architecture: str
    feature: str

    @property
    def label(self) -> str:
        return f"{self.model}:{self.architecture}:{self.feature}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BENCH_JSON/STRESS_JSON records from two real-model summary files."
    )
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline summary JSON.")
    parser.add_argument("--current", required=True, type=Path, help="Current summary JSON.")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        choices=DEFAULT_METRICS,
        help="Metric to compare. Repeat for multiple metrics. Defaults to all throughput metrics.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.90,
        help="Minimum current/baseline ratio for every selected metric. Defaults to 0.90.",
    )
    parser.add_argument(
        "--test-duration-max-ratio",
        type=float,
        default=DEFAULT_TEST_DURATION_MAX_RATIO,
        help=(
            "Maximum current/baseline ratio for passed real-model feature-test durations. "
            f"Defaults to {DEFAULT_TEST_DURATION_MAX_RATIO:.2f}. Use 0 to skip duration checks."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the comparison report as JSON instead of a text table.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except OSError as error:
        raise SystemExit(f"Cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"Cannot parse {path}: {error}") from error

    if not isinstance(value, dict):
        raise SystemExit(f"Summary is not a JSON object: {path}")
    return value


def aggregate_records(summary: dict[str, Any], path: Path) -> dict[BenchmarkKey, BenchmarkAggregate]:
    records = summary.get("benchmark_records")
    if not isinstance(records, list):
        raise SystemExit(f"Summary missing benchmark_records array: {path}")

    parse_errors = summary.get("benchmark_parse_errors", [])
    if parse_errors:
        raise SystemExit(f"Summary contains benchmark_parse_errors: {path}")

    grouped: dict[BenchmarkKey, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue
        key = BenchmarkKey(
            kind=str(record.get("kind", "unknown")),
            model=str(record.get("model", "unknown")),
            architecture=str(record.get("architecture", "unknown")),
        )
        grouped[key].append(record)

    if not grouped:
        raise SystemExit(f"Summary contains no benchmark records: {path}")

    return {
        key: BenchmarkAggregate(
            count=len(key_records),
            metrics=average_metrics(key_records),
        )
        for key, key_records in grouped.items()
    }


def aggregate_tests(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = summary.get("tests", [])
    if not isinstance(records, list):
        return {}

    tests: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        label = record.get("label")
        if not isinstance(label, str) or not label:
            continue
        tests[label] = {
            "status": str(record.get("status", "unknown")),
            "duration_seconds": numeric_value(record.get("duration_seconds")),
        }
    return tests


def aggregate_coverage(summary: dict[str, Any]) -> dict[CoverageKey, dict[str, Any]]:
    coverage = summary.get("feature_coverage")
    if not isinstance(coverage, dict):
        return {}
    rows = coverage.get("rows", [])
    if not isinstance(rows, list):
        return {}

    records: dict[CoverageKey, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = optional_string(row.get("model_id"))
        architecture = optional_string(row.get("architecture"))
        feature = optional_string(row.get("feature_key"))
        if model is None or feature is None:
            continue
        key = CoverageKey(
            model=model,
            architecture=architecture or "unknown",
            feature=feature,
        )
        records[key] = {
            "status": str(row.get("status", "unknown")),
            "label": optional_string(row.get("label")),
            "test_status": optional_string(row.get("test_status")),
            "duration_seconds": numeric_value(row.get("duration_seconds")),
        }
    return records


def average_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    averaged: dict[str, float] = {}
    for metric in DEFAULT_METRICS:
        values = [
            float(record[metric])
            for record in records
            if isinstance(record.get(metric), (int, float))
        ]
        if values:
            averaged[metric] = mean(values)
    return averaged


def numeric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def compare(
    baseline: dict[BenchmarkKey, BenchmarkAggregate],
    current: dict[BenchmarkKey, BenchmarkAggregate],
    baseline_tests: dict[str, dict[str, Any]],
    current_tests: dict[str, dict[str, Any]],
    baseline_coverage: dict[CoverageKey, dict[str, Any]],
    current_coverage: dict[CoverageKey, dict[str, Any]],
    metrics: list[str],
    min_ratio: float,
    test_duration_max_ratio: float,
) -> dict[str, Any]:
    rows = []
    failures = []

    for key in sorted(baseline, key=lambda item: item.label):
        baseline_record = baseline[key]
        current_record = current.get(key)
        if current_record is None:
            failure = {
                "key": key.label,
                "metric": None,
                "status": "missing_current",
            }
            rows.append(failure)
            failures.append(failure)
            continue

        for metric in metrics:
            baseline_value = baseline_record.metrics.get(metric)
            current_value = current_record.metrics.get(metric)
            row = {
                "key": key.label,
                "metric": metric,
                "baseline": baseline_value,
                "current": current_value,
                "baseline_count": baseline_record.count,
                "current_count": current_record.count,
                "ratio": ratio(current_value, baseline_value),
            }
            row["status"] = status_for(row["ratio"], min_ratio)
            rows.append(row)
            if row["status"] != "passed":
                failures.append(row)

    extra_current = [
        key.label for key in sorted(set(current).difference(baseline), key=lambda item: item.label)
    ]
    test_rows, test_failures = compare_tests(
        baseline_tests,
        current_tests,
        test_duration_max_ratio,
    )
    extra_current_tests = sorted(set(current_tests).difference(baseline_tests))
    failures.extend(test_failures)
    coverage_rows, coverage_failures = compare_coverage(
        baseline_coverage,
        current_coverage,
    )
    extra_current_coverage = [
        key.label
        for key in sorted(
            set(current_coverage).difference(baseline_coverage),
            key=lambda item: item.label,
        )
    ]
    failures.extend(coverage_failures)
    return {
        "schema_version": 1,
        "min_ratio": min_ratio,
        "test_duration_max_ratio": test_duration_max_ratio,
        "metrics": metrics,
        "rows": rows,
        "test_rows": test_rows,
        "coverage_rows": coverage_rows,
        "extra_current": extra_current,
        "extra_current_tests": extra_current_tests,
        "extra_current_coverage": extra_current_coverage,
        "failure_count": len(failures),
        "passed": not failures,
    }


def compare_tests(
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    max_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for label in sorted(baseline):
        baseline_record = baseline[label]
        if baseline_record["status"] != "passed":
            continue
        current_record = current.get(label)
        row = {
            "label": label,
            "baseline_status": baseline_record["status"],
            "current_status": current_record["status"] if current_record else None,
            "baseline_duration_seconds": baseline_record["duration_seconds"],
            "current_duration_seconds": (
                current_record["duration_seconds"] if current_record else None
            ),
            "duration_ratio": None,
        }

        if current_record is None:
            row["status"] = "missing_current_test"
        elif current_record["status"] != "passed":
            row["status"] = "test_status_regressed"
        else:
            row["duration_ratio"] = ratio(
                current_record["duration_seconds"],
                baseline_record["duration_seconds"],
            )
            row["status"] = duration_status(
                row["duration_ratio"],
                current_record["duration_seconds"],
                baseline_record["duration_seconds"],
                max_ratio,
            )

        rows.append(row)
        if row["status"] not in {"passed", "duration_unchecked"}:
            failures.append(row)
    return rows, failures


def compare_coverage(
    baseline: dict[CoverageKey, dict[str, Any]],
    current: dict[CoverageKey, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for key in sorted(baseline, key=lambda item: item.label):
        baseline_record = baseline[key]
        if baseline_record["status"] != "passed":
            continue
        current_record = current.get(key)
        row = {
            "key": key.label,
            "model": key.model,
            "architecture": key.architecture,
            "feature_key": key.feature,
            "baseline_status": baseline_record["status"],
            "current_status": current_record["status"] if current_record else None,
            "baseline_label": baseline_record["label"],
            "current_label": current_record["label"] if current_record else None,
        }

        if current_record is None:
            row["status"] = "missing_current_coverage"
        elif current_record["status"] != "passed":
            row["status"] = "coverage_regressed"
        else:
            row["status"] = "passed"

        rows.append(row)
        if row["status"] != "passed":
            failures.append(row)
    return rows, failures


def ratio(current_value: float | None, baseline_value: float | None) -> float | None:
    if current_value is None or baseline_value is None or baseline_value <= 0:
        return None
    return current_value / baseline_value


def status_for(value: float | None, min_ratio: float) -> str:
    if value is None:
        return "missing_metric"
    if value < min_ratio:
        return "regressed"
    return "passed"


def duration_status(
    ratio_value: float | None,
    current_value: float | None,
    baseline_value: float | None,
    max_ratio: float,
) -> str:
    if baseline_value is None:
        return "duration_unchecked"
    if current_value is None:
        return "missing_duration"
    if max_ratio <= 0 or baseline_value <= 0:
        return "duration_unchecked"
    if ratio_value is None:
        return "missing_duration"
    if ratio_value > max_ratio:
        return "duration_regressed"
    return "passed"


def print_text_report(report: dict[str, Any]) -> None:
    print(f"Benchmark comparison min_ratio={report['min_ratio']:.3f}")
    for row in report["rows"]:
        if row["status"] == "missing_current":
            print(f"FAIL {row['key']} missing in current summary")
            continue

        ratio_value = row["ratio"]
        ratio_text = "n/a" if ratio_value is None else f"{ratio_value:.3f}"
        baseline = format_number(row["baseline"])
        current = format_number(row["current"])
        prefix = "PASS" if row["status"] == "passed" else "FAIL"
        print(
            f"{prefix} {row['key']} {row['metric']} "
            f"baseline={baseline} current={current} ratio={ratio_text}"
        )

    if report["extra_current"]:
        print("Extra current records:")
        for key in report["extra_current"]:
            print(f"- {key}")

    if report["test_rows"]:
        print(
            "Feature-test duration comparison "
            f"max_ratio={report['test_duration_max_ratio']:.3f}"
        )
    for row in report["test_rows"]:
        ratio_value = row["duration_ratio"]
        ratio_text = "n/a" if ratio_value is None else f"{ratio_value:.3f}"
        baseline = format_number(row["baseline_duration_seconds"])
        current = format_number(row["current_duration_seconds"])
        prefix = "PASS" if row["status"] in {"passed", "duration_unchecked"} else "FAIL"
        print(
            f"{prefix} {row['label']} duration_seconds "
            f"baseline={baseline} current={current} ratio={ratio_text} status={row['status']}"
        )

    if report["extra_current_tests"]:
        print("Extra current tests:")
        for label in report["extra_current_tests"]:
            print(f"- {label}")

    if report["coverage_rows"]:
        print("Feature coverage comparison")
    for row in report["coverage_rows"]:
        prefix = "PASS" if row["status"] == "passed" else "FAIL"
        print(
            f"{prefix} {row['key']} "
            f"baseline={row['baseline_status']} current={row['current_status']} "
            f"status={row['status']}"
        )

    if report["extra_current_coverage"]:
        print("Extra current feature coverage:")
        for key in report["extra_current_coverage"]:
            print(f"- {key}")


def format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "n/a"


def main() -> int:
    args = parse_args()
    metrics = args.metrics or list(DEFAULT_METRICS)
    baseline_summary = load_summary(args.baseline)
    current_summary = load_summary(args.current)
    baseline = aggregate_records(baseline_summary, args.baseline)
    current = aggregate_records(current_summary, args.current)
    report = compare(
        baseline,
        current,
        aggregate_tests(baseline_summary),
        aggregate_tests(current_summary),
        aggregate_coverage(baseline_summary),
        aggregate_coverage(current_summary),
        metrics,
        args.min_ratio,
        args.test_duration_max_ratio,
    )

    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print_text_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
