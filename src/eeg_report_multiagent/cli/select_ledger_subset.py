from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["row_index"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _score(row: Dict[str, str], score_column: str) -> float:
    try:
        return float(row.get(score_column) or 0.0)
    except ValueError:
        return 0.0


def _sample(items: List[Dict[str, str]], n: int, rng: random.Random) -> List[Dict[str, str]]:
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def select_celm_stratified(rows: List[Dict[str, str]], n: int, score_column: str, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    pending = [r for r in rows if r.get("our_B_status", "not_started") in {"", "not_started", "error"}]
    generated_nonzero = [
        r for r in pending
        if r.get("celm_generated_exists") == "true" and r.get("celm_nonzero_text_metric") == "true"
    ]
    zero_or_missing = [
        r for r in pending
        if not (r.get("celm_generated_exists") == "true" and r.get("celm_nonzero_text_metric") == "true")
    ]

    generated_nonzero = sorted(generated_nonzero, key=lambda r: _score(r, score_column))
    thirds = max(len(generated_nonzero) // 3, 1) if generated_nonzero else 1
    low = generated_nonzero[:thirds]
    mid = generated_nonzero[thirds:2 * thirds]
    high = generated_nonzero[2 * thirds:]

    allocation = {
        "low": n // 4,
        "mid": n // 4,
        "high": n // 4,
        "zero_or_missing": n - 3 * (n // 4),
    }
    selected: List[Dict[str, str]] = []
    selected.extend(_sample(low, allocation["low"], rng))
    selected.extend(_sample(mid, allocation["mid"], rng))
    selected.extend(_sample(high, allocation["high"], rng))
    selected.extend(_sample(zero_or_missing, allocation["zero_or_missing"], rng))

    selected_ids = {r["row_index"] for r in selected}
    remaining = [r for r in pending if r.get("row_index") not in selected_ids]
    if len(selected) < n:
        selected.extend(_sample(remaining, n - len(selected), rng))

    return sorted(selected[:n], key=lambda r: int(r["row_index"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select row indices from an experiment ledger")
    parser.add_argument("--ledger-csv", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--strategy", default="celm_stratified", choices=["celm_stratified", "first_pending"])
    parser.add_argument("--score-column", default="celm_rougeL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-row-indices", required=True)
    parser.add_argument("--output-selected-csv", required=True)
    args = parser.parse_args()

    rows = _read_rows(Path(args.ledger_csv))
    if args.strategy == "first_pending":
        selected = [r for r in rows if r.get("our_B_status", "not_started") in {"", "not_started", "error"}][: args.n]
    else:
        selected = select_celm_stratified(rows, n=args.n, score_column=args.score_column, seed=args.seed)

    out_indices = Path(args.output_row_indices)
    out_indices.parent.mkdir(parents=True, exist_ok=True)
    out_indices.write_text("\n".join(str(r["row_index"]) for r in selected) + ("\n" if selected else ""), encoding="utf-8")
    _write_csv(Path(args.output_selected_csv), selected)

    by_group = {
        "celm_generated_nonzero": sum(1 for r in selected if r.get("celm_generated_exists") == "true" and r.get("celm_nonzero_text_metric") == "true"),
        "celm_zero_or_missing": sum(1 for r in selected if not (r.get("celm_generated_exists") == "true" and r.get("celm_nonzero_text_metric") == "true")),
    }
    print(json.dumps({
        "selected": len(selected),
        "row_indices_path": str(out_indices),
        "selected_csv": args.output_selected_csv,
        "by_group": by_group,
        "first_rows": [r["row_index"] for r in selected[:10]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
