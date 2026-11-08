"""Report rendering for the evaluation harness.

Deliberately prints a per-field matrix instead of a single blended figure. A
single "accuracy" number hides the difference between a wrong total_amount
(a wrong reimbursement) and a wrong expense_category (a nuisance).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from app.eval.compare import Outcome
from app.eval.groundtruth import FIELD_BY_NAME, Gate

GATE_THRESHOLDS = {Gate.CRITICAL: 0.95, Gate.STANDARD: 0.90}


@dataclass
class FieldTally:
    outcomes: list[Outcome] = field(default_factory=list)
    ungrounded: int = 0

    @property
    def scored(self) -> int:
        return len([o for o in self.outcomes if o is not Outcome.SKIPPED])

    @property
    def passed(self) -> int:
        return len([o for o in self.outcomes if o.is_pass])

    @property
    def accuracy(self) -> float | None:
        return (self.passed / self.scored) if self.scored else None

    def count(self, outcome: Outcome) -> int:
        return len([o for o in self.outcomes if o is outcome])


@dataclass
class RecoverabilityRow:
    image: str
    print_type: str
    recoverable: int
    total: int
    missing_fields: list[str]

    @property
    def ratio(self) -> float | None:
        return (self.recoverable / self.total) if self.total else None


def render_recoverability(rows: Iterable[RecoverabilityRow], variant_label: str) -> str:
    rows = list(rows)
    lines = [
        "",
        "RECOVERABILITY  --  is the correct answer even present in the OCR text?",
        f"variant: {variant_label}",
        "",
        f"{'receipt':<16}{'type':<20}{'recoverable':>12}   {'':<4}",
        "-" * 72,
    ]
    for row in rows:
        ratio = "n/a" if row.ratio is None else f"{row.ratio:6.1%}"
        lines.append(
            f"{row.image:<16}{row.print_type:<20}"
            f"{row.recoverable:>4}/{row.total:<3}{ratio:>10}"
        )

    printed = [r for r in rows if r.print_type == "machine_printed"]
    handwritten = [r for r in rows if r.print_type == "handwritten_form"]
    lines.append("-" * 72)
    lines.append(_subtotal("machine printed", printed))
    lines.append(_subtotal("handwritten", handwritten))
    lines.append(_subtotal("ALL", rows))

    lines.append("")
    lines.append("Fields whose true value is ABSENT from the OCR text (unreachable")
    lines.append("by any downstream model, regardless of size):")
    for row in rows:
        if row.missing_fields:
            lines.append(f"  {row.image:<16} {', '.join(sorted(row.missing_fields))}")
    return "\n".join(lines)


def _subtotal(label: str, rows: list[RecoverabilityRow]) -> str:
    recoverable = sum(r.recoverable for r in rows)
    total = sum(r.total for r in rows)
    ratio = f"{recoverable / total:6.1%}" if total else "   n/a"
    return f"{label:<36}{recoverable:>4}/{total:<3}{ratio:>10}"


def render_accuracy(tallies: dict[str, FieldTally], receipt_count: int) -> str:
    lines = [
        "",
        f"FIELD ACCURACY  --  {receipt_count} receipts",
        "",
        f"{'field':<22}{'gate':<10}{'n':>4}{'acc':>8}{'ok':>5}{'wrong':>7}"
        f"{'missed':>8}{'fp':>5}{'trap':>6}{'ungr':>6}  status",
        "-" * 96,
    ]
    for name, tally in tallies.items():
        spec = FIELD_BY_NAME[name]
        threshold = GATE_THRESHOLDS.get(spec.gate)
        accuracy = tally.accuracy
        if accuracy is None:
            status, accuracy_text = "no data", "    -"
        elif threshold is None:
            status, accuracy_text = "untracked", f"{accuracy:6.1%}"
        else:
            status = "PASS" if accuracy >= threshold else "FAIL"
            accuracy_text = f"{accuracy:6.1%}"
        lines.append(
            f"{name:<22}{spec.gate.value:<10}{tally.scored:>4}{accuracy_text:>8}"
            f"{tally.passed:>5}{tally.count(Outcome.WRONG):>7}"
            f"{tally.count(Outcome.MISSED):>8}"
            f"{tally.count(Outcome.FALSE_POSITIVE):>5}"
            f"{tally.count(Outcome.CONFUSED):>6}{tally.ungrounded:>6}  {status}"
        )
    return "\n".join(lines)


def render_fabrication(total_populated: int, ungrounded: int) -> str:
    rate = (ungrounded / total_populated) if total_populated else 0.0
    verdict = "PASS" if ungrounded == 0 else "FAIL"
    return (
        "\nFABRICATION  --  values emitted with no supporting text in the OCR output\n"
        f"  populated fields: {total_populated}\n"
        f"  ungrounded:       {ungrounded}\n"
        f"  rate:             {rate:.1%}   target 0.0%   [{verdict}]"
    )


def empty_tallies(field_names: Iterable[str]) -> dict[str, FieldTally]:
    tallies: dict[str, FieldTally] = defaultdict(FieldTally)
    for name in field_names:
        tallies[name] = FieldTally()
    return tallies
