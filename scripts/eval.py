"""Evaluation harness CLI.

Modes
-----
recoverability
    Runs OCR only and asks, for every field whose ground truth we know: does the
    correct answer even appear in the OCR text? This is the hard ceiling for any
    downstream extractor - a value that was never transcribed cannot be
    extracted by a model of any size. Requires no running services.

Usage
-----
    python scripts/eval.py recoverability
    python scripts/eval.py recoverability --variant raw6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.compare import is_grounded  # noqa: E402
from app.eval.groundtruth import CORPORA, GroundTruth, load_ground_truth  # noqa: E402
from app.eval.ocr_runner import (  # noqa: E402
    OcrResult, run_baseline, run_optimized, run_pooled, run_raw,
)
from app.eval.report import RecoverabilityRow, render_recoverability  # noqa: E402

VARIANTS = {
    "baseline": run_baseline,
    "raw3": lambda path: run_raw(path, psm=3),
    "raw6": lambda path: run_raw(path, psm=6),
    "raw11": lambda path: run_raw(path, psm=11),
    "optimized": run_optimized,
    "candidate": run_pooled,
    # Backward-compatible name for existing evaluation commands.
    "pooled": run_pooled,
}


def recoverability(variant: str, dump_dir: Path | None, corpus: str = "real") -> int:
    truths = load_ground_truth(corpus=corpus)
    runner = VARIANTS[variant]
    rows: list[RecoverabilityRow] = []

    for truth in truths:
        if not truth.image_path.exists():
            print(f"  ! missing image: {truth.image_path}")
            continue
        result: OcrResult = runner(truth.image_path)
        if dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)
            (dump_dir / f"{Path(truth.image).stem}.{variant}.txt").write_text(
                result.text, encoding="utf-8"
            )
        rows.append(_score_recoverability(truth, result))

    print(render_recoverability(rows, variant_label=variant))
    return 0


def _score_recoverability(truth: GroundTruth, result: OcrResult) -> RecoverabilityRow:
    recoverable = 0
    total = 0
    missing: list[str] = []

    for spec in truth.scored_fields():
        if not spec.transcribed:
            continue  # derived, not read off the page
        expected = truth.truth_for(spec.name)
        if expected is None:
            continue  # nothing to recover; scored in accuracy mode instead
        total += 1
        if is_grounded(spec, expected, result.text):
            recoverable += 1
        else:
            missing.append(spec.name)

    return RecoverabilityRow(
        image=truth.image,
        print_type=truth.print_type,
        recoverable=recoverable,
        total=total,
        missing_fields=missing,
    )


def sweep() -> int:
    """Run every OCR variant over the corpus and compare recoverability.

    Also reports 'best-of', the ceiling reachable when the pipeline is allowed to
    try several variants and keep the strongest result - the Q6 design.
    """
    truths = [t for t in load_ground_truth() if t.image_path.exists()]
    names = sorted(VARIANTS)
    per_variant: dict[str, list[RecoverabilityRow]] = {}
    best_recoverable: dict[str, set[str]] = {t.image: set() for t in truths}
    totals: dict[str, int] = {}

    for name in names:
        rows = []
        for truth in truths:
            result = VARIANTS[name](truth.image_path)
            row = _score_recoverability(truth, result)
            rows.append(row)
            totals[truth.image] = row.total
            found = {
                spec.name
                for spec in truth.scored_fields()
                if spec.transcribed and truth.truth_for(spec.name) is not None
                and spec.name not in row.missing_fields
            }
            best_recoverable[truth.image] |= found
        per_variant[name] = rows

    header = f"{'receipt':<16}" + "".join(f"{n:>12}" for n in names) + f"{'best-of':>12}"
    lines = ["", "VARIANT SWEEP  --  recoverable transcribed fields", "", header,
             "-" * len(header)]
    for truth in truths:
        cells = ""
        for name in names:
            row = next(r for r in per_variant[name] if r.image == truth.image)
            cells += f"{row.recoverable:>6}/{row.total:<5}" if row.total else f"{'-':>12}"
        best = len(best_recoverable[truth.image])
        total = totals[truth.image]
        cells += f"{best:>6}/{total:<5}" if total else f"{'-':>12}"
        lines.append(f"{truth.image:<16}{cells}")

    lines.append("-" * len(header))
    grand_total = sum(totals.values())
    summary = f"{'TOTAL':<16}"
    for name in names:
        recovered = sum(r.recoverable for r in per_variant[name])
        summary += f"{recovered:>6}/{grand_total:<5}"
    summary += f"{sum(len(v) for v in best_recoverable.values()):>6}/{grand_total:<5}"
    lines.append(summary)

    percent = f"{'':<16}"
    for name in names:
        recovered = sum(r.recoverable for r in per_variant[name])
        percent += f"{recovered / grand_total:>11.1%} "
    best_total = sum(len(v) for v in best_recoverable.values())
    percent += f"{best_total / grand_total:>11.1%} "
    lines.append(percent)
    print("\n".join(lines))
    return 0


def accuracy(
    verbose: bool,
    corpus: str = "real",
    limit: int | None = None,
    names: list[str] | None = None,
) -> int:
    """Score deterministic extraction against ground truth. No LLM required."""
    import asyncio

    from PIL import Image

    from app.core.confidence import SERMS_REVIEW_THRESHOLD, compute_confidence
    from app.core.extraction import extract, fast_path_sufficient
    from app.core.ocr_engine import read_pooled
    from app.core.verification import verify
    from app.eval.compare import Outcome, classify, is_grounded
    from app.eval.groundtruth import FIELD_SPECS, Gate
    from app.eval.items_score import ItemTally, render_items, score_items
    from app.eval.report import (
        empty_tallies, render_accuracy, render_fabrication,
    )

    truths = [t for t in load_ground_truth(corpus=corpus) if t.image_path.exists()]
    if names:
        wanted = {Path(name).name for name in names}
        truths = [t for t in truths if Path(t.image).name in wanted]
    if limit:
        truths = truths[:limit]
    if not truths:
        print(f"no receipts matched the filter (corpus={corpus})", file=sys.stderr)
        return 1
    print(f"corpus: {corpus}  ({len(truths)} receipts)")
    tallies = empty_tallies(spec.name for spec in FIELD_SPECS)
    populated = ungrounded_total = 0
    abstention_rows: list[tuple[str, bool, bool]] = []
    by_type: dict[str, list[tuple[str, Outcome]]] = {}
    confidence_rows: list[tuple[str, bool, float, bool, list[str]]] = []
    item_tally = ItemTally()
    by_degradation: dict[str, list[tuple[str, Outcome]]] = {}
    by_template: dict[str, list[tuple[str, Outcome]]] = {}

    for truth in truths:
        with Image.open(truth.image_path) as img:
            bundle = asyncio.run(read_pooled(img, lang="eng"))
        result = extract(bundle)
        # Mirror the production early-exit: escalate to the full pool when the
        # single-pass result does not reconcile.
        if bundle.early_exit and not fast_path_sufficient(result):
            with Image.open(truth.image_path) as img:
                bundle = asyncio.run(read_pooled(img, lang="eng", fast_path=False))
            result = extract(bundle)

        verification = verify(
            result.as_dict(),
            bundle.combined_text,
            reconciled=result.reconciled,
            locale_resolved=bool(result.locale and result.locale.resolved),
            mean_word_confidence=bundle.confidence,
            tax_id_ambiguous=_tax_id_ambiguous(result),
            derived_fields=set(result.money.derived) if result.money else set(),
        )
        breakdown = compute_confidence(
            verification,
            anchor_score=bundle.primary.score,
            mean_word_confidence=bundle.confidence,
            reconciled=result.reconciled,
            locale_certainty=result.locale.certainty if result.locale else 0.0,
        )
        produced = verification.fields
        country = produced.get("country")
        confidence_rows.append(
            (truth.image, truth.expect_manual_review, breakdown.score,
             breakdown.flagged_by_consumer, verification.reasons, None)
        )
        row_index = len(confidence_rows) - 1
        receipt_fields_correct = True

        if verbose:
            print(f"\n--- {truth.image} "
                  f"[{bundle.primary.variant}/psm{bundle.primary.psm}] ---")
            print(f"    locale: {produced.get('country')} / {produced.get('currency')}"
                  f"   reconciled={result.reconciled}"
                  f"   score={breakdown.score:.3f}")
            if verification.rejected:
                print(f"    REJECTED: {verification.rejected}")
            if verification.reasons:
                print(f"    reasons: {', '.join(verification.reasons)}")
            for key, note in sorted(result.evidence.items()):
                print(f"    {key}: {note}")

        for spec in FIELD_SPECS:
            if spec.name in truth.unknown or spec.name not in truth.expected:
                tallies[spec.name].outcomes.append(Outcome.SKIPPED)
                continue
            outcome = classify(
                spec,
                truth.truth_for(spec.name),
                produced.get(spec.name),
                country=country,
                trap_values=truth.not_expected_values.get(spec.name),
                accepted_alternatives=truth.also_acceptable.get(spec.name),
            )
            tallies[spec.name].outcomes.append(outcome)
            if spec.gate is not Gate.UNTRACKED and not outcome.is_pass:
                receipt_fields_correct = False
            if outcome.is_populated:
                populated += 1
                # Arithmetically derived values are exempt, matching the rule the
                # verification gate applies. Counting them as fabrications would
                # measure the harness rather than the pipeline.
                derived = set(result.money.derived) if result.money else set()
                if spec.name in derived:
                    continue
                if not is_grounded(spec, produced.get(spec.name), bundle.combined_text):
                    tallies[spec.name].ungrounded += 1
                    ungrounded_total += 1

        confidence_rows[row_index] = (
            *confidence_rows[row_index][:5], receipt_fields_correct,
        )

        if "items" not in truth.unknown:
            score_items(
                item_tally,
                image=truth.image,
                expected=truth.expected_items,
                produced=result.item_scan.payload() if result.item_scan else [],
                reconciled=bool(result.item_scan and result.item_scan.reconciled),
            )

        # Abstention is expected when the figures are unreadable. A receipt whose
        # arithmetic is deliberately inconsistent has a perfectly legible total,
        # so the correct behaviour there is to flag it, not to withhold it.
        if not truth.expect_reconciliation_failure:
            abstention_rows.append(
                (truth.image, truth.expect_manual_review,
                 produced.get("total_amount") is None)
            )
        by_type.setdefault(truth.print_type, []).extend(
            (spec.name, tallies[spec.name].outcomes[-1]) for spec in FIELD_SPECS
        )
        by_degradation.setdefault(truth.degradation, []).extend(
            (spec.name, tallies[spec.name].outcomes[-1]) for spec in FIELD_SPECS
        )
        by_template.setdefault(truth.template, []).extend(
            (spec.name, tallies[spec.name].outcomes[-1]) for spec in FIELD_SPECS
        )

    print(render_accuracy(tallies, receipt_count=len(truths)))
    print(_render_by_type(by_type))
    if len(by_degradation) > 1:
        print(_render_by_type(by_degradation, title="BY DEGRADATION TIER"))
    if len(by_template) > 1:
        print(_render_by_type(by_template, title="BY TEMPLATE"))
    print(render_items(item_tally))
    print(render_fabrication(populated, ungrounded_total))
    print(_render_abstention(abstention_rows))
    print(_render_routing(confidence_rows, SERMS_REVIEW_THRESHOLD))
    return 0


def _tax_id_ambiguous(result) -> bool:
    """Several tax ids and no way to tell which belongs to the vendor.

    Not ambiguous when one carries an explicit vendor marker such as
    'VAT Reg. TIN' - that is a positive identification, however many other numbers
    appear on the page.
    """
    if any(candidate.role == "vendor" for candidate in result.tax_id_candidates):
        return False
    unresolved = [c for c in result.tax_id_candidates if c.role == "unknown"]
    return len(unresolved) > 1


def _render_routing(rows: list, threshold: float) -> str:
    """Does the confidence score route each receipt safely?

    The two ways of getting this wrong are not equivalent, and an earlier version
    of this report treated them as if they were:

    * FALSE ACCEPT - scored above the threshold while a gated field is wrong. The
      receipt reaches the financial system unreviewed and incorrect. This is the
      failure that matters, and the target is zero.
    * CAUTIOUS FLAG - scored below the threshold although the fields were right.
      Costs a reviewer's time and nothing else.

    Reporting a single "routed correctly" figure hid the distinction, and counted a
    correctly-flagged misread under severe degradation as a mistake.
    """
    false_accepts: list[tuple] = []
    cautious: list[tuple] = []
    safe_accepts = correct_flags = 0

    for row in rows:
        image, expect_review, score, flagged, reasons = row[:5]
        fields_correct = row[5] if len(row) > 5 else None

        if flagged:
            if fields_correct is False or expect_review:
                correct_flags += 1
            else:
                cautious.append((image, score, reasons))
        else:
            if fields_correct is False or expect_review:
                false_accepts.append((image, score, reasons))
            else:
                safe_accepts += 1

    total = max(len(rows), 1)
    lines = [
        "",
        f"CONSUMER ROUTING  --  SERMS flags score < {threshold:.2f} for manual review",
        "",
        f"  accepted and correct      {safe_accepts:>4}  ({safe_accepts / total:.1%})",
        f"  flagged, needed review    {correct_flags:>4}  ({correct_flags / total:.1%})",
        f"  flagged but was fine      {len(cautious):>4}  "
        f"({len(cautious) / total:.1%})  cautious - costs review time only",
        f"  ACCEPTED BUT WRONG        {len(false_accepts):>4}  "
        f"({len(false_accepts) / total:.1%})  target 0",
    ]

    safe = len(false_accepts) == 0
    lines.append(f"\n  unreviewed incorrect data reaching SERMS: "
                 f"{'NONE [PASS]' if safe else 'PRESENT [FAIL]'}")

    if false_accepts:
        lines.append("\n  false accepts:")
        for image, score, reasons in false_accepts[:15]:
            lines.append(f"    {image:<28}{score:>7.3f}  {','.join(reasons[:3]) or '-'}")
    if cautious:
        lines.append("\n  cautious flags (first 10):")
        for image, score, reasons in cautious[:10]:
            lines.append(f"    {image:<28}{score:>7.3f}  {','.join(reasons[:3]) or '-'}")
    return "\n".join(lines)


def _render_by_type(by_type: dict[str, list], title: str = "BY DOCUMENT TYPE") -> str:
    """Accuracy split by a grouping key.

    Reported per group rather than blended. The gate applies to machine-printed
    receipts, and knowing which degradation tier or template is failing is more
    actionable than a single figure that averages them together.
    """
    from app.eval.groundtruth import FIELD_BY_NAME, Gate

    lines = ["", f"{title}  --  gated fields only (untracked excluded)", ""]
    for print_type, entries in sorted(by_type.items()):
        scored = [
            outcome
            for name, outcome in entries
            if outcome.name != "SKIPPED" and FIELD_BY_NAME[name].gate is not Gate.UNTRACKED
        ]
        if not scored:
            continue
        passed = len([o for o in scored if o.is_pass])
        lines.append(
            f"  {print_type:<20}{passed:>3}/{len(scored):<4}{passed / len(scored):>8.1%}"
        )
    return "\n".join(lines)


def _render_abstention(rows: list[tuple[str, bool, bool]]) -> str:
    lines = ["", "ABSTENTION  --  handwritten receipts must not produce a total", ""]
    correct = 0
    for image, expect_review, abstained in rows:
        if not expect_review:
            verdict = "n/a (printed)"
            ok = True
        else:
            ok = abstained
            verdict = "abstained" if abstained else "PRODUCED A TOTAL"
        correct += 1 if ok else 0
        lines.append(f"  {image:<16} {verdict}")
    lines.append(f"  -> {correct}/{len(rows)} behaved correctly")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR pipeline evaluation harness")
    sub = parser.add_subparsers(dest="mode", required=True)

    recover = sub.add_parser("recoverability", help="OCR-only ceiling measurement")
    recover.add_argument("--variant", choices=sorted(VARIANTS), default="baseline")
    recover.add_argument("--corpus", choices=sorted(CORPORA), default="real")
    recover.add_argument("--dump-dir", type=Path, default=None,
                         help="write raw OCR text per receipt for inspection")

    sub.add_parser("sweep", help="compare all OCR variants side by side")

    acc = sub.add_parser("accuracy", help="score deterministic extraction")
    acc.add_argument("--verbose", action="store_true", help="print evidence per receipt")
    acc.add_argument("--corpus", choices=sorted(CORPORA), default="real")
    acc.add_argument("--limit", type=int, default=None,
                     help="score only the first N receipts")
    acc.add_argument("--names", nargs="+", default=None,
                     help="score only receipts whose filename matches (repeatable)")

    args = parser.parse_args()
    if args.mode == "recoverability":
        return recoverability(args.variant, args.dump_dir, corpus=args.corpus)
    if args.mode == "sweep":
        return sweep()
    if args.mode == "accuracy":
        return accuracy(args.verbose, corpus=args.corpus, limit=args.limit,
                        names=args.names)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
