"""Audit an authorized, de-identified classifier holdout before scoring it."""
import argparse
import hashlib
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from app.core.taxonomy import LABEL2ID


def load_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def audit(holdout, reference, threshold=0.9):
    reference_groups = {row["group_id"] for row in reference}
    reference_texts = {normalize(row["text"]) for row in reference}
    seen = set()
    for index, row in enumerate(holdout, 1):
        text = normalize(row["text"])
        labels = row["labels"]
        sample_id, group_id = row["id"], row["group_id"]
        if not sample_id or sample_id in seen or not text or not group_id or not row["bucket"]:
            raise ValueError(f"Invalid or duplicate holdout sample at row {index}")
        if not isinstance(labels, list) or not labels or not set(labels) <= LABEL2ID.keys():
            raise ValueError(f"Invalid multilabel annotation at row {index}")
        if group_id in reference_groups or text in reference_texts:
            raise ValueError(f"Holdout overlaps reference data at row {index}")
        for candidate in reference_texts:
            if min(len(candidate), len(text)) * 2 < threshold * (len(candidate) + len(text)):
                continue
            if SequenceMatcher(None, text, candidate, autojunk=False).ratio() >= threshold:
                raise ValueError(f"Review near-duplicate reference sample for holdout row {index}")
        seen.add(sample_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path, help="Include all augmented examples")
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    required = ("version", "source", "approval_ref", "redaction_rules", "labeling_rules", "split_rules")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if any(not isinstance(manifest.get(key), str) or not manifest[key].strip() for key in required):
        parser.error("Manifest requires " + ", ".join(required))
    samples = load_jsonl(args.holdout)
    reference = load_jsonl(args.train) + load_jsonl(args.validation)
    if any(not row.get("group_id") for row in reference):
        parser.error("Every reference row must carry its conversation or source group ID")
    audit(samples, reference)
    print(f"Audited {len(samples)} holdout samples against {len(reference)} reference rows")
    if args.audit_only:
        return

    from scripts.ch10.evaluate import main as evaluate

    metadata = {
        "version": manifest["version"],
        "sha256": hashlib.sha256(args.holdout.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "near_duplicate_threshold": 0.9,
    }
    if args.report_dir.resolve() == Path("data/ch10/reports").resolve():
        parser.error("Use a separate directory for real-data reports")
    evaluate(samples=samples, reports=args.report_dir, dataset_metadata=metadata)


if __name__ == "__main__":
    main()
