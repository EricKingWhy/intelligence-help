"""Audit an authorized, de-identified classifier holdout before scoring it."""
import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from app.core.taxonomy import ID2LABEL, LABEL2ID


def load_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def _canonical_id(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value) if type(value) is int else ''


def audit(holdout, reference, threshold=0.9):
    reference_groups = set()
    reference_texts = set()
    for index, row in enumerate(reference, 1):
        if not isinstance(row, dict) or not (group_id := _canonical_id(row.get("group_id"))):
            raise ValueError(f"Invalid reference group_id at row {index}")
        text = row.get("text")
        if not isinstance(text, str) or not (normalized := normalize(text)):
            raise ValueError(f"Invalid reference text at row {index}")
        reference_groups.add(group_id)
        reference_texts.add(normalized)

    seen = set()
    for index, row in enumerate(holdout, 1):
        if (not isinstance(row, dict)
                or not isinstance(row.get("bucket"), str) or not row["bucket"].strip()
                or not isinstance(row.get("text"), str)):
            raise ValueError(f"Invalid holdout sample at row {index}")
        text = normalize(row["text"])
        labels = row.get("labels")
        sample_id = _canonical_id(row.get("id"))
        group_id = _canonical_id(row.get("group_id"))
        if not sample_id or sample_id in seen or not group_id or not text:
            raise ValueError(f"Invalid or duplicate holdout sample at row {index}")
        if (not isinstance(labels, list) or not labels
                or any(not isinstance(label, str) or label not in LABEL2ID for label in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError(f"Invalid multilabel annotation at row {index}")
        if group_id in reference_groups or text in reference_texts:
            raise ValueError(f"Holdout overlaps reference data at row {index}")
        for candidate in reference_texts:
            if min(len(candidate), len(text)) * 2 < threshold * (len(candidate) + len(text)):
                continue
            if SequenceMatcher(None, text, candidate, autojunk=False).ratio() >= threshold:
                raise ValueError(f"Review near-duplicate reference sample for holdout row {index}")
        seen.add(sample_id)


def validate_checkpoint(id_to_label, problem_type, threshold) -> None:
    if id_to_label != ID2LABEL:
        raise ValueError("Holdout checkpoint label order does not match the topic taxonomy")
    if problem_type != "multi_label_classification":
        raise ValueError("Holdout checkpoint must be a multi-label classifier")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("Holdout validation threshold must be a finite number in [0, 1]")


def error_patterns(samples, probabilities, threshold, errors):
    wrong_ids = {error["id"] for error in errors}
    groups = {"low_confidence": [], "multi_intent": []}
    for sample, scores in zip(samples, probabilities, strict=True):
        if max(scores) < threshold:
            groups["low_confidence"].append(sample["id"])
        if len(sample["labels"]) > 1:
            groups["multi_intent"].append(sample["id"])

    pairs = Counter((missed, extra) for error in errors
                    for missed in error["missed"] for extra in error["extra"])
    return {
        **{name: {"n": len(ids), "error_ids": [sample_id for sample_id in ids if sample_id in wrong_ids]}
           for name, ids in groups.items()},
        "missed_extra_pairs": [
            {"missed": missed, "extra": extra, "count": count}
            for (missed, extra), count in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path, help="Include all augmented examples")
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("data/ch10/model"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    required = ("version", "source", "approval_ref", "redaction_rules", "labeling_rules", "split_rules")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(key), str) or not manifest[key].strip() for key in required
    ):
        parser.error("Manifest requires " + ", ".join(required))
    samples = load_jsonl(args.holdout)
    reference = load_jsonl(args.train) + load_jsonl(args.validation)
    try:
        audit(samples, reference)
    except ValueError as error:
        parser.error(str(error))
    print(f"Audited {len(samples)} holdout samples against {len(reference)} reference rows")
    if args.audit_only:
        return

    if args.report_dir.resolve() == Path("data/ch10/reports").resolve():
        parser.error("Use a separate directory for real-data reports")
    if not (args.model_dir / "threshold.json").is_file():
        parser.error("Model directory must contain the validation threshold.json")

    from scripts.ch10.evaluate import main as evaluate

    metadata = {
        "version": manifest["version"],
        "sha256": _sha256(args.holdout),
        "train_sha256": _sha256(args.train),
        "validation_sha256": _sha256(args.validation),
        "manifest_sha256": _sha256(args.manifest),
        "model_files_sha256": {file.name: _sha256(file) for file in sorted(args.model_dir.iterdir())
                               if file.is_file()},
        "near_duplicate_threshold": 0.9,
    }
    evaluate(samples=samples, reports=args.report_dir, dataset_metadata=metadata, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
