# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch10 评测:留出测试集上每类 P/R/F1 + 每类混淆矩阵 + 容错红线 + 判错样本导出。
运行:make ch10-eval。评测集扎在自家电商场景(dataset/test.jsonl),不引公开榜单。
产物一式两份:.md 给人读、.json 给验收页读(/acceptance/eval、/acceptance/errors),同一次评测同一份数。"""
import datetime as dt
import json
import pathlib

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.taxonomy import LABEL2ID, NUM_CLASSES, SEVERITY, TOPIC_NAMES
from scripts.ch10.inference_lib import apply_threshold

MODEL_DIR = pathlib.Path("data/ch10/model")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")
REPORTS = pathlib.Path("data/ch10/reports")
RED_LINES = {"严": 0.9, "中": 0.8}   # 档位 F1 红线;宽档不设线


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def predict_probabilities(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt").to(device)
            probs_all.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    return np.concatenate(probs_all)


def main(*, samples=None, reports=REPORTS, dataset_metadata=None, model_dir=MODEL_DIR) -> None:
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
    threshold = json.loads((model_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    if samples is None:
        samples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [s["text"] for s in samples]
    gold = np.zeros((len(samples), NUM_CLASSES), dtype=int)
    for i, s in enumerate(samples):
        for lb in s["labels"]:
            gold[i][LABEL2ID[lb]] = 1
    probabilities = predict_probabilities(model, tokenizer, texts, device)
    preds = apply_threshold(probabilities, threshold)

    p, r, f1, support = precision_recall_fscore_support(gold, preds, zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0)
    cms = multilabel_confusion_matrix(gold, preds)

    reports.mkdir(parents=True, exist_ok=True)
    lines = ["# ch10 分类器评测报告(留出测试集)", "",
             f"测试集 {len(samples)} 条;判定阈值 {threshold}(验证集扫描所得)。", "",
             f"**micro**: P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}  |  "
             f"**macro**: P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}", "",
             "## 每类指标(容错红线:严档 F1 ≥ 0.9,中档 ≥ 0.8,宽档不设线)", "",
             "| 类目 | 档位 | P | R | F1 | support | 红线 |",
             "|---|---|---|---|---|---|---|"]
    for i, name in enumerate(TOPIC_NAMES):
        sev = SEVERITY[name]
        # 严/中档各设硬红线;宽档不受约束,画 — 而不是 ✅,免得读者以为它也过了某条线
        line = RED_LINES.get(sev)
        if line is not None:
            flag = f"{line} {'🔴 不达标,先回头搞数据' if f1[i] < line else '✅'}"
        else:
            flag = "—"
        lines.append(f"| {name} | {sev} | {p[i]:.3f} | {r[i]:.3f} | {f1[i]:.3f} "
                     f"| {int(support[i])} | {flag} |")
    lines += ["", "## 每类混淆矩阵(TN FP / FN TP)", ""]
    for i, name in enumerate(TOPIC_NAMES):
        tn, fp = cms[i][0]
        fn, tp = cms[i][1]
        lines.append(f"- **{name}**: TN={tn} FP={fp} FN={fn} TP={tp}")
    err = ["# ch10 判错样本(人工复核:错在哪一类?标注本身有没有毛病?)", ""]
    errors: list[dict] = []
    for i, s in enumerate(samples):
        pred_labels = [TOPIC_NAMES[j] for j in range(NUM_CLASSES) if preds[i][j]]
        if set(pred_labels) != set(s["labels"]):
            sample_ref = s["id"] if dataset_metadata is not None else s["text"]
            err.append(f"- {sample_ref}\n  标准: {s['labels']}  预测: {pred_labels}")
            # 错误方向性:漏打只记放跑、多打只记冤枉、错位两头都记——错例条数与矩阵笔数
            # 对不上就是从这来的(验收页要把这笔账摊开,别让读者自己减)
            missed = [lb for lb in s["labels"] if lb not in pred_labels]
            extra = [lb for lb in pred_labels if lb not in s["labels"]]
            kind = "错位" if missed and extra else ("漏打" if missed else "多打")
            reference = {"id": sample_ref, "bucket": s["bucket"]} if dataset_metadata else {"text": sample_ref}
            errors.append({**reference, "gold": list(s["labels"]), "pred": pred_labels,
                           "missed": missed, "extra": extra, "kind": kind,
                           "matrix_entries": len(missed) + len(extra)})
    (reports / "error_samples.md").write_text("\n".join(err), encoding="utf-8")

    report = {
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "test_size": len(samples),
        "threshold": threshold,
        "red_lines": RED_LINES,
        "micro": {"p": round(float(micro_p), 4), "r": round(float(micro_r), 4),
                  "f1": round(float(micro_f1), 4)},
        "macro": {"p": round(float(macro_p), 4), "r": round(float(macro_r), 4),
                  "f1": round(float(macro_f1), 4)},
        "classes": [
            {"name": name, "severity": SEVERITY[name],
             "p": round(float(p[i]), 4), "r": round(float(r[i]), 4),
             "f1": round(float(f1[i]), 4), "support": int(support[i]),
             "red_line": RED_LINES.get(SEVERITY[name]),
             # 宽档不设线 → passed 为 None,页面画 —,不画 ✅,免得读者以为它也过了某条线
             "passed": (None if SEVERITY[name] not in RED_LINES
                        else bool(f1[i] >= RED_LINES[SEVERITY[name]])),
             "tn": int(cms[i][0][0]), "fp": int(cms[i][0][1]),
             "fn": int(cms[i][1][0]), "tp": int(cms[i][1][1])}
            for i, name in enumerate(TOPIC_NAMES)
        ],
        "errors": errors,
    }
    report["total_cells"] = len(samples) * NUM_CLASSES
    report["total_fp"] = sum(c["fp"] for c in report["classes"])
    report["total_fn"] = sum(c["fn"] for c in report["classes"])
    report["red_line_passed"] = all(c["passed"] is not False for c in report["classes"])
    if dataset_metadata:
        from collections import Counter

        from scripts.ch10.holdout import error_patterns

        report["dataset"] = dataset_metadata
        report["buckets"] = {}
        for bucket in sorted({s["bucket"] for s in samples}):
            indices = [i for i, s in enumerate(samples) if s["bucket"] == bucket]
            report["buckets"][bucket] = {
                "n": len(indices),
                **{f"{average}_f1": float(precision_recall_fscore_support(
                    gold[indices], preds[indices], average=average, zero_division=0,
                )[2]) for average in ("micro", "macro")},
            }
        baseline = json.loads((REPORTS / "eval_report.json").read_text(encoding="utf-8"))
        report["baseline_f1_delta"] = {average: report[average]["f1"] - baseline[average]["f1"]
                                       for average in ("micro", "macro")}
        report["missed_labels"] = Counter(label for error in errors for label in error["missed"])
        report["extra_labels"] = Counter(label for error in errors for label in error["extra"])
        report["error_patterns"] = patterns = error_patterns(samples, probabilities, threshold, errors)
        for cls in report["classes"]:
            cls["limited_support"] = cls["support"] < 10
        lines += ["", "## 真实留出集", "",
                  f"版本 {dataset_metadata['version']}。每类少于 10 条正例时，指标需结合样本数解释。",
                  f"对既有报告的 F1 差异 {report['baseline_f1_delta']}。"]
        lines += ["", "| 场景 | 样本数 | 错例数 |", "|---|---|---|"]
        for key, label in (("low_confidence", "所有标签均低于阈值"), ("multi_intent", "人工标注多个主题")):
            pattern = patterns[key]
            lines.append(f"| {label} | {pattern['n']} | {len(pattern['error_ids'])} |")
        lines += ["", "后续优先补采少于 10 条正例的类别和反复同时漏打、多打的标签组合。"]
        for pair in patterns["missed_extra_pairs"][:10]:
            lines.append(f"- 漏打「{pair['missed']}」、多打「{pair['extra']}」同时出现 {pair['count']} 次。")
    (reports / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    (reports / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"micro-F1 {micro_f1:.4f} / macro-F1 {macro_f1:.4f};"
          f"报告与判错样本已落 {reports}/(.md + .json)")


if __name__ == "__main__":
    main()
