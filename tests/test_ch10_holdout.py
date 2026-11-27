import pytest

from scripts.ch10 import holdout
from scripts.ch10.holdout import audit


def test_holdout_audit_rejects_cross_split_overlap():
    train = [{"group_id": "train-1", "text": "这个商品如何开增值税发票"}]
    sample = {"id": "sample-1", "group_id": "real-1", "bucket": "logistics",
              "text": "犹豫晚点提交订单时还能不能修改配送地址", "labels": ["订单修改"]}
    audit([sample], train)
    with pytest.raises(ValueError, match="overlaps"):
        audit([{**sample, "group_id": "train-1"}], train)
    with pytest.raises(ValueError, match="near-duplicate"):
        audit([{**sample, "text": "这个商品如何开电子增值税发票"}], train)


@pytest.mark.parametrize("group_id", ["   ", 123, None])
def test_holdout_rejects_unusable_reference_group_ids(group_id):
    train = [{"group_id": group_id, "text": "这个商品如何开增值税发票"}]
    sample = {"id": "sample-1", "group_id": "real-1", "bucket": "logistics",
              "text": "犹豫晚点提交订单时还能不能修改配送地址", "labels": ["订单修改"]}
    with pytest.raises(ValueError, match="reference.*group_id"):
        audit([sample], train)


def test_holdout_report_separates_low_confidence_and_multi_intent_errors():
    samples = [
        {"id": "a", "labels": ["订单修改"]},
        {"id": "b", "labels": ["物流", "订单修改"]},
        {"id": "c", "labels": ["订单修改"]},
        {"id": "d", "labels": ["订单修改"]},
    ]
    probabilities = [[0.3, 0.4, 0.2], [0.2, 0.7, 0.6], [0.4, 0.3, 0.1], [0.8, 0.6, 0.1]]
    errors = [
        {"id": "a", "missed": ["订单修改"], "extra": ["物流"]},
        {"id": "b", "missed": ["订单修改"], "extra": ["支付"]},
        {"id": "d", "missed": [], "extra": ["物流"]},
    ]

    assert holdout.error_patterns(samples, probabilities, 0.5, errors) == {
        "low_confidence": {"n": 2, "error_ids": ["a"]},
        "multi_intent": {"n": 1, "error_ids": ["b"]},
        "missed_extra_pairs": [
            {"missed": "订单修改", "extra": "支付", "count": 1},
            {"missed": "订单修改", "extra": "物流", "count": 1},
        ],
    }
