import pytest

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
