# 复跑客服评测

## RAG 四策略

302 道题的全量对照见[评测报告](../data/ch04/reports/rag_tradeoffs_full.md)，包括按问题类型的质量、延迟和成本数据。

先按 [快速启动](quickstart.md) 写入测试知识库，再指定这次索引对应的知识快照：

```bash
uv run python -m scripts.eval_rag_tradeoffs --kb-version v1 --output /tmp/rag-tradeoffs.json
```

首次可加 `--limit-per-bucket 1` 检查连通性。正式对照用全量题目，并保留报告中的策略、模型、数据指纹与并发。所有策略共用同一问句改写结果和证据充分性判断，方便比较向量距离、BM25 和重排打分不同的结果。首输出与整轮时延包含检索和证据判定，排除离线裁判。生成成本按上游计量的 token 与生成调用数报告。嵌入和重排按子查询数估计逻辑调用量，重排为上界，估算不计重试。结合错误与超时数、各桶拒答率和 P95 选择策略。样本数不足的分桶需要继续收集数据才能下结论。

## 主题分类器的真实留出集

从维护者批准用于评估的对话中抽样。先移除订单号、联系人、地址及其他可识别内容，再由人工按当前主题词表复核多标签。按匿名会话或来源分组。样本各行用 JSON 对象记录 `id`、`group_id`、`bucket`、脱敏 `text` 和 `labels`。评测桶可用 `multi_intent`、`long_tail` 等，并在标注规则中记录定义。

对照文件要包含模型实际使用的训练样本、所有增强样本和验证样本。每行保留文本及匿名 `group_id`，增强样本沿用源会话分组。来源不明的训练或增强样本需要先恢复其数据血缘。独立的 manifest JSON 记录 `version`、`source`、`approval_ref`、`redaction_rules`、`labeling_rules` 和 `split_rules`。

```bash
uv run --group ml python -m scripts.ch10.holdout \
  --holdout /secure/real.jsonl --manifest /secure/manifest.json \
  --train /secure/train-with-groups.jsonl --validation /secure/val-with-groups.jsonl \
  --model-dir /secure/fixed-model --report-dir /secure/real-report
```

加 `--audit-only` 可在加载模型前检查数据重叠。每条对照样本的组 ID 必须是非空字符串，无首尾空格。省略 `--model-dir` 时使用 `data/ch10/model`。评测读取本地模型和已有的验证集阈值，并记录模型目录文件及输入数据的 SHA-256。

报告包含整体与分桶指标、每类支持数、低置信和多意图样本数及错例 ID。低置信指所有标签分数低于阈值，多意图以人工标注的主题数量为准。匿名错例可在受控目录复核。后续优先补采支持数较低的类别和反复同时漏打、多打的标签组合。
