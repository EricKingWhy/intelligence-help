# 从空库跑通客服演示

需要 Python 3.12（可由 uv 安装）、uv、Docker Compose、make、Bash 和 curl。以下命令在 Linux 或 macOS 的 Bash 中执行。从新克隆进入仓库后：

```bash
uv sync --locked
cp .env.example .env
```

在 `.env` 填写聊天模型和聊天、嵌入、重排的端点及密钥。以示例文件中的变量名为准。把 `CHAT_API_KEY`、`EMBED_API_KEY`、`RERANK_API_KEY` 分别配置为可用密钥。以下使用本地 Milvus，`MILVUS_URI` 指向 `http://localhost:19530`。

```bash
docker compose -f docker-compose.yml -f docker-compose.test.yml up -d --wait --wait-timeout 300 mysql milvus-standalone
make seed
make kb-build
make kb-vectorize
make mcp-up
uv run uvicorn app.main:app --port 8000
```

`docker compose ps` 应显示 MySQL 和 Milvus 为 healthy。这条路径使用 Milvus 的内嵌 etcd 与本地存储，首次启动使用独立的向量数据卷。宿主机 3306 冲突时同时调整 `MYSQL_HOST_PORT` 和 `DATABASE_URL` 的端口。向量化成功后确认终端显示的 Milvus 条数大于零。

打开 [聊天页](http://localhost:8000/)，用新会话分别发送以下消息。

| 输入 | 预期 |
|---|---|
| 订单 1001 的物流到哪了？ | 出现订单及物流查询标记，回答物流状态。 |
| 退货政策是什么？ | 回答七天无理由及适用条件，带可展开的对应政策引用。 |
| 火星车能参加以旧换新吗？ | 明确说明无法从现有知识确认，给出兜底。 |

本地运行 `make test-quick` 检查无外部服务的回归。数据库容器就绪后用 `make test-integration` 跑真实 MySQL 和 Milvus 测试，`TEST_DATABASE_URL` 务必指向独立的测试库。更多启动排查见 [部署说明](../DEPLOY.md)。

RAG 策略比较与分类器留出集用法见 [评测说明](evaluation.md)。

## 已有数据库升级

首次启动自动加载 `sql/`。升级已有数据库时先备份，并在应用进入维护状态时确认三项字段迁移未执行，再运行一次：

```bash
docker exec -it mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -p mewhelp -e 'source /docker-entrypoint-initdb.d/ch11-reliability.sql'
```

使用远程 MySQL 时，在自己的 MySQL 客户端运行 `source sql/ch11-reliability.sql`。

## 真实用户部署

用环境变量 `MEWHELP_AUTH_MODE=required` 启用身份校验，`MEWHELP_AUTH_FILE` 指向仓库外的凭证 JSON 文件。仅经 HTTPS 反向代理访问应用，把数据库和业务 MCP 留在私网。由运营人员给每个账户分配随机 32 字节以上的凭证，文件只记录凭证 UTF-8 字符串的 SHA-256 十六进制摘要。重启服务加载变更。配置文件格式如下：

```json
{
  "customer-a": {"role": "customer", "sha256": "<64 个十六进制字符>"},
  "operator-a": {"role": "operator", "sha256": "<64 个十六进制字符>"}
}
```

浏览器通过 HTTP Basic 的登录框输入账户和随机凭证。客户使用聊天页，运营人员可访问管理页。API 客户端提交工单或退款时传 `Idempotency-Key`，同一次提交的重试复用该值。示例订单和物流依然来自模拟业务后端，真实订单接入时应沿用服务端归属校验与同一可信身份。

默认的 `demo` 模式兼容现有本机演示与课程脚本，仅用于本地测试。
