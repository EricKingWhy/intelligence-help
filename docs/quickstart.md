# 从空库跑通客服演示

需要 Python 3.12（可由 uv 安装）、uv、Docker Compose、make、Bash 和 curl。以下命令在 Linux 或 macOS 的 Bash 中执行。从新克隆进入仓库后：

```bash
uv sync --locked
cp .env.example .env
```

在 `.env` 填写聊天模型和聊天、嵌入、重排的端点及密钥。以示例文件中的变量名为准。把 `CHAT_API_KEY`、`EMBED_API_KEY`、`RERANK_API_KEY` 分别配置为可用密钥。以下使用本地 Milvus，`MILVUS_URI` 指向 `http://localhost:19530`。

聊天端点需要支持 OpenAI 兼容的 Function Calling。如果日志出现流式响应缺少 `tool_calls`，可在启动应用前执行 `export MEWHELP_TOOL_STREAMING=off` 测试该端点的非流式工具调用。

```bash
docker compose -f docker-compose.yml -f docker-compose.test.yml up -d --wait --wait-timeout 300 mysql milvus-standalone
make seed
make kb-build
make kb-vectorize
make mcp-up
uv run uvicorn app.main:app --port 8000
```

`docker compose ps` 应显示 MySQL 和 Milvus 为 healthy。这条路径使用 Milvus 的内嵌 etcd 与本地存储，首次启动使用独立的向量数据卷。宿主机 3306 冲突时，可在 `.env` 设置 `MYSQL_HOST_PORT=3307`，并将 `DATABASE_URL` 的端口改为 3307。使用同一容器运行测试时，也要更新 `TEST_DATABASE_URL` 的端口。容器内部仍使用 3306。向量化成功后确认终端显示的 Milvus 条数大于零。

打开 [聊天页](http://localhost:8000/)，用新会话分别发送以下消息。

| 输入 | 预期 |
|---|---|
| 订单 1001 的物流到哪了？ | 出现订单及物流查询标记，回答物流状态。 |
| 退货政策是什么？ | 回答七天无理由及适用条件，带可展开的对应政策引用。 |
| 火星车能参加以旧换新吗？ | 明确说明无法从现有知识确认，给出兜底。 |

也可以运行 `uv run python -m scripts.smoke_customer`，通过现有聊天 SSE 接口检查三项结果。

本地运行 `make test-quick` 检查无外部服务的回归。数据库容器就绪后用 `make test-integration` 跑真实 MySQL 和 Milvus 测试，`TEST_DATABASE_URL` 务必指向独立的测试库。更多启动排查见 [部署说明](../DEPLOY.md)。

RAG 策略比较与分类器留出集用法见 [评测说明](evaluation.md)。

## 已有数据库升级

Docker 首次启动自动加载 `sql/`。已有数据库升级前，先备份并进入维护状态。迁移命令用 `DATABASE_URL` 指定的库，先运行只读预检，核对待执行版本和已有变更。

```bash
uv run python -m scripts.migrate_mysql
uv run python -m scripts.migrate_mysql --apply
```

`--apply` 按顺序执行 ch11、ch12，并在 `mewhelp_schema_migrations` 记录版本和 SQL 校验和。此前手工执行或由 Docker 初始化的版本，经结构核对后会补记版本记录。对已创建的空数据库，该命令先建 ch02–ch10 的表结构，演示数据仍按上文另行加载。

存量库预检会核对应用 ORM 使用的 ch02–ch10 基线列及两处历史枚举，并按版本检查 ch11、ch12 新增对象的定义和记录校验和。即使已有版本记录，基线缺列也会列出具体的 `表.列`，退出码为 2。补齐基线后可重新预检。

预检发现部分完成时会列出已存在和缺失的对象，退出码为 2。若中断发生在完整的 `ALTER TABLE` 之间，`--apply` 可继续执行剩余语句。若同一条语句的对象只出现一部分、结构与预期不同，或版本校验和不符，`--apply` 会在写入前停止。此时根据报告核对结构和备份，再运行预检。

## 政策向量更新与补跑

更新知识文件后执行 `make kb-repatch`，再用 `make kb-vectorize` 补齐 `pending`。若嵌入期间正文又被修改，该块会留待下一次向量化。发布时锁定当前知识块，Milvus 刷盘后才回标完成。

此变更无需迁移。升级前先结束旧版本的向量化作业和补丁脚本。发布期间 MySQL 连接失效时，应用会用新事务尝试将受影响的块恢复为 `pending`。若恢复也失败，可从报错日志核对块 ID。若存量块的 MySQL 正文和已发布索引不一致，暂停写入作业后按已核对的知识块 ID 将 MySQL 的 `vectorize_status` 恢复为 `pending`，再运行新版 `make kb-vectorize` 并检查实际检索结果。补跑沿用原知识块 ID。

## 真实用户部署

用环境变量 `MEWHELP_AUTH_MODE=required` 启用身份校验，`MEWHELP_AUTH_FILE` 指向仓库外的凭证 JSON 文件。仅经 HTTPS 反向代理访问应用，把数据库和业务 MCP 留在私网。由运营人员给每个账户分配随机 32 字节以上的凭证，文件只记录凭证 UTF-8 字符串的 SHA-256 十六进制摘要。重启服务加载变更。配置文件格式如下：

```json
{
  "customer-a": {"role": "customer", "sha256": "<64 个十六进制字符>"},
  "operator-a": {"role": "operator", "sha256": "<64 个十六进制字符>"}
}
```

浏览器通过 HTTP Basic 的登录框输入账户和随机凭证。客户使用聊天页，运营人员可访问管理页。API 客户端提交工单或退款时传 `Idempotency-Key`，同一次提交的重试复用该值。示例订单和物流依然来自模拟业务后端，真实订单接入时应沿用服务端归属校验与同一可信身份。

现有 SQLite 检查点和作业运行器按单应用进程部署。保持一个 Uvicorn worker，使同会话请求与同名作业共享互斥状态。

默认的 `demo` 模式兼容现有本机演示与课程脚本，仅用于本地测试。

## Langfuse 观测

配置 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` 和 `LANGFUSE_BASE_URL` 后，应用会导出意图标签、时延、模型与 token 用量。导出省略客服正文、工具输入和结果，并关闭 trace 媒体上传。会话使用 `LANGFUSE_SECRET_KEY` 派生的 HMAC 标识，轮换密钥后新 trace 会使用新的标识。

未配置或初始化失败时，聊天照常处理。遮蔽失败时，SDK 会丢弃该批次。更新只影响新导出，先前已保存的 trace 需要在 Langfuse 中按部署方的数据保留要求处理。

## 人工转接验收

在聊天页输入“转人工”并确认，记录工单号。运营账户打开 `/handoffs`，展开该工单，核对本轮问题以及已有的订单和检索参考，接单后填写对客户可见的结论并结单。回到同一聊天会话，在“人工工单”里刷新并查看处理状态与结论。点击取消时不会创建工单，同一轮重复确认沿用原工单号。
