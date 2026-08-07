# AI Developer Copilot —— LLM 开发知识库项目文档

> 本文将第一阶段开发方案与第二阶段演进规划整合为一份完整项目文档。项目的核心目标不变：构建一个面向 AI 开发者、能够提供精准、可追溯、可引用技术问答的 LLM 开发知识库。

## 一、项目概述

### 项目名称

**AI Developer Copilot —— LLM 开发知识库**

### 项目定位

构建一个面向 AI 开发者的智能知识库系统，整合官方文档、GitHub 开源项目、技术博客及论文等高可信度知识，通过 RAG（Retrieval-Augmented Generation）技术，为开发者提供精准、可追溯、可引用的技术问答服务。

项目并非普通聊天机器人，而是帮助 AI 工程师快速定位技术问题、理解源码设计、学习框架原理及解决开发 Bug 的专业知识助手。

### 核心价值

- 以高质量、可维护的知识为基础，而非仅依赖模型记忆。
- 通过检索增强生成，让回答可回溯至具体文档和内容片段。
- 为后续扩展企业级 AI 开发助手、代码知识库或多 Agent 系统保留统一的数据与架构基础。

---

## 二、项目目标、范围与阶段

### 总体目标

建立高质量的 **LLM 开发知识库**，使开发者能够检索 LLM 开发相关知识，并获得附带来源引用的技术回答。

### 第一阶段：可交付的基础 RAG 服务

第一阶段聚焦于知识库与问答闭环，包含以下能力：

- 高质量知识采集与本地归档。
- 文档导入、解析、清洗与文本切分。
- 使用 BGE-M3 生成向量并写入 Milvus。
- 基于向量检索获取相关知识片段。
- 通过 FastAPI 提供导入、解析、向量化、检索和问答服务。
- 调用 LLM 生成回答，并返回来源、引用文档和相似度信息。
- 使用 MySQL 记录文档元数据和处理状态，保证数据可管理、可追溯。

第一阶段不实现 Agent 执行、工具调用、多轮规划、混合检索和重排等高级能力。LangChain、LangGraph 中与 Agent、Tool、Memory 相关的资料可以作为知识内容收录，但系统本身不在第一阶段执行这些能力。

### 第二阶段：可扩展的智能检索与工程治理

第二阶段在不改变核心目标的前提下，提升检索质量、交互深度和工程治理能力：

- 检索质量：Hybrid Search（BM25 + Milvus）、Reranker（BGE Reranker）、多路召回、查询改写和 Query Expansion。
- 交互与知识范围：多轮对话、GitHub Issue 检索、Web Search、多知识库切换。
- 智能执行：Agent 与 MCP 工具调用。
- 工程治理：Redis 缓存、RAGAS 评测、Langfuse 可观测、用户权限管理。

第二阶段功能以新增适配器、策略实现和特性开关接入。第一阶段需要先固定数据标识、服务接口和 RAG 编排边界，保证后续升级不重写已有路由、核心问答流程或历史数据。完整设计见“第二阶段详细设计”与“跨阶段迭代与发布策略”。

---

## 三、总体方案

### 总体架构

```text
入库链路
知识来源（官方文档 / GitHub / 技术博客 / 论文 / 个人笔记）
                         │
                         ▼
                  Document Loader
                         │
                         ▼
                   文档解析 Parser
                         │
                         ▼
                   文本切分 Chunk
                    │          └──────→ MySQL（知识库、文档、版本与 Chunk 事实数据）
                    ▼
              Embedding（BGE-M3）
                    │
                    ▼
       Milvus（可重建的向量索引投影）

查询链路
用户提问
  │
  ▼
Query Embedding
  │
  ▼
Milvus 检索 TopK
  │
  ▼
Context Assemble / Prompt（必要时读取 MySQL 文档和引用元数据）
  │
  ▼
LLM（Qwen3 / DeepSeek）
  │
  ▼
回答 + 引用来源 + 引用文档 + 相似度
```

### 数据流与可追溯性

1. 导入原始文档并记录来源地址、类别、版本和采集时间。
2. 解析文档，保留标题、正文、章节层级及必要的来源信息。
3. 按语义或文档结构切分为 Chunk，为每个 Chunk 关联稳定的文档版本、序号和内容哈希，并写入 MySQL。
4. 生成向量后，将 Chunk 的向量和检索所需元数据作为索引投影写入 Milvus；文档级元数据、版本与处理状态保留在 MySQL。
5. 查询时返回命中的 Chunk，并将其来源信息一并组装进 LLM 上下文。
6. 问答接口以结构化引用返回支撑答案的文档，保证用户能够核验答案依据。

---

## 四、技术栈

### 第一阶段

| 模块 | 技术 | 用途 |
| --- | --- | --- |
| Web Framework | FastAPI | 提供文档处理、检索和问答 API |
| 向量数据库 | Milvus | 存储可重建的 Chunk 向量索引投影并执行相似度检索 |
| Embedding Model | BGE-M3 | 文档与查询向量化 |
| LLM | Qwen3 / DeepSeek | 基于检索上下文生成回答 |
| 文档解析 | Unstructured、Markdown、PyMuPDF | 解析不同格式的知识文档 |
| ORM | SQLAlchemy | 访问和管理关系型数据 |
| 元数据数据库 | MySQL | 作为事实数据源保存知识库、文档、版本、Chunk、来源和处理状态 |
| 部署 | Docker | 本地与服务化部署 |
| API 文档 | Swagger | 展示和调试 FastAPI 接口 |

### 第二阶段候选组件

| 模块 | 技术或实现方式 | 目的 |
| --- | --- | --- |
| 关键词检索 | BM25 全文索引 | 为语义检索补充精确关键词召回 |
| 融合检索 | RRF 或可配置融合策略 | 合并向量、BM25 与其他来源的候选结果 |
| Reranker | bge-reranker-v2-m3 | 对初步检索结果进行重排 |
| 缓存与会话 | Redis | 缓存热点查询、会话状态和限流信息 |
| 可观测性 | Langfuse | 记录调用链、Prompt、模型与检索轨迹 |
| RAG 评测 | RAGAS | 评估检索和回答质量，支撑回归比较 |
| 外部数据接入 | GitHub / Web Source Connector | 将外部资料规范化后接入同一入库链路 |
| 工具协议 | MCP Client / Tool Executor | 在受控条件下执行 Agent 工具调用 |

---

## 五、知识库范围与来源治理

### 知识范围

第一阶段仅围绕 **LLM 开发** 建设知识库

### 知识主题与优先来源

| 分类 | 重点内容 | 优先来源 |
| --- | --- | --- |
| PyTorch | nn.Module、Tensor、Autograd、DataLoader、Optimizer、CUDA、AMP、Distributed | PyTorch 官方文档 |
| Transformers / HuggingFace | AutoModel、Tokenizer、Trainer、Pipeline、Generation、Quantization | Hugging Face 官方文档 |
| PEFT | LoRA、QLoRA、Prefix Tuning、Prompt Tuning | PEFT 官方文档与原始论文 |
| LangChain | Runnable、PromptTemplate、Retriever、Memory、Agent、Tool | LangChain 官方文档 |
| LangGraph | StateGraph、Node、Edge、Memory、Checkpoint | LangGraph 官方文档 |
| LlamaFactory | 微调、数据集、LoRA、QLoRA、DPO、PPO | 官方文档、GitHub README、GitHub Wiki |
| vLLM | Serving、KV Cache、Prefix Cache、Tensor Parallel、OpenAI API | vLLM 官方文档 |
| CUDA | CUDA Memory、CUDA Stream、OOM、Device | NVIDIA 官方文档 |
| Docker | Dockerfile、Compose、GPU Runtime | Docker 官方文档 |
| Linux | 常用命令、文件权限、GPU 查看、Shell | 发行版或工具官方文档 |
| Papers | Attention Is All You Need、BERT、GPT-3、LoRA、QLoRA、RAG、Self-RAG、GraphRAG、ReAct | 原始论文及其官方发布页 |
| Personal Notes | Transformer 学习笔记、PyTorch 总结、Docker 部署经验、CUDA 踩坑记录、LLM 微调记录 | 经人工校验并标明参考来源的个人笔记 |

### 来源治理原则

- 优先收录官方文档、官方 GitHub 仓库和原始论文；技术博客与个人笔记作为补充。
- 每份文档至少记录标题、来源、URL、类别、版本或发布日期、采集时间。
- 文档更新时保留版本信息，避免不同版本内容在检索中混淆。
- 个人笔记应限定为 LLM 开发相关内容，并尽可能附上原始参考链接。

---

## 六、文档处理、存储与检索设计

### 文档处理流程

```text
导入文档
  → 格式识别与解析
  → 内容清洗与结构提取
  → Chunk 切分
  → 生成 Embedding
  → 写入 Milvus
  → 更新 MySQL 处理状态
```

### 存储职责

| 存储 | 职责 |
| --- | --- |
| 原始知识文件 | 保存下载或导入的原始文档，便于重新解析和审计 |
| MySQL | 作为事实数据源，保存知识库、文档、文档版本、Chunk、导入任务、来源、处理状态及失败原因 |
| Milvus | 保存可由 MySQL 和原始文件重建的向量索引投影，用于向量检索与必要的过滤 |

一期即应将 Chunk 正文、章节定位、内容哈希和版本关系写入 MySQL；Milvus 中的同类字段仅用于检索。这样二期增加 BM25、重排、索引切换或 Embedding 升级时，无需反向依赖或重建业务数据源。

### 跨阶段稳定数据约束

为保证一期升级到二期不需要大规模迁移或重写业务代码，一期应落实以下约束：

- `knowledge_base_id`、`document_id`、`document_version_id` 和 `chunk_id` 使用全局稳定 ID。Milvus 的 `id` 只作为物理记录 ID，不能出现在 API 引用或跨索引关联中。
- 一期创建唯一的 `default` 知识库，所有文档与查询都携带该范围或使用该默认值；二期新增知识库仅增加记录、权限和过滤条件。
- 文档更新创建新的 `document_version_id`，旧版本软下线但保留，保证历史回答中的引用仍可解析。
- `content_hash`、解析器版本、Chunk 切分配置和 Embedding 版本应由 MySQL 记录，用于幂等处理、回填和结果复现。
- Milvus Collection、BM25 索引和 Redis 缓存均为派生存储。业务代码通过索引注册或配置获取当前活跃索引，不能硬编码 `developer_docs` 作为唯一物理 Collection。
- 模型维度或 Collection Schema 不兼容时创建新索引代次并并行回填，完成校验后切换活跃索引；不得原地删除重建生产索引。

---

## 七、跨阶段架构与兼容契约

### 非破坏性迭代原则

第二阶段允许新增表、字段、索引、适配器、Worker 和配置，但不重写第一阶段的公开 API、入库状态机、引用结构、`ChatService` 主链路或数据事实源。具体约束如下：

- 先稳定领域模型、API Schema 和模块接口，再增加具体实现。
- 路由层只调用应用服务；应用服务依赖抽象接口，不能直接耦合 Milvus、Redis、LLM 或第三方 GitHub / Web SDK。
- 二期采用新增实现、组合或装饰器接入，避免在路由和核心编排中散布阶段判断。
- `/api/v1` 只允许增加可选字段和新端点，不删除字段、不修改已有字段含义或默认行为。确需破坏兼容性时，发布 `/api/v2` 并保留 `/api/v1` 适配期。
- 所有结构变更遵循“先扩展、后迁移、再切换、最后清理”的顺序；旧请求、旧数据和旧索引在观察期内持续可用。

---

## 八、项目目录设计

目录边界约定：

- `view` 是唯一对外入口，只负责 HTTP 路由、请求校验、响应封装和异常映射，不直接调用模型、数据库或工具。
- `services` 是业务主干，负责配置、依赖注入、知识库管理、检索、问答、任务状态和基础设施适配。
- `agents` 只负责任务理解、规划和工具编排；所有副作用能力必须通过 `tools.executor` 执行并留下审计记录。
- `chat_model` 只封装模型调用，不保存业务状态；切换 Qwen、DeepSeek、BGE 或后续本地模型时不影响 `view` 和 `agents`。
- `tools` 面向 Agent 提供可控能力，工具默认只读、白名单注册、参数强校验，并统一设置超时、权限和调用日志。
- 推荐调用方向为 `view -> services -> agents / chat_model / tools`。禁止 `chat_model`、`tools`、`agents` 反向依赖 `view`，避免接口层污染核心逻辑。

当前工作区目录名为 `chat_model`；如果后续希望与复数命名保持一致，可以统一调整为 `chat_models`，但文档和代码引用必须同步修改。



---

## 十、问答示例

用户提问：

> LoRA 为什么只训练 A、B 矩阵？

系统回答：

LoRA 将原始权重矩阵分解为两个低秩矩阵 A 和 B，仅训练新增参数，以减少显存占用和训练成本，同时保持较好的模型性能。

引用来源：

- HuggingFace PEFT Documentation
- LoRA: Low-Rank Adaptation of Large Language Models（论文）
- Personal Notes：LoRA 微调实践

---

## 十一、第一阶段开发计划与验收

### 开发计划

| 周期 | 工作内容 |
| --- | --- |
| 第一周 | 项目初始化、FastAPI 搭建、Milvus 部署、知识目录建立、稳定 ID 与默认知识库设计 |
| 第二周 | 文档采集、Markdown 解析、Chunk 切分、Embedding、MySQL Chunk 事实表与索引记录 |
| 第三周 | 检索模块、Prompt 拼接、LLM 接入、Retriever / Reranker 等稳定接口与 Noop 实现 |
| 第四周 | 来源引用、API 完善、契约测试、Docker 部署、README 编写 |

### 第一阶段验收标准

第一阶段完成后，系统应能够：

- 导入、解析、切分并向量化 LLM 开发相关文档。
- 在 Milvus 中检索与问题相关的 TopK 知识片段。
- 通过 API 返回基于检索上下文生成的技术回答。
- 为回答附带可定位的来源、引用文档和相似度信息。
- 使用稳定的知识库、文档、版本和 Chunk 标识，并通过可插拔检索接口保持后续策略扩展能力。
- 保证 `/api/v1` 的基础请求和响应 Schema 具备契约测试，作为第二阶段兼容基线。
- 通过 Docker 部署基础服务，并通过 Swagger 调试 API。

---

## 十二、第二阶段详细设计

### 演进目标与边界

第二阶段不是替换第一阶段的基础 RAG 服务，而是在保持一期 API、数据标识和问答语义稳定的前提下，提升检索质量、交互深度和工程治理能力。默认请求仍使用一期的单知识库向量检索；二期能力均通过检索策略、插件注册和特性开关逐步启用。

`/api/v1/chat` 始终是只读、可引用的知识问答入口。Agent、MCP 和具有副作用的工具调用必须走独立任务端点，并且经过显式授权、参数校验、超时控制和审计记录，不能让已有聊天接口悄然获得外部执行能力。

### 稳定的领域数据模型

一期即建立以下逻辑实体。即使暂时只使用默认知识库，也要写入默认值，避免二期引入多知识库、版本回溯或权限范围时重构主表。

| 实体 | 核心字段 | 设计要求 |
| --- | --- | --- |
| KnowledgeBase | `knowledge_base_id`、名称、状态、配置、修订号 | 一期创建 `default`；所有文档和查询都有明确知识库范围 |
| KnowledgeBaseMembership | `knowledge_base_id`、`principal_id`、角色、状态 | 一期由公共主体获得 `default` 的只读访问；二期扩展用户、组织和角色 |
| Document | `document_id`、`knowledge_base_id`、`source_type`、`source_uri`、分类、状态 | 表示稳定的逻辑文档，内容更新不更换 ID |
| DocumentVersion | `document_version_id`、`document_id`、版本号、`content_hash`、采集时间、解析器版本 | 每次更新创建新版本，不覆盖旧版本 |
| Chunk | `chunk_id`、`document_version_id`、`chunk_index`、正文、标题路径、偏移量、`content_hash` | Chunk ID 不复用；历史 Chunk 通过软下线保留 |
| ChunkIndexEntry | `chunk_id`、索引配置、物理索引名、向量记录 ID、索引代次、状态 | 解耦业务 Chunk 与 Milvus / BM25 物理索引 |
| IngestionJob | 作业 ID、动作、幂等键、状态、失败原因、重试次数 | 导入、解析和索引可恢复；二期换队列不改变业务语义 |
| RetrievalTrace | 请求 ID、策略、候选 Chunk、分数、模型版本、开关快照 | 一期可按需记录，二期用于评测、追踪和回放 |

`metadata_json` 可保存来源特有属性，例如 GitHub Issue 编号、网页发布日期或标签；用于过滤、排序、权限和关联的字段必须独立建列与索引，不能只存入 JSON。

### 检索质量升级

二期检索流程在一期向量检索前后增加可插拔步骤，但始终输出相同的 `EvidenceBundle`：

```text
原始问题
  → QueryTransformer（可选：改写 / 扩展）
  → Vector Retriever + BM25 Retriever（并行候选）
  → Result Fusion（可选：RRF）
  → Reranker（可选：BGE）
  → Context Assembler
  → LLM 回答 + 引用
```

实施要求：

1. 建立独立的 BM25 全文索引，复用同一批 `chunk_id`、`document_version_id` 和知识库过滤条件；BM25 不是另一套文档数据链路。
2. 向量与关键词检索分别生成候选集，再通过 RRF 等融合策略合并，避免直接相加 COSINE 分数和 BM25 分数。
3. Reranker 只对候选集重排，必须保留 `vector`、`lexical`、`rerank` 和 `final` 等阶段分数，便于解释和评测。
4. 查询改写和 Query Expansion 仅生成附加检索词，不修改或覆盖用户原始问题；原始问题、实际检索词和策略版本都写入 `RetrievalTrace`。
5. 新策略先进行影子检索和离线评测，达标后才灰度成为默认策略；调用失败时无条件回退 `vector-default-v1`。
6. 关键词独有的候选在进入 API 响应前，必须使用同一查询向量补齐 COSINE 相似度；RRF 和重排分数只记录在 `scores` 中。

一个可配置的二期检索策略示例：

```json
{
  "name": "hybrid-rerank-v1",
  "query_transformer": "none",
  "retrievers": [
    {"name": "vector", "top_k": 40},
    {"name": "lexical", "top_k": 40}
  ],
  "fusion": {"type": "rrf", "k": 60},
  "reranker": {"name": "bge-reranker-v2-m3", "top_n": 8}
}
```

### 多知识库、多轮对话与外部知识源

- **多知识库**：路由先创建 `RequestContext`，再由 `AuthorizationService` 计算可访问知识库范围，Retriever 只接收授权后的范围。二期新增知识库只需创建记录、配置索引和授权关系，已有 `default` 知识库及其文档无需迁移。
- **多轮对话**：新增 Conversation、Message 和 ConversationCitation 数据。服务端根据 `conversation_id` 加载受长度控制的历史消息；历史上下文只辅助理解问题，不能替代每次问答的检索证据。
- **GitHub Issue 与 Web Search**：统一实现为 `SourceConnector`，先规范化成 Document、DocumentVersion 和 Chunk，再复用已有导入、解析、切分、版本化和索引流程。不得为 Issue 或网页建立绕开主链路的独立数据模型。
- **即时网页证据**：必须携带来源 URL、抓取时间、可信度等级和来源类型。未经审核的结果不能覆盖已归档文档；网络失败、限流或可信度不足时，只降级为本地知识库检索。

### Agent、MCP 与任务执行

- Agent 通过 `POST /api/v1/agent/runs` 等独立端点创建任务，返回 `run_id`、状态、工具记录和引用证据；基础 RAG 链路始终可以单独运行。
- 所有工具经 `ToolGateway` / `ToolExecutor` 执行，并记录工具名称、输入摘要、输出引用、执行状态和请求 ID。
- 工具默认只读、白名单授权，并设置超时、参数校验和权限范围；具有副作用的操作必须要求用户再次确认。
- 长耗时 Agent、导入和外部检索任务共享 `IngestionJob` 风格的异步状态模型，支持查询状态、重试和取消。由一期 `InlineJobRunner` 切换到二期 Worker 时，业务接口保持不变。

### 工程治理

- **缓存**：Redis 仅缓存可失效的热点检索、会话摘要和限流状态。缓存键至少包含知识库、权限版本、知识库修订号、检索策略、规范化问题哈希和模型配置，防止跨知识库或跨权限复用结果。
- **可观测性**：Langfuse 通过 Trace Adapter 接入，不应成为问答可用性的前置依赖。每个 Trace 记录原始问题、最终查询、策略、索引代次、Prompt、模型版本、候选与最终引用、耗时、Token 用量和错误信息。
- **评测**：RAGAS 以版本化评测集和 `RetrievalTrace` 为输入，比较召回、引用正确性、答案有据率、延迟、成本和失败率。评测结果必须关联知识库修订号、策略和模型版本。
- **权限**：权限校验必须发生在检索前。一期的 `PublicAuthorizationService` 也要使用相同的 `RequestContext` 与服务接口；二期仅替换为用户、组织和角色实现，不改变 Document、Chunk、引用 ID 或检索调用链。

---

## 十三、第二阶段开发计划与验收

### 实施计划

| 周期 | 工作内容 | 与一期的衔接方式 |
| --- | --- | --- |
| 第一周 | 核验并完善稳定 ID、版本、Chunk 事实表、索引注册、任务状态、RequestContext / 授权接口与契约测试 | 仅做兼容性补强；默认仍走一期向量检索 |
| 第二周 | 建立 BM25 索引、HybridRetriever、RRF 融合和影子检索 | 复用现有 Chunk 与 `/api/v1/search` 响应结构 |
| 第三周 | 接入 BGE Reranker、查询改写、Query Expansion 和离线 RAG 评测 | 通过 `retrieval_profile` 与 Feature Flag 灰度启用 |
| 第四周 | 实现多知识库、会话存储、Redis 缓存和权限范围过滤 | 旧请求使用 `default` 知识库并保持无状态语义 |
| 第五周 | 接入 GitHub Issue、Web Source Connector、Langfuse 与评测看板 | 外部资料先走统一入库链路；故障时回退本地知识库 |
| 第六周 | 实现独立 Agent / MCP 任务、全链路压测、回滚演练和文档完善 | 不改变 `/api/v1/chat` 的基础 RAG 语义 |

### 第二阶段验收标准

- 所有一期 `/api/v1` 请求样例在二期服务中无需修改即可成功，且响应 JSON 仅新增可选字段。
- 关闭全部二期开关后，系统使用一期数据和 `vector-default-v1` 正常完成检索、问答和引用返回。
- 向量、BM25、融合、重排和查询改写使用同一组稳定 Chunk 标识；任一新模块异常时都能回退到基础向量检索。
- 从一期 MySQL 与 Milvus 快照迁移后，文档、文档版本、Chunk、来源 URL、处理状态和历史引用无丢失。
- 新 Embedding 或 Collection Schema 通过新索引代次回填和切换完成，不中断现网检索，并可回滚到上一代索引。
- 多知识库、会话、缓存、外部来源和 Agent 功能均经过 `RequestContext` 授权隔离与异常降级验证。
- 固定金标问题集的 RAG 回归评测不以牺牲引用可追溯性换取表面回答质量。

---

## 十四、跨阶段迭代与发布策略

### 数据与索引迁移顺序

所有结构调整遵循以下顺序，避免一次性改动源代码或生产数据：

1. **扩展**：先新增表、列、索引、配置和适配器；不删除、重命名或收紧一期字段。
2. **双写**：新旧索引或字段并存期间，按照 `chunk_id + content_hash` 幂等写入两侧；旧链路继续服务。
3. **回填**：后台按 `document_version_id` 回填 BM25、重排特征或新 Embedding，不阻塞在线检索。
4. **校验**：核对文档数、Chunk 数、内容哈希、引用可解析率、权限过滤和评测指标，记录迁移批次与失败项。
5. **影子与灰度**：新策略先只记录结果，再按知识库、用户或比例逐步启用；活跃索引通过配置或注册指针切换。
6. **回滚与清理**：异常时立即关闭开关并切回上一策略或索引。旧 Collection、兼容代码和数据映射在明确观察期后才清理。

### 兼容性测试基线

- **API 契约测试**：验证一期请求可在二期服务直接运行，OpenAPI 和响应 JSON 只增加可选字段。
- **模块契约测试**：每个 Retriever、EmbeddingProvider、VectorIndex、SourceConnector、ConversationStore 和 ToolGateway 使用同一组输入输出夹具验证。
- **迁移与回滚测试**：使用一期数据快照验证双写、回填、索引切换和回退后，历史引用仍能定位。
- **特性开关矩阵测试**：所有开关关闭时等同一期行为；任一单项或组合异常都可降级到基础 RAG。
- **端到端与隔离测试**：覆盖导入、版本更新、检索、聊天、引用定位、外部来源失败、Agent 任务失败，以及从一期公共主体到二期多知识库角色的权限隔离。

通过上述数据事实源、稳定接口、Feature Flag 和迁移策略，二期的工作将表现为“新增能力与配置”，而不是重做一期既有功能。

---

## 十五、项目最终价值

完成第一阶段后，系统具备面向 AI 开发者的专业知识问答能力，能够基于官方文档和高质量资料检索回答，并附带可追溯的引用来源。

第二阶段在不牺牲知识质量、可维护性和可追溯性的前提下，进一步提升检索准确性、交互深度、可观测性和企业级可用性。由于数据标识、服务契约与 RAG 编排在一期已保持稳定，项目可以持续扩展为企业级 AI 开发助手、代码知识库或多 Agent 系统，而无需推倒重写已有基础能力。
