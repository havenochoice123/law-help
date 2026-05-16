# ClassDesign — LLM 微调与 RAG 课程设计（最小原型）

本仓库是为课程项目“LLM 领域微调 + RAG 系统”准备的最小可运行原型（3 周冲刺版）。

主要目标
- 使用 DeepSeek-R1-1.5B（通过 ollama 部署）为基座模型，完成 LoRA / QLoRA quick-run。
- 构建混合检索（BM25 + 向量）小型索引（FAISS/本地实现）并实现 RAG demo（FastAPI）。
- 提供数据准备、微调与评估的脚本骨架，方便四人并行开发。

快速开始（smoke-test）
1. 创建并激活虚拟环境：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 使用 DISC-Law-SFT 数据（示例）生成 chunks：

```powershell
python scripts/data_prep.py --from-disc-law-sft data/raw/disc_law_sft.jsonl --out data/
```

3. 生成简单嵌入并构建索引：

```powershell
python scripts/build_embeddings.py --chunks_dir data/chunks/ --out outputs/embeddings.jsonl
python scripts/index_build.py --embeddings outputs/embeddings.jsonl --out outputs/indexes/
```

4. 运行 LoRA quick-run（展示命令，不强制执行训练）：

```powershell
python scripts/train_lora_quick.py --config configs/lora_quick.yaml --dry-run
python scripts/train_lora_quick.py --config configs/lora_quick.yaml
python scripts/train_qlora_quick.py --config configs/qlora_quick.yaml --dry-run
python scripts/verify_adapter.py --adapter-dir outputs/models/lora-quick
```

5. 启动 RAG demo：

```powershell
python scripts/rag_demo.py --port 8000
# 然后访问 http://localhost:8000/qa?question=你的问题
```

仓库结构（简要）
- configs/: 训练与索引配置样例
- data/: 原始、清洗、chunks 与 embeddings
- scripts/: 数据处理、训练、索引、检索与 demo
- outputs/: 模型、索引、评估结果

团队协作
- 推荐分支策略：feature 分支 -> PR -> main
- 请将 DISC-Law-SFT 放在 `data/raw/` 下并通知团队路径

如需我继续：我可以运行初步的语法检查并提交一个初始 commit（仅在本地创建文件）。

一览（3 周压缩目标）  
功能目标（3 周）：完成数据清洗/分片、LoRA quick-run + QLoRA quick-run（小规模试验）、构建小规模 BM25+FAISS 向量索引、实现一个 FastAPI RAG demo（可查询并返回来源）、并产出 1000 条评测 QA（或从 DISC-Law-SFT 中抽样/扩增到 1000 条）。    
评估目标：在 test set（≥1000）上跑 Baseline 与 LoRA/QLoRA（快速模型）以及至少一版微调+RAG 的自动指标（BLEU-4/ROUGE-L），并做人工抽样 60 条评估。  
团队规模：4 人（建议并行分工，见下文）。  
时间线与每日任务（3 周，工作日为 15 个工作日） 注：每人并行，可把任务细分为负责模块；示例人员 A/B/C/D。  
  
数据接入细则（DISC-Law-SFT）  
假定格式：每行 JSON 如 {"instruction": "...", "input": "...", "output": "...", "meta": {...}}。如果不是，请把样例粘给我，我会给转换脚本。  
最小转换脚本功能（scripts/data_prep.py --from-disc-law-sft）：  
读取 JSONL，做分句/清洗、去 HTML、去多余空格；  
生成两种输出：data/finetune/train.jsonl（SFT 格式：instruction+input+output），与 data/chunks/*.jsonl（id, text, title, source）；  
最小格式示例（微调 JSONL）： {"id":"sft-0001","instruction":"请根据下列法律条文回答…","input":"相关背景…","output":"期望答案…","meta":{"source":"DISC-Law-SFT","domain":"法律"}}  
LoRA / QLoRA quick-run 与 full-run 超参（建议）  
LoRA quick-run（用于 3 周内 demo）：  
micro_batch_size = 4 (per device)  
gradient_accumulation_steps = 8 (if needed)  
epochs = 1（quick），full: 3-5  
lr = 2e-4, warmup_steps = 100  
lora_r = 8, lora_alpha = 32, lora_dropout = 0.05  
peft_target_modules = ["q_proj","v_proj"]  
QLoRA quick-run（需要 bitsandbytes 支持 4-bit）：  
bnb_4bit = True, quant_type = "nf4", compute_dtype = "float16"  
micro_batch_size = 4, gradient_accumulation_steps = 8  
epochs = 1（quick）, full: 3  
lr = 1e-4  
lora_r = 16 可选  
LLaMA-Factory 快速训练示例（powershell）：  
python scripts/train_lora_quick.py --config configs/lora_quick.yaml --dry-run  
python scripts/train_lora_quick.py --config configs/lora_quick.yaml  
python scripts/train_qlora_quick.py --config configs/qlora_quick.yaml  
混合检索最小实现路径（优先快速可复现）  
轻量方案（快速上线）：  
BM25：Whoosh（纯 Python）或 Elasticsearch（如果已有服务）  
向量：FAISS（本地，CPU/GPU 均可）  
流程（retriever 简述）：  
BM25 top_k=20 -> 取 top 20 文本  
对 top_k 用嵌入计算向量相似度 -> re-rank top 5 (cosine)  
返回 top 3-5 片段用于拼接 context  
推荐参数：BM25 top_k=20, vector top_k=5, context token budget <= 2048（留 512 token 给模型生成）  
RAG 拼接与 prompt（中文示例）  
拼接顺序：最相关片段（每个片段标注来源） -> 简短“问题元信息” -> 指令模板。  
中文模板（示例）： 系统：你是法律领域助理，使用下面的知识片段并引用来源来回答问题。  
知识片段：{context}  
问题：{question}  
要求：请基于证据回答，并在回答末尾列出引用来源的 id 或 url；若知识不足，请诚实说明。  
生成参数（演示使用）：temperature=0.0, max_new_tokens=256, top_p=0.9  
评估设计（压缩为 3 周的可执行方案）  
自动指标：BLEU-4、ROUGE-L（跑在 1000 条 testset 上，若时间受限可跑 500 条）  
人工评估：抽样 60 条（20 条 per 模型场景或 30 条对比），人工给出 0/1/2 分，维度：正确性、完整性、流畅度、引用准确性  
统计方法：用 bootstrap 或 Wilcoxon 做两两比较（若时间短则给置信区间即可）  
Git 协作建议（适合 4 人短周期）  
推荐分支策略：main（可部署/演示稳定）+ feature branches (feat/data-prep, feat/train-lora, feat/index, feat/rag-demo)  
Pull Request 流程：每人负责 1-2 个 feature 分支；PR 至 main 前至少一人 review 并通过 CI（若无 CI，至少一人 reviewer）。  
Commit 风格：feat: add data_prep skeleton、fix: correct eval metric、docs: update README  
PR 与 issue 模板（建议放入 .github/）：简短说明、运行步骤、测试点。  
简单 git 命令（powershell）：  
git checkout -b feat/train-lora  
git add .  
git commit -m "feat: add lora quick training script"  
git push origin feat/train-lora  
合并（在 GitHub/GitLab）：创建 PR，assign reviewer，merge when approved。若你们偏爱命令行整合：git merge --no-ff feat/...（合并到 main 需先 pull 最新）。  
人员分配建议（4 人示例，按并行化）  
人员 A（数据工程）：data_prep.py, 负责 DISC-Law-SFT 转换、chunking、去重、embeddings pipeline。  
人员 B（训练工程）：train_lora_quick.py、train_qlora_quick.py、模型保存 & load 验证。  
人员 C（检索工程）：index_build.py、retriever.py、FAISS/Whoosh/ES 集成。  
人员 D（集成与评估）：rag_demo.py、evaluate.py、演示页面与 README、人工评估组织。  
每天 stand-up（10–15 分）同步进度。  
风险、压缩期注意事项与缓解  
风险：显存不足（QLoRA 可能失败） -> 缓解：优先 LoRA，QLoRA 仅做小规模尝试或使用 gradient accumulation + 4-bit。  
风险：DISC-Law-SFT 格式不一致 -> 缓解：第一天强行示例格式化脚本并发给组内复核。  
风险：时间不足做 1000 条人工评估 -> 缓解：自动指标先行，人工评估抽样 60 条做定性报告。  
风险：索引扩建耗时 -> 缓解：先构建小规模索引（demo 1000-5000 chunks），本地 FAISS 做快速实验。  
交付物（3 周最小可交付）  
代码：scripts/{data_prep.py,build_embeddings.py,index_build.py,train_lora_quick.py,train_qlora_quick.py,retriever.py,rag_demo.py,evaluate.py}  
配置：configs/{lora_quick.yaml,qlora_quick.yaml,index_config.json}  
文档：README.md（如何运行 quick-run smoke-test）  
结果：outputs/{models/,indexes/,eval/results.csv}  
快速 smoke-test（复现步骤） — powershell 命令块  
安装依赖（在项目根）：  
python -m venv .venv; .\.venv\Scripts\Activate.ps1  
pip install -r requirements.txt  
数据准备（假设 DISC-Law-SFT 存于 data/raw/disc_law_sft.jsonl）：  
python scripts/data_prep.py --from-disc-law-sft data/raw/disc_law_sft.jsonl --out data/  
构建 embeddings 并索引（FAISS）：  
python scripts/build_embeddings.py --input data/chunks/ --embed-model bge-small --out outputs/embeddings/  
python scripts/index_build.py --embeddings outputs/embeddings/ --out outputs/indexes/  
LoRA quick-run：  
python scripts/train_lora_quick.py --config configs/lora_quick.yaml  
python scripts/train_qlora_quick.py --config configs/qlora_quick.yaml  
python scripts/verify_adapter.py --adapter-dir outputs/models/lora-quick  
启动 RAG demo（本地 FastAPI）：  
python scripts/rag_demo.py --model outputs/models/lora-quick --index outputs/indexes/ --port 8000  
向 demo 发送请求（curl/wget 或浏览器）检查返回结果。  
Requirements coverage（本次变更说明）  
原计划所有点：已压缩并映射到 3 周可执行计划（Done）。  
代码实现：Deferred — 等你确认我会立即按计划创建仓库骨架并实现最小 demo（包含 quick-run 测试）。  
数据：DISC-Law-SFT 已有（你们已下载）——我在计划中给出了接入脚本说明与格式（Done）。  
