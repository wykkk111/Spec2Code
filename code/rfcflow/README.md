### rfcflow（论文代码：可复现 workflow 版）

这个目录是一个**论文代码开源可运行**的 pipeline，目标是：

- **默认复现你原有实现逻辑（paper-compatible）**：SR 抽取/分组/关键词资源与后续阶段设计保持一致（只是把路径硬编码改成自动推断/相对路径，方便别人运行）。
- 同时提供一个 **quick 模式**：用于不装 clang/不配 LLM key 时快速跑通流程（它是 *fallback*，不用于复现实验结果）。

- **当前阶段（扩充后：paper-aligned）**
  - RFC 文本 → SR 抽取（对齐 `code/scripts/parse_rfc.py` 的输出设计）：保留 section/paragraph/sentence 索引与 `prev/next` 段落上下文
  - 源码目录 → 函数+依赖抽取（对齐 `code/scripts/code_analyzer.py` 的输出结构）：`called_functions/callers/used_structs/used_globals/used_macros/...`
  - filter1（对应 `code/scripts/protocol_filter.py`）：协议相关性过滤
  - dual_filter：keyword + 语义分类（subcategories）→ 映射到 SR cluster（`matching_sr_ids`）
  - pre_verify：对 (sr, func) 初筛
  - final_verify：最终验证（会把 dependencies 注入到 prompt）

这条链路可以直接产出 `final_verify.jsonl`（最终验证结果），用于后续统计/分析/一致性检查。

`group_info / keywords / prompt` 都作为**预定义资源**随目录提供（你也可以用参数指定外部路径）。

---

### 安装

```bash
pip install -r code/rfcflow/requirements.txt
```

如果你要用 LLM：
- 设置 `OPENAI_API_KEY`

如果你要做“函数依赖抽取（recommended）”：
- 需要本机有 **libclang**（`.so`），并通过环境变量或默认路径让 python clang bindings 找到它：
  - **推荐**：设置 `LIBCLANG_PATH=/path/to/libclang.so`
  - 本项目会尝试使用你原实验的默认路径（如果存在）：`/data/a/ykw/local/clang/lib/libclang.so`

---

### 运行

查看帮助：

```bash
python code/rfcflow/run.py --help
```

#### 0）环境变量（常用）

- **OPENAI_API_KEY**：跑 `filter/dual-filter/pre-verify/final-verify/extract-constraints` 需要
- **RFC_FINAL_ROOT**（可选）：兼容旧实验目录布局（存在 `raw_rfc/` 与 `data/<proj>/source_code/` 时可用）
- **LIBCLANG_PATH**（可选但推荐）：libclang 的绝对路径（用于依赖抽取）
- **RFCFLOW_MAX_ITEMS**（可选）：对 `filter/extract-constraints` 做小样本 smoke test 限流
- **RFCFLOW_MAX_PAIRS**（可选）：对 `pre-verify/final-verify` 做小样本 smoke test 限流

#### 目录布局（本 repo 默认）

- RFC 文本：`data/rfc/rawrfc/rfc8446.txt` / `data/rfc/rawrfc/rfc9110.txt` ...
- 源码：`data/repos/<project>/rawdata`（httpd 是 `rawcode`）

#### 1）抽 SR（不需要 API key）

```bash
python code/rfcflow/run.py extract-sr \
  --rfc-id 8446 \
  --out-dir /tmp/rfcflow_out
```

#### 2）抽函数（默认优先 clang+deps；失败会 fallback 到 regex）（不需要 API key）

```bash
python code/rfcflow/run.py extract-functions \
  --project boringssl \
  --out-dir /tmp/rfcflow_out
```

#### 3）filter1（协议相关性过滤）

- **论文复现模式（默认，与你原逻辑一致）**：需要 `OPENAI_API_KEY`
- **quick 模式（fallback）**：不需要 key，只用于跑通流程/调试，不用于复现实验结果

```bash
export OPENAI_API_KEY=...   # 论文复现模式需要

python code/rfcflow/run.py filter \
  --protocol tls \
  --out-dir /tmp/rfcflow_out
```

如果你只是想不配 key 跑通（fallback）：

```bash
python code/rfcflow/run.py filter --mode quick \
  --protocol tls \
  --out-dir /tmp/rfcflow_out
```

#### 4）dual_filter / pre_verify / final_verify（最终产出）

```bash
python code/rfcflow/run.py dual-filter \
  --protocol tls \
  --out-dir /tmp/rfcflow_out

python code/rfcflow/run.py extract-constraints \
  --protocol tls \
  --out-dir /tmp/rfcflow_out

python code/rfcflow/run.py pre-verify \
  --protocol tls \
  --out-dir /tmp/rfcflow_out

python code/rfcflow/run.py final-verify \
  --protocol tls \
  --out-dir /tmp/rfcflow_out
```

#### 输出文件一览（都在同一个 out-dir 内）

- `sr.jsonl` / `sr.csv`
- `functions.jsonl` / `functions.csv`
- `functions_filtered.jsonl`
- `dual_filter.jsonl`
- `sr_constraints.jsonl`
- `pre_verify.jsonl`
- `final_verify.jsonl`

---

### 关于 prompts（复现关键）

`RFC/rfcflow/assets/prompts/` 下的 prompt **应当与原实现保持一致**（复现论文结果的关键因素之一）。当前目录里的：
- `http_prompt.txt` / `http_user_prompt.txt`
- `tls_prompt.txt` / `tls_user_prompt.txt`

都来自你原目录 `RFC/final/prompt/` 的同名文件（保持一致用于复现）。


