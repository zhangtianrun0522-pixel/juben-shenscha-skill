# 剧本连贯性审查工具 (Continuity Checker)

剧本连贯性审查工具是一款基于大语言模型（LLM）的 Python 库，专为短剧、网剧剧本设计，自动检测角色视觉资产（外貌/服装/道具/伤疤等）及剧情时间线的跨集前后矛盾。它通过三步 LLM+规则流水线生成结构化冲突报告和修复建议，帮助编剧和制片团队在开机前发现剧本硬伤。

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 设置环境变量

```bash
export LLM_PROVIDER=anthropic       # 或 openai
export LLM_API_KEY=your_key_here
export LLM_MODEL=claude-sonnet-4-20250514   # 可选，覆盖默认模型
```

### 基本用法

```python
from continuity_checker import full_check, parse_file

# 从文件解析剧本
script = parse_file("script.pdf")   # 支持 .pdf / .docx / .txt

# 执行完整审查
report = full_check(script)

print(report.summary)
print(report.to_markdown())

# 导出 JSON（供 agent 消费）
with open("report.json", "w") as f:
    f.write(report.to_json())
```

---

## 核心接口

```python
# 文件解析
parse_file(file_path: str) -> str

# 三个子功能（可单独调用）
extract_assets(script_text: str) -> AssetRegistry
check_continuity(assets: AssetRegistry) -> ConflictReport
suggest_fixes(conflicts: ConflictReport, assets: AssetRegistry) -> FixSuggestions

# 主入口（最常用）
full_check(
    script_text: str,
    check_mode: str = "full",        # full | assets_only | conflicts_only
    severity_filter: str = "all"     # all | high | high_medium
) -> ContinuityReport
```

**check_mode 说明：**
- `full`：完整三步流水线（LLM提取 → 规则比对 → LLM语义审查 → 修复建议）
- `assets_only`：只提取资产档案，不做冲突检测
- `conflicts_only`：检测冲突，不生成修复建议

**severity_filter 说明：**
- `all`：返回全部级别冲突
- `high`：只返回 P0
- `high_medium`：返回 P0 + P1

---

## 审查机制

### 三步流水线

**Step 1 — LLM 信息提取**

调用大模型从剧本中提取四类结构化信息：
- 角色视觉资产（每次出现均记录）
- 状态变化节点（受伤、痊愈、道具消失等）
- 剧内时间线锚点（含闪回标记）
- 人物硬性设定（不喝酒、海鲜过敏等）

**Step 2 — 规则比对（纯代码，不调用 LLM）**

| 规则 | 检测内容 | 级别 |
|---|---|---|
| R01 | 资产首次出现在第 2 集或更晚，缺乏铺垫记录 | P3 |
| R02 | 受伤到痊愈跨度 ≤ 1 集且无特殊说明，恢复过快 | P1 |
| R03 | 同一角色同一集对同一资产有两条不同描述 | P1 |
| R04 | 闪回中存在的资产在现代时间线消失且无交代 | P1 |

**Step 3 — LLM 语义审查**

- 任务A：人物设定一致性（硬性设定是否被后续情节违反）
- 任务B：模糊资产标记（is_vague 条目的制作风险评估）
- 任务C：隐性连贯性问题（规则覆盖不到的逻辑矛盾）

### 冲突分级

| 级别 | 含义 | 处理建议 |
|---|---|---|
| P0 🔴 | 已确认矛盾，必须修改 | 编剧处理 |
| P1 🟠 | 高度疑似矛盾，需人工确认 | 编剧确认 |
| P2 🟡 | 模糊风险，建议补充描述 | 制作指导处理 |
| P3 🔵 | 信息备注（资产首次出现位置等） | 制作团队参考 |

---

## 输出格式

`full_check()` 返回 `ContinuityReport`：

```python
class ContinuityReport(BaseModel):
    script_language: str           # 自动检测，"zh" / "en"
    asset_registry: AssetRegistry  # 完整角色资产档案
    conflicts: ConflictReport      # 冲突列表（含各级别计数）
    fix_suggestions: FixSuggestions
    summary: str                   # 人类可读总结
```

内置导出方法：
- `report.to_json()` — 返回 JSON 字符串，供 agent 消费
- `report.to_markdown()` — 返回含冲突表格和修复建议的 Markdown 报告

完整输出示例见 `examples/sample_output.json`。

---

## Agent 接入

项目根目录的 `tool_schema.json` 符合 OpenAI function calling / tool 标准，可直接导入任何支持 function calling 的 agent 框架（Dify、OpenClaw、自定义 agent 等）。

```python
import json
from continuity_checker import full_check

# 在你的 agent 中注册工具
with open("tool_schema.json") as f:
    tool_schema = json.load(f)

# agent 收到 function call 后执行
def handle_continuity_check(script_text, check_mode="full", severity_filter="all"):
    report = full_check(script_text, check_mode, severity_filter)
    return json.loads(report.to_json())
```

---

## 环境变量

| 变量名 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `LLM_PROVIDER` | 是 | `anthropic` | `anthropic` 或 `openai` |
| `LLM_API_KEY` | 是 | 无 | 对应 provider 的 API 密钥 |
| `LLM_MODEL` | 否 | 见下 | 覆盖默认模型名 |

默认模型：anthropic → `claude-sonnet-4-20250514`，openai → `gpt-4o`
