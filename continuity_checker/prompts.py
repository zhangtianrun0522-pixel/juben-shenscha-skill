import json


def get_extraction_prompt(script_text):
    return f"""
你是资深剧本连续性审查助理。请从输入剧本文本中提取四类信息，并只输出严格 JSON，不要输出 Markdown，不要解释。

需要提取的信息：

1. assets：角色资产/外观/道具/身体特征等持续性元素。
字段：
- character: 角色名，无法确定则填"未知"
- asset_type: 资产类型，例如"服装"、"道具"、"身体特征"、"发型"、"妆容"、"伤口"、"车辆"等
- asset_name: 资产名称，尽量保留剧本中的具体表述
- episode: 集数，整数；无法判断则填1
- scene: 场次，整数；无法判断则填1
- page: 页码，整数或 null
- raw_text: 原文证据
- status: 状态，例如"exists"、"appears"、"disappears"、"damaged"、"lost"、"changed"等；默认"exists"
- is_vague: 是否是模糊指代，例如"那件衣服"、"那道疤"，布尔值

2. state_changes：状态变化，例如受伤到痊愈、衣服从干净到破损。
字段：
- character
- asset_type
- change_from
- change_to
- episode
- scene
- in_story_time_note: 剧情内时间说明，没有则 null

3. timeline：时间线锚点，特别是闪回。
字段：
- episode
- scene
- time_label
- is_flashback

4. character_settings：角色设定，如年龄、职业、固定背景、永久身体特征等。
字段：
- character
- setting_type
- content
- episode
- scene
- raw_text

输出 JSON 格式必须为：
{{
  "assets": [],
  "state_changes": [],
  "timeline": [],
  "character_settings": [],
  "identities": []
}}

注意：
- 不要编造文本中不存在的信息。
- 集数和场次如文本中出现"E01S03"、"第1集第3场"等，请正确解析。
- 如果同一资产被不同称呼提及，不要在此步骤合并，保留原始提取结果。
- 只输出 JSON。

剧本文本：
{script_text}
""".strip()


def get_normalization_prompt(assets_json: str) -> str:
    return f"""
你是剧本连续性审查系统中的“资产归一化”模块。

任务：
对输入的 AssetEntry 列表进行语义归一化，判断哪些条目指向同一个真实资产，并输出分组结果。

归组原则：
1. 只能在“同一角色 character、同一资产类型 asset_type”下归组。
2. 如果多个描述明显指向同一真实实物、同一身体特征、同一道伤疤、同一件衣服、同一件道具，则归为一组。
   例：
   - “黑色风衣”“深色大衣”“那件黑外套”“林墨的风衣”可以归为同一件服装。
   - “背部刀疤”“那道疤”“从颈根到腰侧的疤”“刀疤”可以归为同一身体特征。
3. 不同角色即使描述相似，也绝对不能归为一组。
4. 不同资产类型即使描述相似，也绝对不能归为一组。
5. 不确定时宁可不归组，避免错误合并。
6. canonical_name 请选择 aliases 中最具体、最稳定、最适合作为标准名称的那个名称。
7. aliases 包含该组内所有等价名称，去重。
8. entry_indices 是输入列表中属于该组的条目下标，从 0 开始。
9. 只需要输出包含至少 2 个 entry_indices 的分组；单个未归组条目无需输出，系统会自动处理。

输入是 AssetEntry 列表 JSON：
{assets_json}

输出格式必须是严格 JSON，不要 Markdown，不要注释，不要解释：
{{
  "groups": [
    {{
      "canonical_name": "背部刀疤",
      "asset_type": "身体特征",
      "character": "陆北辰",
      "aliases": ["背部刀疤", "那道疤", "刀疤"],
      "entry_indices": [0, 1]
    }}
  ]
}}
""".strip()


def get_semantic_review_prompt(asset_registry_json):
    return f"""
你是资深剧本连续性审查专家。请基于资产注册表进行语义层面的连续性审查，只输出严格 JSON。

输入 AssetRegistry JSON：
{asset_registry_json}

请查找机器规则可能难以发现的问题，例如：
- 角色设定与后文行为明显冲突
- 永久身体特征前后不一致
- 重要道具出现/消失缺少交代
- 时间线、闪回、现代线资产状态矛盾

输出格式：
{{
  "conflicts": [
    {{
      "rule_id": "SR01",
      "severity": "P1",
      "description": "问题描述",
      "character": "角色名或null",
      "episode_a": 1,
      "scene_a": 1,
      "episode_b": 1,
      "scene_b": 2,
      "raw_evidence": "证据"
    }}
  ]
}}

severity 只能是 "P0"、"P1"、"P2"、"P3"。
只输出 JSON。
""".strip()


def get_fix_suggestion_prompt(conflicts_json, asset_registry_json):
    return f"""
你是剧本修订顾问。请根据冲突列表和资产注册表生成修复建议，只输出严格 JSON。

冲突列表 JSON：
{conflicts_json}

资产注册表 JSON：
{asset_registry_json}

输出格式：
{{
  "suggestions": [
    {{
      "conflict_rule_id": "R03",
      "conflict_description": "冲突描述",
      "suggestion": "具体修复建议",
      "priority": "high"
    }}
  ]
}}

priority 可使用 high / medium / low。
不要输出 Markdown，不要解释，只输出 JSON。
""".strip()


def dumps_json_for_prompt(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)