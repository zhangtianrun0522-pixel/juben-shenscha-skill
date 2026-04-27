def get_extraction_prompt(script_text: str) -> str:
    return f"""你是一个专业的影视剧本连贯性审查专家。请从以下剧本内容中提取四类信息，并以纯JSON格式输出。

【剧本内容】
{script_text}

【提取任务与字段说明】

1. assets（角色视觉资产，每次出现都必须单独记录，不可只记首次）：
- character: 角色名称
- asset_type: 资产类型（如：服装、配饰、发型、妆容、道具、身体特征等）
- asset_name: 资产具体名称或描述
- episode: 集数（整数）
- scene: 场次（整数）
- page: 页码（整数或null）
- raw_text: 剧本中提及该资产的原文
- status: 状态，默认填写"exists"
- is_vague: 布尔值，描述是否模糊

2. state_changes（状态变化节点）：
- character: 角色名称
- asset_type: 发生变化的资产类型
- change_from: 变化前的状态
- change_to: 变化后的状态
- episode: 集数（整数）
- scene: 场次（整数）
- in_story_time_note: 故事内时间说明（字符串或null）

3. timeline（时间线锚点）：
- episode: 集数（整数）
- scene: 场次（整数）
- time_label: 时间点描述
- is_flashback: 是否为闪回（布尔值）

4. character_settings（人物硬性设定）：
- character: 角色名称
- setting_type: 设定类型（如：身份、职业、生理特征、习惯、人际关系等）
- content: 设定具体内容
- episode: 集数（整数）
- scene: 场次（整数）
- raw_text: 剧本中提及该设定的原文

【特别提示】
- 动作暗示：密切关注"动作暗示"的隐性视觉信息。例如"他下意识捂住左颈"暗示该处存在疤痕或伤痕，必须将其记为隐性资产线索（asset_type可标为"隐性视觉特征"），并在raw_text中保留原文。
- 模糊描述：服装描述中若使用模糊词汇（如"便装"、"日常穿着"、"正式场合着装"、"华丽服饰"等缺乏具体视觉指向的词），必须将is_vague标记为true。如果描述具体明确（如"黑色双排扣西装"），则is_vague为false。
- 重复记录：同一视觉资产在同一角色的不同场次出现时，每次出现都必须单独记录一条，绝不可省略或合并。
- 输出语言：请使用与剧本原文相同的语言进行描述和提取。

【输出格式】
严格输出纯JSON对象，不要包含任何markdown标记、注释或额外说明。格式如下：
{{"assets":[...],"state_changes":[...],"timeline":[...],"character_settings":[...]}}"""


def get_semantic_review_prompt(asset_registry_json: str) -> str:
    return f"""你是一个顶级的影视剧本连贯性审查专家。你将接收一份资产档案JSON，请基于全局语义理解执行以下三个审查任务，找出其中的冲突与问题。

【资产档案数据】
{asset_registry_json}

【审查任务】

任务A：人物设定一致性审查
检查character_settings中的硬性设定是否在后续情节（assets和state_changes）中被违反。例如：设定为"左腿残疾需拄拐"，但后续asset中出现"奔跑"或未携带拐杖的记录。

任务B：视觉资产描述模糊性标记
针对is_vague为true的资产条目，评估其模糊程度和制作风险。如果模糊描述可能导致美术、服装或道具部门理解歧义或穿帮，需报告该问题，并给出具体的补充描述建议。

任务C：隐性连贯性问题
基于全局语义理解，寻找规则检测难以覆盖的逻辑矛盾。例如：角色在雨中淋透后，下一场戏没有换装却表现为干燥状态；或者在已知时间线约束下，角色不可能在极短时间内跨越遥远地点。

【输出格式】
找出所有冲突和问题，严格输出纯JSON对象，不要包含任何markdown标记、注释或额外说明。输出语言须与档案数据中的原文语言一致。格式如下：
{{"conflicts":[{{"rule_id":"A1或B1或C1等，按顺序递增","severity":"P0(致命穿帮)/P1(明显逻辑漏洞)/P2(模糊风险)/P3(轻微建议)","description":"对冲突的详细描述","character":"涉及的角色名称","episode_a":null,"scene_a":null,"episode_b":null,"scene_b":null,"raw_evidence":"引发冲突的原文证据"}}]}}"""


def get_fix_suggestion_prompt(conflicts_json: str, asset_registry_json: str) -> str:
    return f"""你是一个资深的影视剧本医生。你将接收到一组已发现的连贯性冲突列表和完整的资产档案，请为每一条冲突提供具体、可执行的修复建议。

【冲突列表】
{conflicts_json}

【资产档案】
{asset_registry_json}

【任务要求】
针对每一条冲突，结合资产档案中的上下文，提供具体的修改建议。建议应当：
1. 针对性强：直接指出应修改哪一场戏、哪个角色的哪项描述。
2. 可执行性高：给出具体的替换文本、增补方案或调整方案，而非空泛的原则。
3. 优先级匹配：严格按照冲突本身的严重程度给出相应紧急程度的建议。

【输出格式】
严格输出纯JSON对象，不要包含任何markdown标记、注释或额外说明。输出语言须与冲突和档案中的原文语言一致。格式如下：
{{"suggestions":[{{"conflict_rule_id":"对应的冲突rule_id","conflict_description":"对应的冲突描述","suggestion":"具体可执行的修复建议","priority":"P0/P1/P2/P3，与冲突严重程度对应"}}]}}"""
