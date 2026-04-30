import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .models import (
    AssetEntry,
    AssetIdentity,
    AssetRegistry,
    CharacterSetting,
    Conflict,
    ConflictReport,
    ContinuityReport,
    FixSuggestion,
    FixSuggestions,
    StateChange,
    TimelineAnchor,
)
from .prompts import (
    get_extraction_prompt,
    get_fix_suggestion_prompt,
    get_normalization_prompt,
    get_semantic_review_prompt,
)


def _model_dump(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"Object {type(obj)} is not a pydantic model")


def _model_dump_json(obj: Any, indent: Optional[int] = None) -> str:
    if hasattr(obj, "model_dump_json"):
        try:
            return obj.model_dump_json(indent=indent, ensure_ascii=False)
        except TypeError:
            return obj.model_dump_json(indent=indent)
    if hasattr(obj, "json"):
        return obj.json(indent=indent, ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False, indent=indent)


def _parse_json_from_text(text: str) -> Any:
    """
    Robustly parse JSON from an LLM response.

    Accepts:
    - pure JSON
    - fenced ```json blocks
    - text containing a top-level JSON object/array
    """
    if text is None:
        raise ValueError("Cannot parse JSON from None")

    stripped = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    candidates = []
    first_obj = stripped.find("{")
    last_obj = stripped.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(stripped[first_obj : last_obj + 1])

    first_arr = stripped.find("[")
    last_arr = stripped.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(stripped[first_arr : last_arr + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Failed to parse JSON from LLM response: {text[:500]}")


def _call_llm(prompt: str, llm_client: Any = None, model: Optional[str] = None) -> Optional[str]:
    """
    Generic LLM caller.

    Supported usages:
    1. llm_client is a callable: llm_client(prompt) -> str
    2. llm_client has chat.completions.create compatible OpenAI client
    3. llm_client has complete(prompt) -> str
    4. No client but OPENAI_API_KEY exists: use openai.OpenAI

    Returns None if no callable client is available.
    """
    if llm_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                llm_client = OpenAI(api_key=api_key)
            except Exception:
                llm_client = None

    if llm_client is None:
        return None

    if callable(llm_client):
        result = llm_client(prompt)
        return str(result) if result is not None else None

    if hasattr(llm_client, "complete") and callable(llm_client.complete):
        result = llm_client.complete(prompt)
        return str(result) if result is not None else None

    if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
        chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = llm_client.chat.completions.create(
            model=chosen_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的剧本连续性审查助手。所有回答必须遵循用户要求的 JSON 格式。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    return None


def _asset_registry_from_dict(data: Dict[str, Any]) -> AssetRegistry:
    """
    Convert arbitrary dict to AssetRegistry while tolerating missing keys.
    """
    if data is None:
        data = {}

    assets = data.get("assets", [])
    state_changes = data.get("state_changes", [])
    timeline = data.get("timeline", [])
    character_settings = data.get("character_settings", [])
    identities = data.get("identities", [])

    return AssetRegistry(
        assets=[AssetEntry(**item) for item in assets if isinstance(item, dict)],
        state_changes=[StateChange(**item) for item in state_changes if isinstance(item, dict)],
        timeline=[TimelineAnchor(**item) for item in timeline if isinstance(item, dict)],
        character_settings=[
            CharacterSetting(**item) for item in character_settings if isinstance(item, dict)
        ],
        identities=[AssetIdentity(**item) for item in identities if isinstance(item, dict)],
    )


def extract_assets(script_text: str, llm_client: Any = None) -> AssetRegistry:
    """
    Extract AssetRegistry from script text.

    If no LLM is configured, returns an empty registry so the package remains runnable.
    In production, pass an OpenAI-compatible client or set OPENAI_API_KEY.
    """
    prompt = get_extraction_prompt(script_text)
    response = _call_llm(prompt, llm_client=llm_client)

    if not response:
        return AssetRegistry()

    data = _parse_json_from_text(response)
    if not isinstance(data, dict):
        raise ValueError("Extraction response must be a JSON object")

    return _asset_registry_from_dict(data)


def _safe_id_part(value: str) -> str:
    value = str(value or "未知").strip()
    value = re.sub(r"\s+", "_", value)
    value = value.replace("/", "_").replace("\\", "_")
    return value or "未知"


def _make_asset_id(character: str, asset_type: str, seq: int) -> str:
    return f"{_safe_id_part(character)}_{_safe_id_part(asset_type)}_{seq:03d}"


def _choose_canonical_name(names: Iterable[str]) -> str:
    cleaned = []
    seen = set()
    for name in names:
        n = str(name or "").strip()
        if not n or n in seen:
            continue
        seen.add(n)
        cleaned.append(n)

    if not cleaned:
        return "未知资产"

    vague_markers = ("那", "这", "某", "一件", "一个", "一道", "其", "它", "他", "她")
    ranked = sorted(
        cleaned,
        key=lambda x: (
            any(marker in x for marker in vague_markers),
            -len(x),
        ),
    )
    return ranked[0]


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _entry_sort_key(entry: AssetEntry) -> Tuple[int, int]:
    return entry.episode, entry.scene


def _fallback_normalization_groups(assets: List[AssetEntry]) -> List[Dict[str, Any]]:
    """
    Conservative heuristic fallback when no LLM is available.

    It groups only highly similar entries under the same character and asset_type.
    This is intentionally conservative to respect “不确定时宁可不归组”.
    """
    groups = []
    buckets: Dict[Tuple[str, str], List[Tuple[int, AssetEntry]]] = defaultdict(list)
    for idx, entry in enumerate(assets):
        buckets[(entry.character, entry.asset_type)].append((idx, entry))

    vague_words = {"那件", "那道", "那个", "这件", "这道", "这个", "其", "他的", "她的", "的"}
    color_words = {"黑色", "深色", "黑", "白色", "白", "红色", "红", "蓝色", "蓝", "灰色", "灰"}

    def normalize_name(name: str) -> str:
        s = str(name or "")
        for word in vague_words:
            s = s.replace(word, "")
        return s.strip()

    def token_set(name: str) -> Set[str]:
        s = normalize_name(name)
        tokens = set()
        for word in color_words:
            if word in s:
                tokens.add(word)
        for keyword in [
            "风衣",
            "大衣",
            "外套",
            "刀疤",
            "疤",
            "疤痕",
            "戒指",
            "项链",
            "手表",
            "枪",
            "刀",
            "包",
            "帽",
            "眼镜",
        ]:
            if keyword in s:
                tokens.add(keyword)
        if not tokens and s:
            tokens.add(s)
        if "黑色" in tokens:
            tokens.add("黑")
        if "深色" in tokens:
            tokens.add("黑")
        if "疤痕" in tokens:
            tokens.add("疤")
        if "刀疤" in tokens:
            tokens.add("疤")
        if "风衣" in tokens or "大衣" in tokens:
            tokens.add("外套")
        return tokens

    for (character, asset_type), indexed_entries in buckets.items():
        used = set()
        for i, entry_a in indexed_entries:
            if i in used:
                continue
            current_indices = [i]
            current_names = [entry_a.asset_name]
            tokens_a = token_set(entry_a.asset_name)
            if not tokens_a:
                continue

            for j, entry_b in indexed_entries:
                if j <= i or j in used:
                    continue
                tokens_b = token_set(entry_b.asset_name)
                if not tokens_b:
                    continue

                overlap = tokens_a & tokens_b
                if not overlap:
                    continue

                # Conservative: require a core object token overlap, not only a color.
                core_overlap = overlap - color_words
                if core_overlap:
                    current_indices.append(j)
                    current_names.append(entry_b.asset_name)
                    used.add(j)

            if len(current_indices) >= 2:
                used.add(i)
                groups.append(
                    {
                        "canonical_name": _choose_canonical_name(current_names),
                        "asset_type": asset_type,
                        "character": character,
                        "aliases": _dedupe_keep_order(current_names),
                        "entry_indices": current_indices,
                    }
                )

    return groups


def normalize_assets(registry: AssetRegistry, llm_client: Any = None) -> AssetRegistry:
    """
    Normalize semantically equivalent AssetEntry records into AssetIdentity records.

    Steps:
    1. Send registry.assets to LLM with get_normalization_prompt.
    2. Parse groups containing canonical_name / aliases / entry_indices.
    3. Generate asset_id in format {character}_{asset_type}_{三位序号}.
    4. Assign asset_id back to each AssetEntry.
    5. Create AssetIdentity list.
    6. For entries not covered by LLM groups, create one identity per entry.

    If LLM is unavailable or returns invalid JSON, uses a conservative heuristic fallback.
    """
    assets = registry.assets or []

    for entry in assets:
        entry.asset_id = None

    if not assets:
        registry.identities = []
        return registry

    assets_json = json.dumps([_model_dump(a) for a in assets], ensure_ascii=False, indent=2)
    prompt = get_normalization_prompt(assets_json)

    groups: List[Dict[str, Any]] = []
    response = _call_llm(prompt, llm_client=llm_client)

    if response:
        try:
            parsed = _parse_json_from_text(response)
            if isinstance(parsed, dict) and isinstance(parsed.get("groups"), list):
                groups = [g for g in parsed["groups"] if isinstance(g, dict)]
        except Exception:
            groups = []

    if not groups:
        groups = _fallback_normalization_groups(assets)

    sequence_by_key: Dict[Tuple[str, str], int] = defaultdict(int)
    identities: List[AssetIdentity] = []
    assigned_indices: Set[int] = set()

    def next_id(character: str, asset_type: str) -> str:
        key = (character, asset_type)
        sequence_by_key[key] += 1
        return _make_asset_id(character, asset_type, sequence_by_key[key])

    normalized_groups = []

    for group in groups:
        raw_indices = group.get("entry_indices", [])
        if not isinstance(raw_indices, list):
            continue

        indices = []
        for idx in raw_indices:
            try:
                int_idx = int(idx)
            except Exception:
                continue
            if 0 <= int_idx < len(assets):
                indices.append(int_idx)

        indices = sorted(set(indices))
        indices = [idx for idx in indices if idx not in assigned_indices]

        if len(indices) < 2:
            continue

        entries = [assets[idx] for idx in indices]

        # Enforce hard safety rule: same character and same asset_type only.
        characters = {entry.character for entry in entries}
        asset_types = {entry.asset_type for entry in entries}
        if len(characters) != 1 or len(asset_types) != 1:
            continue

        character = entries[0].character
        asset_type = entries[0].asset_type

        aliases_from_group = group.get("aliases", [])
        if not isinstance(aliases_from_group, list):
            aliases_from_group = []

        aliases = _dedupe_keep_order(
            list(aliases_from_group) + [entry.asset_name for entry in entries]
        )
        canonical_name = str(group.get("canonical_name") or "").strip()
        if not canonical_name:
            canonical_name = _choose_canonical_name(aliases)
        if canonical_name not in aliases:
            aliases.insert(0, canonical_name)

        first_entry = sorted(entries, key=_entry_sort_key)[0]
        asset_id = next_id(character, asset_type)

        for idx in indices:
            assets[idx].asset_id = asset_id
            assigned_indices.add(idx)

        identity = AssetIdentity(
            asset_id=asset_id,
            canonical_name=canonical_name,
            aliases=aliases,
            character=character,
            asset_type=asset_type,
            first_episode=first_entry.episode,
            first_scene=first_entry.scene,
            is_vague=any(entry.is_vague for entry in entries),
        )
        normalized_groups.append(identity)

    identities.extend(normalized_groups)

    for idx, entry in enumerate(assets):
        if idx in assigned_indices:
            continue

        asset_id = next_id(entry.character, entry.asset_type)
        entry.asset_id = asset_id

        identities.append(
            AssetIdentity(
                asset_id=asset_id,
                canonical_name=entry.asset_name,
                aliases=[entry.asset_name],
                character=entry.character,
                asset_type=entry.asset_type,
                first_episode=entry.episode,
                first_scene=entry.scene,
                is_vague=entry.is_vague,
            )
        )
        assigned_indices.add(idx)

    registry.assets = assets
    registry.identities = identities
    return registry


def _is_negative_status(status: str) -> bool:
    value = str(status or "").lower()
    negative_keywords = [
        "disappear",
        "disappears",
        "disappeared",
        "missing",
        "lost",
        "absent",
        "removed",
        "gone",
        "消失",
        "不见",
        "遗失",
        "丢失",
        "脱下",
        "移除",
        "没有",
    ]
    return any(keyword in value for keyword in negative_keywords)


def _is_positive_status(status: str) -> bool:
    value = str(status or "").lower()
    positive_keywords = [
        "exists",
        "appear",
        "appears",
        "appeared",
        "present",
        "wearing",
        "holding",
        "has",
        "存在",
        "出现",
        "穿着",
        "拿着",
        "戴着",
        "带着",
        "有",
    ]
    return any(keyword in value for keyword in positive_keywords) or not _is_negative_status(value)


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    value = str(text or "").lower()
    return any(str(keyword).lower() in value for keyword in keywords)


def _timeline_is_flashback(registry: AssetRegistry, episode: int, scene: int) -> bool:
    for anchor in registry.timeline:
        if anchor.episode == episode and anchor.scene == scene:
            return bool(anchor.is_flashback)

    label_keywords = ["flashback", "回忆", "闪回", "过去", "多年前", "年前"]
    for anchor in registry.timeline:
        if anchor.episode == episode and anchor.scene == scene:
            if _contains_any(anchor.time_label, label_keywords):
                return True

    return False


def check_continuity(assets: AssetRegistry) -> ConflictReport:
    """
    Core rule-based continuity checker.

    Rules:
    - R01: asset首次出现 > 第1集 → P3
    - R02: 受伤 → 痊愈 ≤ 1集 → P1
    - R03: 同集同角色同asset_type存在多个不同真实资产 → P1
           Normalized behavior:
           * same asset_id: no conflict
           * different asset_id but same asset_type: conflict
           * both asset_id are None: fallback to old asset_name comparison
    - R04: 现代线资产消失但闪回中仍存在 → P1
    """
    conflicts: List[Conflict] = []
    registry = assets

    # R01
    first_seen_by_asset: Dict[str, AssetEntry] = {}
    for entry in registry.assets:
        key = entry.asset_id or f"{entry.character}|{entry.asset_type}|{entry.asset_name}"
        current = first_seen_by_asset.get(key)
        if current is None or _entry_sort_key(entry) < _entry_sort_key(current):
            first_seen_by_asset[key] = entry

    for entry in first_seen_by_asset.values():
        if entry.episode > 1 and _is_positive_status(entry.status):
            conflicts.append(
                Conflict(
                    rule_id="R01",
                    severity="P3",
                    description=(
                        f"资产“{entry.asset_name}”首次出现于第{entry.episode}集第{entry.scene}场，"
                        f"可能需要提前铺垫。"
                    ),
                    character=entry.character,
                    episode_a=entry.episode,
                    scene_a=entry.scene,
                    raw_evidence=entry.raw_text,
                )
            )

    # R02
    injury_keywords = ["受伤", "伤", "流血", "骨折", "中刀", "枪伤", "injured", "wounded", "bleeding"]
    heal_keywords = ["痊愈", "好了", "恢复", "愈合", "无碍", "healed", "recovered"]

    changes_by_character_type: Dict[Tuple[str, str], List[StateChange]] = defaultdict(list)
    for change in registry.state_changes:
        changes_by_character_type[(change.character, change.asset_type)].append(change)

    for (character, asset_type), changes in changes_by_character_type.items():
        sorted_changes = sorted(changes, key=lambda c: (c.episode, c.scene))
        injuries = [
            c
            for c in sorted_changes
            if _contains_any(c.change_from, injury_keywords)
            or _contains_any(c.change_to, injury_keywords)
        ]
        heals = [
            c
            for c in sorted_changes
            if _contains_any(c.change_from, heal_keywords)
            or _contains_any(c.change_to, heal_keywords)
        ]

        for injury in injuries:
            for heal in heals:
                if (heal.episode, heal.scene) <= (injury.episode, injury.scene):
                    continue
                if heal.episode - injury.episode <= 1:
                    conflicts.append(
                        Conflict(
                            rule_id="R02",
                            severity="P1",
                            description=(
                                f"{character}的{asset_type}从“{injury.change_to}”到“{heal.change_to}”"
                                f"间隔不超过1集，可能恢复过快。"
                            ),
                            character=character,
                            episode_a=injury.episode,
                            scene_a=injury.scene,
                            episode_b=heal.episode,
                            scene_b=heal.scene,
                            raw_evidence=f"{injury.change_from}->{injury.change_to}; "
                            f"{heal.change_from}->{heal.change_to}",
                        )
                    )
                    break

    # R03
    entries_by_episode_character_type: Dict[Tuple[int, str, str], List[AssetEntry]] = defaultdict(list)
    for entry in registry.assets:
        if _is_positive_status(entry.status):
            entries_by_episode_character_type[
                (entry.episode, entry.character, entry.asset_type)
            ].append(entry)

    for (episode, character, asset_type), entries in entries_by_episode_character_type.items():
        if len(entries) < 2:
            continue

        reported_pairs: Set[Tuple[str, str]] = set()

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a = entries[i]
                b = entries[j]

                if a.scene == b.scene and a.asset_name == b.asset_name:
                    continue

                if a.asset_id is not None and b.asset_id is not None:
                    if a.asset_id == b.asset_id:
                        continue
                    should_report = True
                    pair_key = tuple(sorted([a.asset_id, b.asset_id]))
                elif a.asset_id is None and b.asset_id is None:
                    should_report = a.asset_name != b.asset_name
                    pair_key = tuple(sorted([a.asset_name, b.asset_name]))
                else:
                    # Partial normalization fallback: if one has id and the other does not,
                    # treat different names under same character/type/episode as suspicious.
                    should_report = a.asset_name != b.asset_name
                    pair_key = tuple(sorted([a.asset_id or a.asset_name, b.asset_id or b.asset_name]))

                if not should_report or pair_key in reported_pairs:
                    continue

                reported_pairs.add(pair_key)
                conflicts.append(
                    Conflict(
                        rule_id="R03",
                        severity="P1",
                        description=(
                            f"第{episode}集内，{character}的同类资产“{asset_type}”出现多个不同资产："
                            f"“{a.asset_name}”(asset_id={a.asset_id or 'None'}) 与 "
                            f"“{b.asset_name}”(asset_id={b.asset_id or 'None'})。"
                        ),
                        character=character,
                        episode_a=a.episode,
                        scene_a=a.scene,
                        episode_b=b.episode,
                        scene_b=b.scene,
                        raw_evidence=f"{a.raw_text}\n{b.raw_text}",
                    )
                )

    # R04
    entries_by_asset: Dict[str, List[AssetEntry]] = defaultdict(list)
    for entry in registry.assets:
        key = entry.asset_id or f"{entry.character}|{entry.asset_type}|{entry.asset_name}"
        entries_by_asset[key].append(entry)

    for key, entries in entries_by_asset.items():
        sorted_entries = sorted(entries, key=_entry_sort_key)
        modern_negative_entries = [
            e
            for e in sorted_entries
            if _is_negative_status(e.status) and not _timeline_is_flashback(registry, e.episode, e.scene)
        ]
        flashback_positive_entries = [
            e
            for e in sorted_entries
            if _is_positive_status(e.status) and _timeline_is_flashback(registry, e.episode, e.scene)
        ]

        for neg in modern_negative_entries:
            for fb in flashback_positive_entries:
                if (fb.episode, fb.scene) >= (neg.episode, neg.scene):
                    conflicts.append(
                        Conflict(
                            rule_id="R04",
                            severity="P1",
                            description=(
                                f"现代线中资产“{neg.asset_name}”已标记为消失/缺失，"
                                f"但之后的闪回场景仍显示其存在。请确认时间线或状态描述。"
                            ),
                            character=neg.character,
                            episode_a=neg.episode,
                            scene_a=neg.scene,
                            episode_b=fb.episode,
                            scene_b=fb.scene,
                            raw_evidence=f"{neg.raw_text}\n{fb.raw_text}",
                        )
                    )
                    break

    return ConflictReport.from_conflicts(conflicts)


def semantic_review(registry: AssetRegistry, llm_client: Any = None) -> ConflictReport:
    """
    Optional semantic review powered by LLM.
    Returns empty report if no LLM is available.
    """
    registry_json = _model_dump_json(registry, indent=2)
    prompt = get_semantic_review_prompt(registry_json)
    response = _call_llm(prompt, llm_client=llm_client)

    if not response:
        return ConflictReport.from_conflicts([])

    try:
        parsed = _parse_json_from_text(response)
        if not isinstance(parsed, dict):
            return ConflictReport.from_conflicts([])
        raw_conflicts = parsed.get("conflicts", [])
        conflicts = [Conflict(**item) for item in raw_conflicts if isinstance(item, dict)]
        return ConflictReport.from_conflicts(conflicts)
    except Exception:
        return ConflictReport.from_conflicts([])


def generate_fix_suggestions(
    conflicts: ConflictReport,
    registry: AssetRegistry,
    llm_client: Any = None,
) -> FixSuggestions:
    """
    Generate fix suggestions.

    Uses LLM when available; otherwise returns deterministic rule-based suggestions.
    """
    if not conflicts.conflicts:
        return FixSuggestions(suggestions=[])

    conflicts_json = _model_dump_json(conflicts, indent=2)
    registry_json = _model_dump_json(registry, indent=2)
    prompt = get_fix_suggestion_prompt(conflicts_json, registry_json)
    response = _call_llm(prompt, llm_client=llm_client)

    if response:
        try:
            parsed = _parse_json_from_text(response)
            if isinstance(parsed, dict):
                raw_suggestions = parsed.get("suggestions", [])
                suggestions = [
                    FixSuggestion(**item) for item in raw_suggestions if isinstance(item, dict)
                ]
                return FixSuggestions(suggestions=suggestions)
        except Exception:
            pass

    fallback_suggestions = []
    for conflict in conflicts.conflicts:
        if conflict.rule_id == "R01":
            suggestion = "考虑在第1集或更早场景中增加该资产的铺垫镜头/台词，或明确这是首次出现。"
            priority = "low"
        elif conflict.rule_id == "R02":
            suggestion = "延长伤势恢复时间，增加治疗过程，或降低前文伤势严重程度。"
            priority = "high"
        elif conflict.rule_id == "R03":
            suggestion = "确认这些同类型资产是否确为不同实物；若是同一资产，请统一名称或补充归一化别名；若是不同资产，请增加换装/更换/同时持有的交代。"
            priority = "high"
        elif conflict.rule_id == "R04":
            suggestion = "检查闪回与现代线时间顺序；如闪回发生在资产消失前，请明确时间标签；如发生在消失后，请修正资产状态。"
            priority = "high"
        else:
            suggestion = "根据冲突描述核对原文证据，统一设定或补充过渡说明。"
            priority = "medium"

        fallback_suggestions.append(
            FixSuggestion(
                conflict_rule_id=conflict.rule_id,
                conflict_description=conflict.description,
                suggestion=suggestion,
                priority=priority,
            )
        )

    return FixSuggestions(suggestions=fallback_suggestions)


def merge_conflict_reports(*reports: ConflictReport) -> ConflictReport:
    conflicts: List[Conflict] = []
    for report in reports:
        if report and report.conflicts:
            conflicts.extend(report.conflicts)
    return ConflictReport.from_conflicts(conflicts)


def full_check(
    script_text: str,
    script_language: str = "zh",
    llm_client: Any = None,
    enable_semantic_review: bool = False,
) -> ContinuityReport:
    """
    Full continuity checking pipeline.

    Pipeline:
    1. extract_assets(script_text)
    2. normalize_assets(assets)
    3. check_continuity(assets)
    4. optional semantic_review(assets)
    5. generate_fix_suggestions(...)
    """
    assets = extract_assets(script_text, llm_client=llm_client)

    # 新增：资产归一化。必须在规则检查前执行，否则 R03/R04 等规则会误报或漏报。
    assets = normalize_assets(assets, llm_client=llm_client)

    rule_report = check_continuity(assets)

    if enable_semantic_review:
        semantic_report = semantic_review(assets, llm_client=llm_client)
        conflict_report = merge_conflict_reports(rule_report, semantic_report)
    else:
        conflict_report = rule_report

    fix_suggestions = generate_fix_suggestions(
        conflicts=conflict_report,
        registry=assets,
        llm_client=llm_client,
    )

    summary = (
        f"共发现 {conflict_report.total_count} 个连续性问题："
        f"P0={conflict_report.p0_count}, "
        f"P1={conflict_report.p1_count}, "
        f"P2={conflict_report.p2_count}, "
        f"P3={conflict_report.p3_count}。"
    )

    return ContinuityReport(
        script_language=script_language,
        asset_registry=assets,
        conflicts=conflict_report,
        fix_suggestions=fix_suggestions,
        summary=summary,
    )


__all__ = [
    "extract_assets",
    "normalize_assets",
    "check_continuity",
    "semantic_review",
    "generate_fix_suggestions",
    "merge_conflict_reports",
    "full_check",
]