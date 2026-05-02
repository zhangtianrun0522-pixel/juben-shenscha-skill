import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .models import (
    AssetEntry,
    AssetIdentity,
    AssetLifecycle,
    AssetRegistry,
    AssetStateEvent,
    AssetStateInterval,
    CharacterSetting,
    Conflict,
    ConflictReport,
    ContinuityReport,
    FixSuggestion,
    FixSuggestions,
    StateChange,
    StoryTimeAnchor,
    TimeLayer,
    TimeRelation,
    TimelineAnchor,
)
from .prompts import (
    get_extraction_prompt,
    get_fix_suggestion_prompt,
    get_normalization_prompt,
    get_semantic_review_prompt,
    get_timeline_extraction_prompt,
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


def _model_validate_compat(model_cls, data):
    if data is None:
        return None
    if isinstance(data, model_cls):
        return data
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def _model_dump_compat(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


def _parse_json_from_text(text: str) -> Any:
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


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}

    return {}


def _call_llm(prompt: str, llm_client: Any = None, model: Optional[str] = None) -> Optional[str]:
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


def _call_llm_for_text(llm_client, prompt: str) -> Optional[str]:
    if llm_client is None:
        return None

    try:
        if callable(llm_client):
            result = llm_client(prompt)
        elif hasattr(llm_client, "complete"):
            result = llm_client.complete(prompt)
        elif hasattr(llm_client, "generate"):
            result = llm_client.generate(prompt)
        elif hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
            return _call_llm(prompt, llm_client=llm_client)
        else:
            return None

        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            if "content" in result:
                return result["content"]
            if "text" in result:
                return result["text"]
            if "message" in result and isinstance(result["message"], dict):
                return result["message"].get("content")

        if hasattr(result, "content"):
            return result.content

        if hasattr(result, "text"):
            return result.text

        if hasattr(result, "choices"):
            choices = result.choices
            if choices:
                first = choices[0]
                if hasattr(first, "message") and hasattr(first.message, "content"):
                    return first.message.content
                if hasattr(first, "text"):
                    return first.text

    except Exception:
        return None

    return None


def _scene_key(episode: int, scene: int) -> Tuple[int, int]:
    return int(episode), int(scene)


def _get_story_anchor(
    registry: AssetRegistry, episode: int, scene: int
) -> Optional[StoryTimeAnchor]:
    for anchor in registry.story_anchors:
        if anchor.episode == episode and anchor.scene == scene:
            return anchor
    return None


def _get_time_layer(registry: AssetRegistry, layer_id: str) -> Optional[TimeLayer]:
    for layer in registry.time_layers:
        if layer.layer_id == layer_id:
            return layer
    return None


def _scene_affects_canonical_state(
    registry: AssetRegistry, episode: int, scene: int
) -> bool:
    anchor = _get_story_anchor(registry, episode, scene)
    if not anchor:
        return True

    layer = _get_time_layer(registry, anchor.layer_id)
    if not layer:
        return True

    return bool(layer.affects_canonical_state)


def _story_day_for_scene(
    registry: AssetRegistry, episode: int, scene: int
) -> Optional[float]:
    anchor = _get_story_anchor(registry, episode, scene)
    if not anchor:
        return None

    if anchor.story_offset_days is not None:
        return float(anchor.story_offset_days)

    if anchor.story_offset_min is not None and anchor.story_offset_max is not None:
        return (float(anchor.story_offset_min) + float(anchor.story_offset_max)) / 2.0

    if anchor.story_offset_min is not None:
        return float(anchor.story_offset_min)

    if anchor.story_offset_max is not None:
        return float(anchor.story_offset_max)

    return None


def _asset_registry_from_dict(data) -> AssetRegistry:
    if isinstance(data, AssetRegistry):
        return data

    if data is None:
        return AssetRegistry()

    if not isinstance(data, dict):
        return AssetRegistry()

    def parse_list(key: str, model_cls):
        values = data.get(key, [])
        if values is None:
            return []

        result = []
        for item in values:
            try:
                parsed = _model_validate_compat(model_cls, item)
                if parsed is not None:
                    result.append(parsed)
            except Exception:
                continue
        return result

    lifecycles_raw = data.get("lifecycles", []) or []
    lifecycles: List[AssetLifecycle] = []

    for item in lifecycles_raw:
        if isinstance(item, AssetLifecycle):
            lifecycles.append(item)
            continue

        if not isinstance(item, dict):
            continue

        try:
            lifecycle_data = dict(item)

            lifecycle_data["events"] = [
                _model_validate_compat(AssetStateEvent, event)
                for event in lifecycle_data.get("events", []) or []
                if isinstance(event, (dict, AssetStateEvent))
            ]

            lifecycle_data["intervals"] = [
                _model_validate_compat(AssetStateInterval, interval)
                for interval in lifecycle_data.get("intervals", []) or []
                if isinstance(interval, (dict, AssetStateInterval))
            ]

            lifecycles.append(_model_validate_compat(AssetLifecycle, lifecycle_data))
        except Exception:
            continue

    return AssetRegistry(
        assets=parse_list("assets", AssetEntry),
        state_changes=parse_list("state_changes", StateChange),
        timeline=parse_list("timeline", TimelineAnchor),
        character_settings=parse_list("character_settings", CharacterSetting),
        identities=parse_list("identities", AssetIdentity),
        story_anchors=parse_list("story_anchors", StoryTimeAnchor),
        time_layers=parse_list("time_layers", TimeLayer),
        time_relations=parse_list("time_relations", TimeRelation),
        lifecycles=lifecycles,
    )


def _split_script_into_chunks(script_text: str, max_chars: int = 15000) -> List[str]:
    episode_splits = re.split(r'(第[零一二三四五六七八九十百千\d]+集)', script_text)
    episodes: List[str] = []
    temp = ""
    for part in episode_splits:
        if re.match(r'第[零一二三四五六七八九十百千\d]+集', part):
            if temp:
                episodes.append(temp)
            temp = part
        else:
            temp += part
    if temp:
        episodes.append(temp)

    scene_chunks: List[str] = []
    for ep in episodes:
        if len(ep) <= max_chars:
            scene_chunks.append(ep)
            continue
        scene_splits = re.split(r'(内景|外景)', ep)
        temp = ""
        for part in scene_splits:
            if part in ('内景', '外景'):
                if temp:
                    scene_chunks.append(temp)
                temp = part
            else:
                temp += part
        if temp:
            scene_chunks.append(temp)

    truncated: List[str] = []
    for chunk in scene_chunks:
        if len(chunk) <= max_chars:
            truncated.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                truncated.append(chunk[i:i + max_chars])

    final: List[str] = []
    for i, chunk in enumerate(truncated):
        if i == 0:
            final.append(chunk)
        else:
            prefix = final[-1][-500:]
            final.append(prefix + chunk)
    return final


def _merge_registries(*registries: AssetRegistry) -> AssetRegistry:
    def _g(obj: Any, key: str) -> Any:
        return getattr(obj, key, None)

    assets_map: dict = {}
    for reg in registries:
        for item in (_g(reg, 'assets') or []):
            k = (_g(item, 'character'), _g(item, 'asset_type'), _g(item, 'asset_name'),
                 _g(item, 'episode'), _g(item, 'scene'))
            if k not in assets_map:
                assets_map[k] = item

    sc_map: dict = {}
    for reg in registries:
        for item in (_g(reg, 'state_changes') or []):
            k = (_g(item, 'character'), _g(item, 'asset_name'), _g(item, 'episode'),
                 _g(item, 'scene'), _g(item, 'new_status'))
            if k not in sc_map:
                sc_map[k] = item

    tl_map: dict = {}
    for reg in registries:
        for item in (_g(reg, 'timeline') or []):
            k = (_g(item, 'episode'), _g(item, 'scene'))
            if k not in tl_map:
                tl_map[k] = item

    cs_map: dict = {}
    for reg in registries:
        for item in (_g(reg, 'character_settings') or []):
            k = (_g(item, 'character'), _g(item, 'setting_type'), _g(item, 'content'))
            if k not in cs_map:
                cs_map[k] = item

    merged = AssetRegistry()
    merged.assets = list(assets_map.values())
    merged.state_changes = list(sc_map.values())
    merged.timeline = list(tl_map.values())
    merged.character_settings = list(cs_map.values())
    merged.identities = []
    merged.story_anchors = []
    merged.time_layers = []
    merged.time_relations = []
    merged.lifecycles = []
    return merged


def extract_assets(script_text: str, llm_client: Any = None, chunk_size: int = 15000) -> AssetRegistry:
    if chunk_size > 0 and len(script_text) > chunk_size:
        chunks = _split_script_into_chunks(script_text, chunk_size)
        regs: List[AssetRegistry] = []
        for chunk in chunks:
            prompt = get_extraction_prompt(chunk)
            response = _call_llm(prompt, llm_client=llm_client)
            data = _parse_json_from_text(response) if response else None
            if isinstance(data, dict):
                regs.append(_asset_registry_from_dict(data))
        return _merge_registries(*regs) if regs else AssetRegistry()

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


def build_story_timeline(registry: AssetRegistry, llm_client: Any = None) -> AssetRegistry:
    """
    构建故事内时间轴。

    1. 优先调用 LLM，从旧 timeline 与 state_changes 的时间备注中提取结构化故事时间。
    2. LLM 不可用或解析失败时，从旧 TimelineAnchor fallback。
    3. 更新 registry.story_anchors / registry.time_layers / registry.time_relations。
    """
    if registry is None:
        registry = AssetRegistry()

    context_lines: List[str] = []

    if registry.timeline:
        context_lines.append("已有 TimelineAnchor：")
        for item in registry.timeline:
            context_lines.append(
                f"- episode={item.episode}, scene={item.scene}, "
                f"time_label={item.time_label}, is_flashback={item.is_flashback}"
            )

    if registry.state_changes:
        context_lines.append("\n状态变化中的故事内时间备注：")
        for item in registry.state_changes:
            if item.in_story_time_note:
                context_lines.append(
                    f"- episode={item.episode}, scene={item.scene}, "
                    f"character={item.character}, asset_type={item.asset_type}, "
                    f"change_from={item.change_from}, change_to={item.change_to}, "
                    f"in_story_time_note={item.in_story_time_note}"
                )

    context = "\n".join(context_lines).strip()

    used_llm = False

    if llm_client is not None and context:
        prompt = get_timeline_extraction_prompt(context)
        raw = _call_llm_for_text(llm_client, prompt)
        parsed = _extract_json_object(raw or "")

        if parsed:
            try:
                story_anchors = [
                    _model_validate_compat(StoryTimeAnchor, x)
                    for x in parsed.get("story_anchors", [])
                    if isinstance(x, dict)
                ]
                time_layers = [
                    _model_validate_compat(TimeLayer, x)
                    for x in parsed.get("time_layers", [])
                    if isinstance(x, dict)
                ]
                time_relations = [
                    _model_validate_compat(TimeRelation, x)
                    for x in parsed.get("time_relations", [])
                    if isinstance(x, dict)
                ]

                registry.story_anchors = [x for x in story_anchors if x is not None]
                registry.time_layers = [x for x in time_layers if x is not None]
                registry.time_relations = [x for x in time_relations if x is not None]

                used_llm = bool(
                    registry.story_anchors or registry.time_layers or registry.time_relations
                )
            except Exception:
                used_llm = False

    if not used_llm:
        layers_by_id: Dict[str, TimeLayer] = {
            "main": TimeLayer(
                layer_id="main",
                name="主线现实",
                layer_type="main",
                parent_layer_id=None,
                is_canonical=True,
                affects_canonical_state=True,
                confidence=1.0,
            )
        }

        anchors: List[StoryTimeAnchor] = []

        for item in registry.timeline:
            if item.is_flashback:
                layer_id = "flashback_main"
                temporal_mode = "flashback"
                if layer_id not in layers_by_id:
                    layers_by_id[layer_id] = TimeLayer(
                        layer_id=layer_id,
                        name="闪回",
                        layer_type="flashback",
                        parent_layer_id="main",
                        is_canonical=True,
                        affects_canonical_state=True,
                        confidence=0.8,
                    )
            else:
                layer_id = "main"
                temporal_mode = "present"

            anchors.append(
                StoryTimeAnchor(
                    episode=item.episode,
                    scene=item.scene,
                    layer_id=layer_id,
                    temporal_mode=temporal_mode,
                    relative_time_label=item.time_label,
                    story_offset_days=None,
                    story_offset_min=None,
                    story_offset_max=None,
                    is_time_jump=bool(item.is_flashback),
                    confidence=0.6,
                    evidence=item.time_label or "",
                )
            )

        registry.story_anchors = anchors
        registry.time_layers = list(layers_by_id.values())
        registry.time_relations = []

    if not any(layer.layer_id == "main" for layer in registry.time_layers):
        registry.time_layers.insert(
            0,
            TimeLayer(
                layer_id="main",
                name="主线现实",
                layer_type="main",
                parent_layer_id=None,
                is_canonical=True,
                affects_canonical_state=True,
                confidence=1.0,
            ),
        )

    return registry


def build_asset_lifecycles(registry: AssetRegistry) -> AssetRegistry:
    """
    根据 identities / assets / state_changes 纯代码构建资产生命周期。不调用 LLM。
    """
    if registry is None:
        registry = AssetRegistry()

    identity_by_id: Dict[str, AssetIdentity] = {
        item.asset_id: item for item in registry.identities if item.asset_id
    }

    lifecycle_by_id: Dict[str, AssetLifecycle] = {}

    for identity in registry.identities:
        if not identity.asset_id:
            continue

        lifecycle_by_id[identity.asset_id] = AssetLifecycle(
            asset_id=identity.asset_id,
            canonical_name=identity.canonical_name,
            asset_type=identity.asset_type,
            owner_character=identity.character,
            first_episode=identity.first_episode,
            first_scene=identity.first_scene,
            last_episode=None,
            last_scene=None,
            lifecycle_status="active",
            events=[],
            intervals=[],
        )

    def resolve_asset_id_from_entry(entry: AssetEntry) -> Optional[str]:
        if entry.asset_id:
            return entry.asset_id

        for identity in registry.identities:
            names = [identity.canonical_name] + list(identity.aliases or [])
            if (
                identity.character == entry.character
                and identity.asset_type == entry.asset_type
                and entry.asset_name in names
            ):
                return identity.asset_id

        for identity in registry.identities:
            if identity.character == entry.character and identity.asset_type == entry.asset_type:
                return identity.asset_id

        return None

    def resolve_asset_id_from_state_change(change: StateChange) -> Optional[str]:
        for identity in registry.identities:
            if identity.character == change.character and identity.asset_type == change.asset_type:
                return identity.asset_id
        return None

    def scene_layer_id(episode: int, scene: int) -> str:
        anchor = _get_story_anchor(registry, episode, scene)
        return anchor.layer_id if anchor else "main"

    def scene_affects(episode: int, scene: int) -> bool:
        return _scene_affects_canonical_state(registry, episode, scene)

    for entry in registry.assets:
        asset_id = resolve_asset_id_from_entry(entry)
        if not asset_id:
            continue

        if asset_id not in lifecycle_by_id:
            lifecycle_by_id[asset_id] = AssetLifecycle(
                asset_id=asset_id,
                canonical_name=entry.asset_name,
                asset_type=entry.asset_type,
                owner_character=entry.character,
                first_episode=entry.episode,
                first_scene=entry.scene,
                lifecycle_status="active",
            )

        status = (entry.status or "exists").strip().lower()

        if status in {"exists", "appear", "appears", "present", "shown", "出现", "存在"}:
            event_type = "appear"
            state_dimension = "holder"
            from_value = None
            to_value = entry.character
        elif status in {
            "disappear",
            "disappears",
            "missing",
            "lost",
            "absent",
            "消失",
            "丢失",
            "不见",
        }:
            event_type = "disappear"
            state_dimension = "holder"
            from_value = entry.character
            to_value = None
        elif status in {"destroyed", "broken", "毁坏", "损毁", "破坏"}:
            event_type = "destroyed"
            state_dimension = "physical_condition"
            from_value = None
            to_value = "destroyed"
        else:
            event_type = "appear"
            state_dimension = "holder"
            from_value = None
            to_value = entry.character

        lifecycle_by_id[asset_id].events.append(
            AssetStateEvent(
                asset_id=asset_id,
                episode=entry.episode,
                scene=entry.scene,
                layer_id=scene_layer_id(entry.episode, entry.scene),
                event_type=event_type,
                state_dimension=state_dimension,
                from_value=from_value,
                to_value=to_value,
                affects_canonical_state=scene_affects(entry.episode, entry.scene),
                is_explicit=True,
                confidence=0.9,
                evidence=entry.raw_text,
            )
        )

    for change in registry.state_changes:
        asset_id = resolve_asset_id_from_state_change(change)
        if not asset_id:
            continue

        identity = identity_by_id.get(asset_id)

        if asset_id not in lifecycle_by_id:
            lifecycle_by_id[asset_id] = AssetLifecycle(
                asset_id=asset_id,
                canonical_name=identity.canonical_name if identity else change.asset_type,
                asset_type=identity.asset_type if identity else change.asset_type,
                owner_character=identity.character if identity else change.character,
                first_episode=identity.first_episode if identity else change.episode,
                first_scene=identity.first_scene if identity else change.scene,
                lifecycle_status="active",
            )

        text = f"{change.change_from} {change.change_to}".lower()

        if any(k in text for k in ["destroy", "destroyed", "毁", "损毁", "报废", "粉碎"]):
            event_type = "destroyed"
        elif any(k in text for k in ["repair", "repaired", "修好", "修复", "恢复"]):
            event_type = "repaired"
        elif any(k in text for k in ["damage", "damaged", "broken", "受损", "损坏", "破裂"]):
            event_type = "damaged"
        elif any(k in text for k in ["transfer", "transferred", "交给", "转交", "移交", "给了"]):
            event_type = "transfer"
        elif any(k in text for k in ["lost", "missing", "丢失", "不见"]):
            event_type = "disappear"
        else:
            event_type = "damaged"

        if event_type == "transfer":
            state_dimension = "holder"
        else:
            state_dimension = "physical_condition"

        lifecycle_by_id[asset_id].events.append(
            AssetStateEvent(
                asset_id=asset_id,
                episode=change.episode,
                scene=change.scene,
                layer_id=scene_layer_id(change.episode, change.scene),
                event_type=event_type,
                state_dimension=state_dimension,
                from_value=change.change_from,
                to_value=change.change_to,
                affects_canonical_state=scene_affects(change.episode, change.scene),
                is_explicit=True,
                confidence=0.85,
                evidence=change.in_story_time_note
                or f"{change.change_from}->{change.change_to}",
            )
        )

    for lifecycle in lifecycle_by_id.values():
        lifecycle.events.sort(key=lambda e: (e.episode, e.scene))

        canonical_events = [e for e in lifecycle.events if e.affects_canonical_state]

        if canonical_events:
            lifecycle.last_episode = canonical_events[-1].episode
            lifecycle.last_scene = canonical_events[-1].scene

        intervals: List[AssetStateInterval] = []

        events_by_dimension: Dict[str, List[AssetStateEvent]] = {}
        for event in canonical_events:
            events_by_dimension.setdefault(event.state_dimension, []).append(event)

        for dimension, events in events_by_dimension.items():
            events.sort(key=lambda e: (e.episode, e.scene))

            for idx, event in enumerate(events):
                if event.to_value is None:
                    continue

                next_event = events[idx + 1] if idx + 1 < len(events) else None

                intervals.append(
                    AssetStateInterval(
                        asset_id=lifecycle.asset_id,
                        state_dimension=dimension,
                        value=event.to_value,
                        start_episode=event.episode,
                        start_scene=event.scene,
                        end_episode=next_event.episode if next_event else None,
                        end_scene=next_event.scene if next_event else None,
                        layer_id=event.layer_id,
                        is_open_ended=next_event is None,
                        confidence=event.confidence,
                    )
                )

        lifecycle.intervals = intervals

        if canonical_events:
            last_event = canonical_events[-1]
            if last_event.event_type == "destroyed":
                lifecycle.lifecycle_status = "destroyed"
            elif last_event.event_type == "disappear":
                lifecycle.lifecycle_status = "lost"
            elif last_event.event_type == "transfer":
                lifecycle.lifecycle_status = "transferred"
            else:
                lifecycle.lifecycle_status = "active"
        else:
            lifecycle.lifecycle_status = "unknown"

    registry.lifecycles = list(lifecycle_by_id.values())
    registry.lifecycles.sort(key=lambda x: (x.first_episode, x.first_scene, x.asset_id))

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
    anchor = _get_story_anchor(registry, episode, scene)
    if anchor:
        return anchor.temporal_mode == "flashback"

    for old_anchor in registry.timeline:
        if old_anchor.episode == episode and old_anchor.scene == scene:
            return bool(old_anchor.is_flashback)

    return False


def _is_injury_state_text(text: str) -> bool:
    text = text or ""
    lower = text.lower()
    return any(
        keyword in lower
        for keyword in [
            "injured",
            "injury",
            "hurt",
            "wounded",
            "gunshot",
            "shot",
            "fracture",
            "broken bone",
            "受伤",
            "负伤",
            "中枪",
            "枪伤",
            "骨折",
            "重伤",
            "轻伤",
            "擦伤",
            "昏迷",
        ]
    )


def _is_recovered_state_text(text: str) -> bool:
    text = text or ""
    lower = text.lower()
    return any(
        keyword in lower
        for keyword in [
            "recover",
            "recovered",
            "healed",
            "cured",
            "fine",
            "恢复",
            "痊愈",
            "康复",
            "好了",
            "无碍",
            "行动自如",
        ]
    )


def _injury_required_days(injury_text: str) -> float:
    text = injury_text or ""
    lower = text.lower()

    if any(
        k in lower
        for k in [
            "骨折",
            "枪伤",
            "中枪",
            "重伤",
            "fracture",
            "broken bone",
            "gunshot",
            "shot",
            "serious injury",
            "severe",
        ]
    ):
        return 14.0

    if any(k in lower for k in ["轻伤", "擦伤", "划伤", "minor", "scratch", "bruise"]):
        return 0.5

    return 3.0


def _check_r02_injury_recovery(registry: AssetRegistry) -> List[Conflict]:
    conflicts: List[Conflict] = []

    injuries: List[StateChange] = []
    recoveries: List[StateChange] = []

    for change in registry.state_changes:
        combined = f"{change.change_from} {change.change_to} {change.in_story_time_note or ''}"

        if _is_injury_state_text(combined) and not _is_recovered_state_text(change.change_to):
            injuries.append(change)

        if _is_recovered_state_text(combined):
            recoveries.append(change)

    for injury in injuries:
        for heal in recoveries:
            if heal.character != injury.character:
                continue

            if heal.asset_type != injury.asset_type:
                continue

            if (heal.episode, heal.scene) <= (injury.episode, injury.scene):
                continue

            required_days = _injury_required_days(
                f"{injury.change_from} {injury.change_to} {injury.in_story_time_note or ''}"
            )

            injury_day = _story_day_for_scene(registry, injury.episode, injury.scene)
            heal_day = _story_day_for_scene(registry, heal.episode, heal.scene)

            if injury_day is not None and heal_day is not None:
                actual_days = heal_day - injury_day

                if actual_days < required_days:
                    conflicts.append(
                        Conflict(
                            rule_id="R02",
                            severity="P1",
                            description=(
                                f"{injury.character} 的伤情恢复过快："
                                f"故事内约 {actual_days:.2f} 天从「{injury.change_to}」"
                                f"恢复到「{heal.change_to}」，"
                                f"该伤情建议至少 {required_days:g} 天。"
                            ),
                            character=injury.character,
                            episode_a=injury.episode,
                            scene_a=injury.scene,
                            episode_b=heal.episode,
                            scene_b=heal.scene,
                            raw_evidence=(
                                f"injury={injury.change_from}->{injury.change_to}; "
                                f"heal={heal.change_from}->{heal.change_to}; "
                                f"story_days={actual_days:.2f}; required_days={required_days:g}"
                            ),
                        )
                    )
            else:
                if heal.episode - injury.episode <= 1:
                    conflicts.append(
                        Conflict(
                            rule_id="R02",
                            severity="P1",
                            description=(
                                f"{injury.character} 的伤情恢复可能过快："
                                f"从 E{injury.episode}S{injury.scene} 的「{injury.change_to}」"
                                f"到 E{heal.episode}S{heal.scene} 的「{heal.change_to}」"
                                f"间隔不超过 1 集。"
                            ),
                            character=injury.character,
                            episode_a=injury.episode,
                            scene_a=injury.scene,
                            episode_b=heal.episode,
                            scene_b=heal.scene,
                            raw_evidence=(
                                f"fallback_episode_gap={heal.episode - injury.episode}; "
                                f"injury={injury.change_from}->{injury.change_to}; "
                                f"heal={heal.change_from}->{heal.change_to}"
                            ),
                        )
                    )

    return conflicts


def _check_r04_asset_presence(registry: AssetRegistry) -> List[Conflict]:
    conflicts: List[Conflict] = []

    assets = sorted(registry.assets, key=lambda x: (x.episode, x.scene))

    by_asset_id: Dict[str, List[AssetEntry]] = {}

    for asset in assets:
        if not _scene_affects_canonical_state(registry, asset.episode, asset.scene):
            continue

        asset_id = asset.asset_id
        if not asset_id:
            asset_id = f"{asset.character}|{asset.asset_type}|{asset.asset_name}"

        by_asset_id.setdefault(asset_id, []).append(asset)

    disappear_statuses = {
        "disappear",
        "disappears",
        "missing",
        "lost",
        "absent",
        "destroyed",
        "消失",
        "丢失",
        "不见",
        "毁坏",
        "损毁",
    }

    appear_statuses = {
        "exists",
        "appear",
        "appears",
        "present",
        "shown",
        "出现",
        "存在",
    }

    for asset_id, entries in by_asset_id.items():
        last_disappear: Optional[AssetEntry] = None

        for entry in entries:
            status = (entry.status or "exists").strip().lower()

            if status in disappear_statuses:
                last_disappear = entry
                continue

            if last_disappear is not None and status in appear_statuses:
                conflicts.append(
                    Conflict(
                        rule_id="R04",
                        severity="P1",
                        description=(
                            f"资产「{entry.asset_name}」此前在 "
                            f"E{last_disappear.episode}S{last_disappear.scene} "
                            f"状态为「{last_disappear.status}」，"
                            f"但在 E{entry.episode}S{entry.scene} 又以「{entry.status}」出现，"
                            f"缺少找回、修复或重新获得的过渡。"
                        ),
                        character=entry.character,
                        episode_a=last_disappear.episode,
                        scene_a=last_disappear.scene,
                        episode_b=entry.episode,
                        scene_b=entry.scene,
                        raw_evidence=f"{last_disappear.raw_text}\n{entry.raw_text}",
                    )
                )
                last_disappear = None

    return conflicts


def _r_scene_key(episode, scene):
    return (episode if episode is not None else -1, scene if scene is not None else -1)


def _r_interval_end_key(interval):
    if interval.end_episode is None or interval.end_scene is None or interval.is_open_ended:
        return (10**9, 10**9)
    return (interval.end_episode, interval.end_scene)


def _r_intervals_overlap(a, b):
    a_start = (a.start_episode, a.start_scene)
    a_end = _r_interval_end_key(a)
    b_start = (b.start_episode, b.start_scene)
    b_end = _r_interval_end_key(b)
    return a_start < b_end and b_start < a_end


def _r_later(ep_a, sc_a, ep_b, sc_b):
    return (ep_a, sc_a) > (ep_b, sc_b)


def _r_asset_display_name(lifecycle) -> str:
    return lifecycle.canonical_name or lifecycle.asset_id


def _r_asset_owner(lifecycle) -> str:
    return lifecycle.owner_character or "未知"


def _r_asset_type(lifecycle) -> str:
    return lifecycle.asset_type or "未知"


def _r_has_event_between(lifecycle, ep_start, sc_start, ep_end, sc_end, event_types) -> bool:
    for event in lifecycle.events:
        if event.event_type in event_types:
            key = (event.episode, event.scene)
            if (ep_start, sc_start) < key <= (ep_end, sc_end):
                return True
    return False


def _check_r05_destroyed_reappear(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []
    emitted = set()

    entries_by_asset_id = {}
    for asset in registry.assets:
        if not asset.asset_id or not _is_positive_status(asset.status):
            continue
        if not _scene_affects_canonical_state(registry, asset.episode, asset.scene):
            continue
        entries_by_asset_id.setdefault(asset.asset_id, []).append(asset)

    for lifecycle in registry.lifecycles:
        if lifecycle.lifecycle_status != 'destroyed':
            continue

        destroyed_events = [
            event for event in lifecycle.events
            if event.event_type == 'destroyed' and event.affects_canonical_state
        ]

        if not destroyed_events and lifecycle.last_episode is not None and lifecycle.last_scene is not None:
            destroy_points = [(lifecycle.last_episode, lifecycle.last_scene, None)]
        else:
            destroy_points = [
                (event.episode, event.scene, event)
                for event in destroyed_events
            ]

        repaired_events = [
            event for event in lifecycle.events
            if event.event_type == 'repaired' and event.affects_canonical_state
        ]

        for destroy_episode, destroy_scene, destroy_event in destroy_points:
            destroy_key = (destroy_episode, destroy_scene)
            for asset in entries_by_asset_id.get(lifecycle.asset_id, []):
                appear_key = (asset.episode, asset.scene)
                if appear_key <= destroy_key:
                    continue
                has_repaired = any(destroy_key < (event.episode, event.scene) < appear_key for event in repaired_events)
                if has_repaired:
                    continue

                dedupe_key = (lifecycle.asset_id, destroy_episode, destroy_scene, asset.episode, asset.scene)
                if dedupe_key in emitted:
                    continue
                emitted.add(dedupe_key)

                conflicts.append(Conflict(
                    rule_id='R05',
                    severity='P0',
                    description=f"资产“{lifecycle.canonical_name}”已标记为 destroyed，但在后续场次以正面状态再次出现，且缺少 repaired 事件。",
                    character=asset.character or lifecycle.owner_character,
                    episode_a=destroy_episode,
                    scene_a=destroy_scene,
                    episode_b=asset.episode,
                    scene_b=asset.scene,
                    raw_evidence=asset.raw_text,
                ))

    return conflicts


def _check_r06_overlapping_holder_intervals(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []
    intervals_by_asset_id = {}

    for lifecycle in registry.lifecycles:
        for interval in lifecycle.intervals:
            if interval.state_dimension != 'holder':
                continue
            if interval.asset_id:
                intervals_by_asset_id.setdefault(interval.asset_id, []).append(interval)

    for asset_id, intervals in intervals_by_asset_id.items():
        sorted_intervals = sorted(intervals, key=lambda x: (x.start_episode, x.start_scene, _r_interval_end_key(x)))
        for i in range(len(sorted_intervals)):
            for j in range(i + 1, len(sorted_intervals)):
                a = sorted_intervals[i]
                b = sorted_intervals[j]
                if (b.start_episode, b.start_scene) > _r_interval_end_key(a):
                    break
                if a.value == b.value:
                    continue
                if a.layer_id != b.layer_id:
                    continue
                if not _r_intervals_overlap(a, b):
                    continue

                conflicts.append(Conflict(
                    rule_id='R06',
                    severity='P1',
                    description=f"同一资产“{_r_asset_display_name(registry, asset_id)}”的 holder 区间重叠：同时由“{a.value}”与“{b.value}”持有。",
                    character=_r_asset_owner(registry, asset_id),
                    episode_a=a.start_episode,
                    scene_a=a.start_scene,
                    episode_b=b.start_episode,
                    scene_b=b.start_scene,
                    raw_evidence=f"{asset_id}: holder={a.value} overlaps holder={b.value}",
                ))

    return conflicts


def _check_r07_setting_violated_by_asset_description(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []

    contradiction_patterns = [
        (
            ('不会开车', '不能开车', '不会驾驶', '不能驾驶', '不会驾车', '不能驾车', '不懂开车', '不懂驾驶'),
            ('驾车', '开车', '驾驶汽车', '驾驶轿车', '开着车', '开汽车', '开轿车'),
            '不会开车/驾驶'
        ),
        (
            ('不会游泳', '不能游泳', '不懂游泳'),
            ('游泳', '潜水', '下水游'),
            '不会游泳'
        ),
        (
            ('失明', '盲人', '看不见'),
            ('看见', '看到', '目睹', '凝视', '盯着看'),
            '失明/看不见'
        ),
        (
            ('失聪', '聋', '听不见'),
            ('听见', '听到', '倾听', '听出'),
            '失聪/听不见'
        ),
        (
            ('不会说话', '哑巴', '失语'),
            ('说道', '说话', '喊道', '大喊', '低语', '耳语'),
            '不会说话/失语'
        ),
    ]

    for setting in registry.character_settings:
        setting_text = f"{setting.setting_type} {setting.content} {setting.raw_text}"
        matched_rules = []
        for negative_keywords, positive_keywords, label in contradiction_patterns:
            if any(keyword in setting_text for keyword in negative_keywords):
                matched_rules.append((positive_keywords, label))

        if not matched_rules:
            continue

        for asset in registry.assets:
            if asset.character != setting.character:
                continue
            if (asset.episode, asset.scene) <= (setting.episode, setting.scene):
                continue

            asset_text = f"{asset.asset_type} {asset.asset_name} {asset.raw_text}"
            for positive_keywords, label in matched_rules:
                if not any(keyword in asset_text for keyword in positive_keywords):
                    continue

                conflicts.append(Conflict(
                    rule_id='R07',
                    severity='P1',
                    description=f"角色“{setting.character}”的硬性设定“{label}”被后续资产/动作描述违反。",
                    character=setting.character,
                    episode_a=setting.episode,
                    scene_a=setting.scene,
                    episode_b=asset.episode,
                    scene_b=asset.scene,
                    raw_evidence=f"设定：{setting.raw_text}；后续描述：{asset.raw_text}",
                ))

    return conflicts


def _check_r08_asset_before_first_story_time(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []
    first_scene_by_asset_id = {}

    for identity in registry.identities:
        first_scene_by_asset_id[identity.asset_id] = (identity.first_episode, identity.first_scene, identity.canonical_name)

    for lifecycle in registry.lifecycles:
        first_scene_by_asset_id.setdefault(
            lifecycle.asset_id,
            (lifecycle.first_episode, lifecycle.first_scene, lifecycle.canonical_name)
        )

    emitted = set()

    for asset in registry.assets:
        if not asset.asset_id:
            continue
        if asset.asset_id not in first_scene_by_asset_id:
            continue
        if not _is_positive_status(asset.status):
            continue
        if _timeline_is_flashback(registry, asset.episode, asset.scene):
            continue

        anchor = _get_story_anchor(registry, asset.episode, asset.scene)
        if anchor and anchor.temporal_mode in ('flashback', 'dream', 'hallucination', 'parallel', 'flashforward'):
            if anchor.temporal_mode == 'flashback':
                continue

        current_day = _story_day_for_scene(registry, asset.episode, asset.scene)
        first_episode, first_scene, canonical_name = first_scene_by_asset_id[asset.asset_id]
        first_day = _story_day_for_scene(registry, first_episode, first_scene)

        if current_day is None or first_day is None:
            continue
        if current_day >= first_day:
            continue

        dedupe_key = (asset.asset_id, asset.episode, asset.scene)
        if dedupe_key in emitted:
            continue
        emitted.add(dedupe_key)

        conflicts.append(Conflict(
            rule_id='R08',
            severity='P1',
            description=f"资产“{canonical_name}”出现在早于其首次登记场次故事时间的非闪回场景中。",
            character=asset.character,
            episode_a=first_episode,
            scene_a=first_scene,
            episode_b=asset.episode,
            scene_b=asset.scene,
            raw_evidence=asset.raw_text,
        ))

    return conflicts


def _check_r09_mutually_exclusive_color_assets(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []

    color_pairs = [
        (('黑', 'black'), ('白', 'white')),
        (('红', 'red'), ('绿', 'green')),
        (('蓝', 'blue'), ('黄', 'yellow')),
        (('金', 'gold', 'golden'), ('银', 'silver')),
        (('紫', 'purple'), ('橙', 'orange')),
    ]

    identity_by_asset_id = {identity.asset_id: identity for identity in registry.identities}
    groups = {}

    for asset in registry.assets:
        if not asset.asset_id:
            continue
        if not _is_positive_status(asset.status):
            continue
        key = (asset.episode, asset.character, asset.asset_type)
        groups.setdefault(key, []).append(asset)

    emitted = set()

    for (episode, character, asset_type), assets in groups.items():
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                a = assets[i]
                b = assets[j]
                if not a.asset_id or not b.asset_id or a.asset_id == b.asset_id:
                    continue

                a_name = identity_by_asset_id.get(a.asset_id).canonical_name if a.asset_id in identity_by_asset_id else a.asset_name
                b_name = identity_by_asset_id.get(b.asset_id).canonical_name if b.asset_id in identity_by_asset_id else b.asset_name
                a_text = f"{a_name} {a.asset_name} {a.raw_text}".lower()
                b_text = f"{b_name} {b.asset_name} {b.raw_text}".lower()

                for left_colors, right_colors in color_pairs:
                    a_left = any(color in a_text for color in left_colors)
                    a_right = any(color in a_text for color in right_colors)
                    b_left = any(color in b_text for color in left_colors)
                    b_right = any(color in b_text for color in right_colors)

                    if not ((a_left and b_right) or (a_right and b_left)):
                        continue

                    dedupe_key = tuple(sorted([a.asset_id, b.asset_id])) + (episode, character, asset_type)
                    if dedupe_key in emitted:
                        continue
                    emitted.add(dedupe_key)

                    conflicts.append(Conflict(
                        rule_id='R09',
                        severity='P2',
                        description=f"同一集同角色同类资产存在互斥颜色： “{a_name}” 与 “{b_name}”。",
                        character=character,
                        episode_a=a.episode,
                        scene_a=a.scene,
                        episode_b=b.episode,
                        scene_b=b.scene,
                        raw_evidence=f"{a.raw_text}；{b.raw_text}",
                    ))

    return conflicts


def _check_r10_overlapping_character_locations(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []
    location_intervals_by_character = {}

    for lifecycle in registry.lifecycles:
        character = lifecycle.owner_character or _r_asset_owner(registry, lifecycle.asset_id)
        if not character:
            continue

        for interval in lifecycle.intervals:
            if interval.state_dimension != 'location':
                continue
            location_intervals_by_character.setdefault(character, []).append(interval)

    for character, intervals in location_intervals_by_character.items():
        intervals = sorted(intervals, key=lambda x: (x.start_episode, x.start_scene, _r_interval_end_key(x), x.value))
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                a = intervals[i]
                b = intervals[j]
                if (b.start_episode, b.start_scene) > _r_interval_end_key(a):
                    break
                if a.value == b.value:
                    continue
                if a.layer_id != b.layer_id:
                    continue
                if not _r_intervals_overlap(a, b):
                    continue

                overlap_start_episode = max(a.start_episode, b.start_episode)
                overlap_end_episode = min(_r_interval_end_key(a)[0], _r_interval_end_key(b)[0])
                if overlap_start_episode != overlap_end_episode:
                    continue

                conflicts.append(Conflict(
                    rule_id='R10',
                    severity='P1',
                    description=f"角色“{character}”在同一集内存在重叠的不同 location 区间：“{a.value}”与“{b.value}”。",
                    character=character,
                    episode_a=a.start_episode,
                    scene_a=a.start_scene,
                    episode_b=b.start_episode,
                    scene_b=b.start_scene,
                    raw_evidence=f"location={a.value} overlaps location={b.value}",
                ))

    return conflicts


def _check_r11_condition_jump_without_repair(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []
    bad_values = ('destroyed', 'damaged', '破损', '损坏', '毁坏', '摧毁', '报废', '坏了')
    good_values = ('normal', 'perfect', 'intact', '完好', '正常', '崭新', '无损', '完美')

    for lifecycle in registry.lifecycles:
        condition_events = [
            event for event in lifecycle.events
            if event.state_dimension == 'physical_condition'
            and event.affects_canonical_state
            and (event.to_value is not None or event.from_value is not None)
        ]
        condition_events.sort(key=lambda event: (event.episode, event.scene))

        for i in range(len(condition_events) - 1):
            a = condition_events[i]
            b = condition_events[i + 1]

            if b.episode - a.episode != 1:
                continue

            a_value = (a.to_value or a.from_value or '').lower()
            b_value = (b.to_value or b.from_value or '').lower()

            if not any(value in a_value for value in bad_values):
                continue
            if not any(value in b_value for value in good_values):
                continue

            has_repaired = any(
                event.event_type == 'repaired'
                and event.affects_canonical_state
                and (a.episode, a.scene) < (event.episode, event.scene) < (b.episode, b.scene)
                for event in lifecycle.events
            )
            if has_repaired:
                continue

            conflicts.append(Conflict(
                rule_id='R11',
                severity='P0',
                description=f"资产“{lifecycle.canonical_name}”的 physical_condition 在相邻集之间从“{a_value}”直接跳回“{b_value}”，缺少 repaired 事件。",
                character=lifecycle.owner_character or _r_asset_owner(registry, lifecycle.asset_id),
                episode_a=a.episode,
                scene_a=a.scene,
                episode_b=b.episode,
                scene_b=b.scene,
                raw_evidence=f"{a.evidence}；{b.evidence}",
            ))

    return conflicts


def _check_r12_body_feature_location_conflict(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []

    body_type_keywords = ('疤', 'scar', 'birthmark', '身体特征')
    conflict_pairs = [
        (('左', 'left'), ('右', 'right'), '左右'),
        (('手臂', '胳膊', '臂', 'arm'), ('腿', 'leg'), '手臂/腿'),
        (('手', 'hand'), ('脚', 'foot'), '手/脚'),
        (('脸', '面部', 'face'), ('背', '后背', 'back'), '脸/背'),
        (('额头', 'forehead'), ('下巴', 'chin'), '额头/下巴'),
    ]

    entries_by_asset_id = {}
    for asset in registry.assets:
        if not asset.asset_id:
            continue
        asset_type = (asset.asset_type or '').lower()
        if not any(keyword in asset_type for keyword in body_type_keywords):
            continue
        entries_by_asset_id.setdefault(asset.asset_id, []).append(asset)

    emitted = set()

    for asset_id, entries in entries_by_asset_id.items():
        entries = sorted(entries, key=lambda asset: (asset.episode, asset.scene))
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a = entries[i]
                b = entries[j]
                if (a.episode, a.scene) == (b.episode, b.scene):
                    continue

                a_text = f"{a.asset_name} {a.raw_text}".lower()
                b_text = f"{b.asset_name} {b.raw_text}".lower()

                for left_keywords, right_keywords, label in conflict_pairs:
                    a_left = any(keyword in a_text for keyword in left_keywords)
                    a_right = any(keyword in a_text for keyword in right_keywords)
                    b_left = any(keyword in b_text for keyword in left_keywords)
                    b_right = any(keyword in b_text for keyword in right_keywords)

                    if not ((a_left and b_right) or (a_right and b_left)):
                        continue

                    dedupe_key = (asset_id, a.episode, a.scene, b.episode, b.scene, label)
                    if dedupe_key in emitted:
                        continue
                    emitted.add(dedupe_key)

                    conflicts.append(Conflict(
                        rule_id='R12',
                        severity='P1',
                        description=f"身体特征类资产“{_r_asset_display_name(registry, asset_id)}”在不同场次出现互斥身体部位描述（{label}）。",
                        character=a.character,
                        episode_a=a.episode,
                        scene_a=a.scene,
                        episode_b=b.episode,
                        scene_b=b.scene,
                        raw_evidence=f"{a.raw_text}；{b.raw_text}",
                    ))

    return conflicts


def _check_r13_vague_asset_not_normalized(registry: AssetRegistry) -> List[Conflict]:
    conflicts = []
    groups = {}

    for asset in registry.assets:
        if not _is_positive_status(asset.status):
            continue
        key = (asset.episode, asset.character, asset.asset_type)
        groups.setdefault(key, []).append(asset)

    emitted = set()

    for (episode, character, asset_type), assets in groups.items():
        vague_assets = [asset for asset in assets if asset.is_vague]
        if not vague_assets:
            continue

        for vague_asset in vague_assets:
            for other in assets:
                if other is vague_asset:
                    continue
                if vague_asset.asset_id and other.asset_id and vague_asset.asset_id == other.asset_id:
                    continue

                asset_ids = [asset.asset_id for asset in assets if asset.asset_id]
                if len(asset_ids) != len(set(asset_ids)):
                    continue

                dedupe_key = (
                    episode,
                    character,
                    asset_type,
                    vague_asset.asset_id or vague_asset.asset_name,
                    other.asset_id or other.asset_name,
                )
                if dedupe_key in emitted:
                    continue
                emitted.add(dedupe_key)

                conflicts.append(Conflict(
                    rule_id='R13',
                    severity='P2',
                    description=f"同集同角色同类资产中，模糊资产“{vague_asset.asset_name}”与其他资产“{other.asset_name}”共存且 asset_id 未归一，存在潜在未归一化风险。",
                    character=character,
                    episode_a=vague_asset.episode,
                    scene_a=vague_asset.scene,
                    episode_b=other.episode,
                    scene_b=other.scene,
                    raw_evidence=f"{vague_asset.raw_text}；{other.raw_text}",
                ))

    return conflicts


def check_continuity(assets: AssetRegistry) -> ConflictReport:
    """
    Core rule-based continuity checker.

    Rules:
    - R01: asset首次出现 > 第1集 → P3
    - R02: 伤情恢复速度异常（使用故事天数，fallback 到集数差）
    - R03: 同集同角色同asset_type存在多个不同真实资产 → P1
    - R04: 资产消失后无过渡再次出现（跳过梦境/幻觉等非规范时间层）
    - R05-R13: 见各 _check_rXX_xxx 函数
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
                        f"资产「{entry.asset_name}」首次出现于第{entry.episode}集第{entry.scene}场，"
                        f"可能需要提前铺垫。"
                    ),
                    character=entry.character,
                    episode_a=entry.episode,
                    scene_a=entry.scene,
                    raw_evidence=entry.raw_text,
                )
            )

    # R02
    conflicts.extend(_check_r02_injury_recovery(registry))

    # R03
    entries_by_episode_character_type: Dict[
        Tuple[int, str, str], List[AssetEntry]
    ] = defaultdict(list)
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
                    should_report = a.asset_name != b.asset_name
                    pair_key = tuple(
                        sorted([a.asset_id or a.asset_name, b.asset_id or b.asset_name])
                    )

                if not should_report or pair_key in reported_pairs:
                    continue

                reported_pairs.add(pair_key)
                conflicts.append(
                    Conflict(
                        rule_id="R03",
                        severity="P1",
                        description=(
                            f"第{episode}集内，{character}的同类资产「{asset_type}」出现多个不同资产："
                            f"「{a.asset_name}」(asset_id={a.asset_id or 'None'}) 与 "
                            f"「{b.asset_name}」(asset_id={b.asset_id or 'None'})。"
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
    conflicts.extend(_check_r04_asset_presence(registry))

    # R05-R13
    conflicts.extend(_check_r05_destroyed_reappear(registry))
    conflicts.extend(_check_r06_overlapping_holder_intervals(registry))
    conflicts.extend(_check_r07_setting_violated_by_asset_description(registry))
    conflicts.extend(_check_r08_asset_before_first_story_time(registry))
    conflicts.extend(_check_r09_mutually_exclusive_color_assets(registry))
    conflicts.extend(_check_r10_overlapping_character_locations(registry))
    conflicts.extend(_check_r11_condition_jump_without_repair(registry))
    conflicts.extend(_check_r12_body_feature_location_conflict(registry))
    conflicts.extend(_check_r13_vague_asset_not_normalized(registry))

    return ConflictReport.from_conflicts(conflicts)


def semantic_review(registry: AssetRegistry, llm_client: Any = None) -> ConflictReport:
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
            suggestion = "确认这些同类型资产是否确为不同实物；若是同一资产，请统一名称；若是不同资产，请增加换装/更换/同时持有的交代。"
            priority = "high"
        elif conflict.rule_id == "R04":
            suggestion = "在资产重新出现前补充找回、修复或重新获得的过渡场景或台词。"
            priority = "high"
        elif conflict.rule_id == "R05":
            suggestion = "补充该资产从 destroyed 到再次出现之间的 repaired/修复事件，或将后续出现改为回忆、误认、复制品/替代品，或修正 destroyed 状态。"
            priority = "critical"
        elif conflict.rule_id == "R06":
            suggestion = "拆分或调整 holder 状态区间，明确资产在重叠时间内的唯一持有者；如为共同持有，请改用明确的共享持有描述。"
            priority = "high"
        elif conflict.rule_id == "R07":
            suggestion = "核对角色硬性设定与后续行为/资产描述：修改违反设定的动作，或在此前增加合理解释、能力变化或设定更新。"
            priority = "high"
        elif conflict.rule_id == "R08":
            suggestion = "确认该场景的时间层归属；若为闪回/平行时间线，请在数据中标注正确的 temporal_mode；若为主线错误，请调整资产首次出现位置。"
            priority = "high"
        elif conflict.rule_id == "R09":
            suggestion = "确认同集同角色的两件同类资产是否为同一物；若是，请归一化为同一 asset_id；若确为不同资产，请补充换装或更换说明。"
            priority = "medium"
        elif conflict.rule_id == "R10":
            suggestion = "检查角色在同集内跨地点出现是否有合理的移动交代；若无，请补充移动场景或修正地点信息。"
            priority = "high"
        elif conflict.rule_id == "R11":
            suggestion = "在相邻集之间补充 repaired/修复事件或修复说明，或将 physical_condition 的跳变改为连续、合理的状态变化。"
            priority = "critical"
        elif conflict.rule_id == "R12":
            suggestion = "统一身体特征资产的部位描述；若确有多个不同身体特征，请拆分为不同 asset_id 并分别登记。"
            priority = "high"
        elif conflict.rule_id == "R13":
            suggestion = "对模糊资产进行归一化：确认其是否与同类资产为同一物，若是则合并 asset_id；若不是则补充更明确的名称和识别特征。"
            priority = "medium"
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


def _apply_severity_filter(report: ConflictReport, severity_filter: str) -> ConflictReport:
    if severity_filter == "all":
        return report
    allowed = {"P0"} if severity_filter == "high" else {"P0", "P1"}
    filtered = [c for c in report.conflicts if c.severity in allowed]
    return ConflictReport.from_conflicts(filtered)


def full_check(
    script_text: str,
    script_language: str = "zh",
    llm_client: Any = None,
    enable_semantic_review: bool = False,
    check_mode: str = "full",
    severity_filter: str = "all",
) -> ContinuityReport:
    """
    Full continuity checking pipeline.

    check_mode:
      full           — 完整流水线（提取→规则→语义→修复建议）
      assets_only    — 只提取资产档案，不做冲突检测
      conflicts_only — 提取+冲突检测，不生成修复建议

    severity_filter:
      all            — 返回全部级别冲突
      high           — 只返回 P0
      high_medium    — 返回 P0 + P1
    """
    registry = extract_assets(script_text, llm_client=llm_client)
    registry = normalize_assets(registry, llm_client=llm_client)
    registry = build_story_timeline(registry, llm_client=llm_client)
    registry = build_asset_lifecycles(registry)

    if check_mode == "assets_only":
        return ContinuityReport(
            script_language=script_language,
            asset_registry=registry,
            conflicts=ConflictReport.from_conflicts([]),
            fix_suggestions=FixSuggestions(suggestions=[]),
            summary="assets_only 模式：已完成资产提取，未执行冲突检测。",
        )

    rule_report = check_continuity(registry)

    if enable_semantic_review:
        try:
            semantic_report = semantic_review(registry=registry, llm_client=llm_client)
            if semantic_report and getattr(semantic_report, "conflicts", None):
                all_conflicts = list(rule_report.conflicts) + list(semantic_report.conflicts)
                rule_report = ConflictReport.from_conflicts(all_conflicts)
        except Exception:
            pass

    rule_report = _apply_severity_filter(rule_report, severity_filter)

    if check_mode == "conflicts_only":
        summary = (
            f"共发现 {rule_report.total_count} 个连续性问题："
            f"P0={rule_report.p0_count}, P1={rule_report.p1_count}, "
            f"P2={rule_report.p2_count}, P3={rule_report.p3_count}。"
        )
        return ContinuityReport(
            script_language=script_language,
            asset_registry=registry,
            conflicts=rule_report,
            fix_suggestions=FixSuggestions(suggestions=[]),
            summary=summary,
        )

    fix_suggestions = generate_fix_suggestions(
        conflicts=rule_report,
        registry=registry,
        llm_client=llm_client,
    )

    summary = (
        f"共发现 {rule_report.total_count} 个连续性问题："
        f"P0={rule_report.p0_count}, "
        f"P1={rule_report.p1_count}, "
        f"P2={rule_report.p2_count}, "
        f"P3={rule_report.p3_count}。"
    )

    return ContinuityReport(
        script_language=script_language,
        asset_registry=registry,
        conflicts=rule_report,
        fix_suggestions=fix_suggestions,
        summary=summary,
    )


__all__ = [
    "build_asset_lifecycles",
    "build_story_timeline",
    "check_continuity",
    "extract_assets",
    "full_check",
    "generate_fix_suggestions",
    "merge_conflict_reports",
    "normalize_assets",
    "semantic_review",
]
