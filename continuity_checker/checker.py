import os
import re
import json
import urllib.request
from collections import defaultdict

from .models import (
    AssetEntry, StateChange, TimelineAnchor, CharacterSetting,
    AssetRegistry, Conflict, ConflictReport, FixSuggestion, FixSuggestions, ContinuityReport,
)
from .prompts import get_extraction_prompt, get_semantic_review_prompt, get_fix_suggestion_prompt


def call_llm(prompt: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    api_key = os.environ.get("LLM_API_KEY", "")
    default_model = "claude-sonnet-4-20250514" if provider == "anthropic" else "gpt-4o"
    model = os.environ.get("LLM_MODEL", default_model)
    messages = [{"role": "user", "content": prompt}]

    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {"model": model, "max_tokens": 4096, "messages": messages}
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {"model": model, "max_tokens": 4096, "messages": messages}

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req) as resp:
        response = json.loads(resp.read().decode("utf-8"))

    if provider == "anthropic":
        return response["content"][0]["text"]
    else:
        return response["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    try:
        match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return {}


def detect_language(text: str) -> str:
    cjk_count = len(re.findall(r"[一-鿿]", text))
    return "zh" if len(text) > 0 and cjk_count / len(text) > 0.2 else "en"


def extract_assets(script_text: str) -> AssetRegistry:
    try:
        prompt = get_extraction_prompt(script_text)
        text = call_llm(prompt)
        data = extract_json(text)

        assets_list = []
        for item in data.get("assets", []):
            try:
                assets_list.append(AssetEntry.model_validate(item))
            except Exception:
                pass

        state_changes_list = []
        for item in data.get("state_changes", []):
            try:
                state_changes_list.append(StateChange.model_validate(item))
            except Exception:
                pass

        timeline_list = []
        for item in data.get("timeline", []):
            try:
                timeline_list.append(TimelineAnchor.model_validate(item))
            except Exception:
                pass

        character_settings_list = []
        for item in data.get("character_settings", []):
            try:
                character_settings_list.append(CharacterSetting.model_validate(item))
            except Exception:
                pass

        return AssetRegistry(
            assets=assets_list,
            state_changes=state_changes_list,
            timeline=timeline_list,
            character_settings=character_settings_list,
        )
    except Exception:
        return AssetRegistry()


def check_continuity(assets: AssetRegistry) -> ConflictReport:
    conflicts = []

    # R01: asset first appears after episode 1 — no prior setup
    r01_groups: dict = defaultdict(list)
    for a in assets.assets:
        r01_groups[(a.character, a.asset_type, a.asset_name)].append(a)

    for (character, asset_type, asset_name), items in r01_groups.items():
        min_item = min(items, key=lambda x: x.episode)
        if min_item.episode > 1:
            conflicts.append(
                Conflict(
                    rule_id="R01",
                    severity="P3",
                    description=f"{character}的{asset_name}首次出现在第{min_item.episode}集，之前集数无铺垫记录",
                    character=character,
                    episode_a=min_item.episode,
                    scene_a=min_item.scene,
                )
            )

    # R02: injury heals too fast
    INJURED_KEYWORDS = ["受伤", "injured", "broken", "hurt", "伤"]
    HEALED_KEYWORDS = ["痊愈", "恢复", "normal", "healed", "recovered", "好了"]
    SPECIAL = ["手术", "特殊", "治疗"]

    r02_groups: dict = defaultdict(list)
    for sc in assets.state_changes:
        r02_groups[(sc.character, sc.asset_type)].append(sc)

    for (character, asset_type), items in r02_groups.items():
        injured_list = [sc for sc in items if any(kw in sc.change_to for kw in INJURED_KEYWORDS)]
        healed_list = [sc for sc in items if any(kw in sc.change_to for kw in HEALED_KEYWORDS)]
        for inj in injured_list:
            for heal in healed_list:
                if 0 <= heal.episode - inj.episode <= 1:
                    if not any(sp in (heal.in_story_time_note or "") for sp in SPECIAL):
                        conflicts.append(
                            Conflict(
                                rule_id="R02",
                                severity="P1",
                                description=f"{character}的{asset_type}在第{inj.episode}集受伤，第{heal.episode}集即痊愈，恢复过快",
                                character=character,
                                episode_a=inj.episode,
                                episode_b=heal.episode,
                            )
                        )

    # R03: contradictory asset descriptions within same episode
    r03_groups: dict = defaultdict(list)
    for a in assets.assets:
        r03_groups[(a.character, a.episode, a.asset_type)].append(a)

    for (character, ep, asset_type), items in r03_groups.items():
        if len(items) >= 2:
            names = {it.asset_name for it in items}
            if len(names) >= 2:
                it1 = items[0]
                it2 = next((it for it in items[1:] if it.asset_name != it1.asset_name), None)
                if it2:
                    conflicts.append(
                        Conflict(
                            rule_id="R03",
                            severity="P1",
                            description=f"{character}在第{ep}集第{it1.scene}和{it2.scene}场对{asset_type}有矛盾描述",
                            character=character,
                            episode_a=ep,
                            scene_a=it1.scene,
                            episode_b=ep,
                            scene_b=it2.scene,
                        )
                    )

    # R04: flashback asset conflicts with modern timeline disappearance
    flashback_eps = {a.episode for a in assets.timeline if a.is_flashback}
    GONE_KEYWORDS = ["消失", "removed", "gone", "不见了", "丢失"]

    for lost_sc in assets.state_changes:
        if any(kw in lost_sc.change_to for kw in GONE_KEYWORDS) and lost_sc.episode not in flashback_eps:
            flashback_assets = [
                a
                for a in assets.assets
                if a.character == lost_sc.character
                and a.asset_type == lost_sc.asset_type
                and a.episode in flashback_eps
            ]
            if flashback_assets:
                flashback_ep = min(a.episode for a in flashback_assets)
                conflicts.append(
                    Conflict(
                        rule_id="R04",
                        severity="P1",
                        description=f"{lost_sc.character}的{lost_sc.asset_type}在第{lost_sc.episode}集消失，但闪回中（第{flashback_ep}集）仍存在且无交代",
                        character=lost_sc.character,
                        episode_a=lost_sc.episode,
                        episode_b=flashback_ep,
                    )
                )

    return ConflictReport.from_conflicts(conflicts)


def suggest_fixes(conflicts: ConflictReport, assets: AssetRegistry) -> FixSuggestions:
    try:
        prompt = get_fix_suggestion_prompt(conflicts.model_dump_json(), assets.model_dump_json())
        text = call_llm(prompt)
        data = extract_json(text)
        valid_suggestions = []
        for item in data.get("suggestions", []):
            try:
                valid_suggestions.append(FixSuggestion.model_validate(item))
            except Exception:
                continue
        return FixSuggestions(suggestions=valid_suggestions)
    except Exception:
        return FixSuggestions()


def full_check(
    script_text: str, check_mode: str = "full", severity_filter: str = "all"
) -> ContinuityReport:
    script_language = detect_language(script_text)
    assets = extract_assets(script_text)

    if check_mode == "assets_only":
        return ContinuityReport(
            script_language=script_language,
            asset_registry=assets,
            conflicts=ConflictReport(),
            fix_suggestions=FixSuggestions(),
            summary="",
        )

    rule_report = check_continuity(assets)
    all_conflicts = list(rule_report.conflicts)

    if check_mode == "full":
        try:
            prompt = get_semantic_review_prompt(assets.model_dump_json())
            text = call_llm(prompt)
            data = extract_json(text)
            for item in data.get("conflicts", []):
                try:
                    all_conflicts.append(Conflict.model_validate(item))
                except Exception:
                    continue
        except Exception:
            pass

    if severity_filter == "high":
        filtered_conflicts = [c for c in all_conflicts if c.severity == "P0"]
    elif severity_filter == "high_medium":
        filtered_conflicts = [c for c in all_conflicts if c.severity in ("P0", "P1")]
    else:
        filtered_conflicts = all_conflicts

    filtered_report = ConflictReport.from_conflicts(filtered_conflicts)

    if check_mode == "conflicts_only":
        fixes = FixSuggestions()
    else:
        fixes = suggest_fixes(filtered_report, assets)

    p0 = filtered_report.p0_count
    p1 = filtered_report.p1_count
    p2 = filtered_report.p2_count
    p3 = filtered_report.p3_count
    total = filtered_report.total_count

    if script_language == "zh":
        summary = f"共发现{total}条冲突：P0 {p0}条，P1 {p1}条，P2 {p2}条，P3 {p3}条。"
    else:
        summary = f"A total of {total} conflicts found: {p0} P0, {p1} P1, {p2} P2, {p3} P3."

    return ContinuityReport(
        script_language=script_language,
        asset_registry=assets,
        conflicts=filtered_report,
        fix_suggestions=fixes,
        summary=summary,
    )
