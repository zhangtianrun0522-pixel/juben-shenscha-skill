from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AssetIdentity(BaseModel):
    asset_id: str
    canonical_name: str
    aliases: List[str] = Field(default_factory=list)
    character: str
    asset_type: str
    first_episode: int
    first_scene: int
    is_vague: bool = False


class AssetEntry(BaseModel):
    character: str
    asset_type: str
    asset_name: str
    episode: int
    scene: int
    page: Optional[int] = None
    raw_text: str
    status: str = 'exists'
    is_vague: bool = False
    asset_id: Optional[str] = None


class StateChange(BaseModel):
    character: str
    asset_type: str
    change_from: str
    change_to: str
    episode: int
    scene: int
    in_story_time_note: Optional[str] = None


class TimelineAnchor(BaseModel):
    episode: int
    scene: int
    time_label: str
    is_flashback: bool = False


class TimeLayer(BaseModel):
    layer_id: str
    name: str
    layer_type: str  # 'main'/'flashback'/'dream'/'hallucination'/'parallel'/'flashforward'
    parent_layer_id: Optional[str] = None
    is_canonical: bool = True
    affects_canonical_state: bool = True
    confidence: float = 1.0


class StoryTimeAnchor(BaseModel):
    episode: int
    scene: int
    layer_id: str  # 对应 TimeLayer.layer_id
    temporal_mode: str  # 'present'/'flashback'/'dream'/'hallucination'/'parallel'/'flashforward'
    relative_time_label: Optional[str] = None  # 原文：「三个月后」「十年前」
    story_offset_days: Optional[float] = None  # 相对主线第一场的天数偏移，正=未来负=过去
    story_offset_min: Optional[float] = None   # 最小偏移，模糊时间用范围
    story_offset_max: Optional[float] = None
    is_time_jump: bool = False
    confidence: float = 1.0
    evidence: str = ''


class TimeRelation(BaseModel):
    from_episode: int
    from_scene: int
    to_episode: int
    to_scene: int
    relation_type: str  # 'before'/'after'/'same_time'/'immediately_after'
    duration_days_min: Optional[float] = None
    duration_days_max: Optional[float] = None
    evidence: str = ''
    confidence: float = 1.0


class CharacterSetting(BaseModel):
    character: str
    setting_type: str
    content: str
    episode: int
    scene: int
    raw_text: str


class AssetStateEvent(BaseModel):
    asset_id: str
    episode: int
    scene: int
    layer_id: str = 'main'
    event_type: str  # 'appear'/'disappear'/'transfer'/'damaged'/'repaired'/'destroyed'
    state_dimension: str  # 'holder'/'location'/'physical_condition'/'ownership'
    from_value: Optional[str] = None
    to_value: Optional[str] = None
    affects_canonical_state: bool = True
    is_explicit: bool = True
    confidence: float = 1.0
    evidence: str = ''


class AssetStateInterval(BaseModel):
    asset_id: str
    state_dimension: str
    value: str
    start_episode: int
    start_scene: int
    end_episode: Optional[int] = None
    end_scene: Optional[int] = None
    layer_id: str = 'main'
    is_open_ended: bool = False
    confidence: float = 1.0


class AssetLifecycle(BaseModel):
    asset_id: str
    canonical_name: str
    asset_type: str
    owner_character: Optional[str] = None
    first_episode: int
    first_scene: int
    last_episode: Optional[int] = None
    last_scene: Optional[int] = None
    lifecycle_status: str = 'active'  # 'active'/'destroyed'/'lost'/'transferred'/'unknown'
    events: List[AssetStateEvent] = Field(default_factory=list)
    intervals: List[AssetStateInterval] = Field(default_factory=list)


class AssetRegistry(BaseModel):
    assets: List[AssetEntry] = Field(default_factory=list)
    state_changes: List[StateChange] = Field(default_factory=list)
    timeline: List[TimelineAnchor] = Field(default_factory=list)
    character_settings: List[CharacterSetting] = Field(default_factory=list)
    identities: List[AssetIdentity] = Field(default_factory=list)

    story_anchors: List[StoryTimeAnchor] = Field(default_factory=list)
    time_layers: List[TimeLayer] = Field(default_factory=list)
    time_relations: List[TimeRelation] = Field(default_factory=list)
    lifecycles: List[AssetLifecycle] = Field(default_factory=list)


class Conflict(BaseModel):
    rule_id: str
    severity: Literal['P0', 'P1', 'P2', 'P3']
    description: str
    character: Optional[str] = None
    episode_a: Optional[int] = None
    scene_a: Optional[int] = None
    episode_b: Optional[int] = None
    scene_b: Optional[int] = None
    raw_evidence: Optional[str] = None


class ConflictReport(BaseModel):
    conflicts: List[Conflict] = Field(default_factory=list)
    total_count: int = 0
    p0_count: int = 0
    p1_count: int = 0
    p2_count: int = 0
    p3_count: int = 0

    @classmethod
    def from_conflicts(cls, conflicts):
        p0 = sum(1 for c in conflicts if c.severity == 'P0')
        p1 = sum(1 for c in conflicts if c.severity == 'P1')
        p2 = sum(1 for c in conflicts if c.severity == 'P2')
        p3 = sum(1 for c in conflicts if c.severity == 'P3')
        return cls(
            conflicts=conflicts,
            total_count=len(conflicts),
            p0_count=p0,
            p1_count=p1,
            p2_count=p2,
            p3_count=p3,
        )


class FixSuggestion(BaseModel):
    conflict_rule_id: str
    conflict_description: str
    suggestion: str
    priority: str


class FixSuggestions(BaseModel):
    suggestions: List[FixSuggestion] = Field(default_factory=list)


class ContinuityReport(BaseModel):
    script_language: str
    asset_registry: AssetRegistry
    conflicts: ConflictReport
    fix_suggestions: FixSuggestions
    summary: str

    def to_json(self):
        return self.model_dump_json(indent=2, ensure_ascii=False)

    @staticmethod
    def _md_escape(value: str) -> str:
        return value.replace('|', '\\|')

    def to_markdown(self):
        lines = ['# 剧本连贯性审查报告\n']
        lines.append(f'**语言**: {self.script_language}\n')
        lines.append(f'**摘要**: {self.summary}\n')
        lines.append('## 冲突列表\n')
        lines.append('| Rule ID | Severity | Character | Description | Location A | Location B |')
        lines.append('|---|---|---|---|---|---|')

        for c in self.conflicts.conflicts:
            loc_a = f'E{c.episode_a}S{c.scene_a}' if c.episode_a is not None and c.scene_a is not None else '-'
            loc_b = f'E{c.episode_b}S{c.scene_b}' if c.episode_b is not None and c.scene_b is not None else '-'
            desc = self._md_escape(c.description)
            character = c.character or '-'
            lines.append(
                f'| {c.rule_id} | {c.severity} | {character} | {desc} | {loc_a} | {loc_b} |'
            )

        lines.append('\n## 修复建议\n')
        lines.append('| Rule ID | Conflict Description | Suggestion | Priority |')
        lines.append('|---|---|---|---|')

        for s in self.fix_suggestions.suggestions:
            conflict_desc = self._md_escape(s.conflict_description)
            suggestion = self._md_escape(s.suggestion)
            lines.append(
                f'| {s.conflict_rule_id} | {conflict_desc} | {suggestion} | {s.priority} |'
            )

        return '\n'.join(lines)