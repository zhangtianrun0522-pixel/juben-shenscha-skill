from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AssetEntry(BaseModel):
    character: str
    asset_type: str
    asset_name: str
    episode: int
    scene: int
    page: Optional[int] = None
    raw_text: str
    status: str = "exists"
    is_vague: bool = False


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


class CharacterSetting(BaseModel):
    character: str
    setting_type: str
    content: str
    episode: int
    scene: int
    raw_text: str


class AssetRegistry(BaseModel):
    assets: List[AssetEntry] = Field(default_factory=list)
    state_changes: List[StateChange] = Field(default_factory=list)
    timeline: List[TimelineAnchor] = Field(default_factory=list)
    character_settings: List[CharacterSetting] = Field(default_factory=list)


class Conflict(BaseModel):
    rule_id: str
    severity: Literal["P0", "P1", "P2", "P3"]
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
    def from_conflicts(cls, conflicts: List[Conflict]) -> "ConflictReport":
        p0 = sum(1 for c in conflicts if c.severity == "P0")
        p1 = sum(1 for c in conflicts if c.severity == "P1")
        p2 = sum(1 for c in conflicts if c.severity == "P2")
        p3 = sum(1 for c in conflicts if c.severity == "P3")
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

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        lines = ["# 剧本连贯性审查报告\n"]
        lines.append(f"**语言**: {self.script_language}\n")
        lines.append(f"**摘要**: {self.summary}\n")

        lines.append("## 冲突列表\n")
        lines.append("| Rule ID | Severity | Character | Description | Location A | Location B |")
        lines.append("|---|---|---|---|---|---|")
        for c in self.conflicts.conflicts:
            loc_a = f"E{c.episode_a}S{c.scene_a}" if c.episode_a is not None and c.scene_a is not None else "-"
            loc_b = f"E{c.episode_b}S{c.scene_b}" if c.episode_b is not None and c.scene_b is not None else "-"
            desc = c.description.replace("|", "\\|")
            char = c.character.replace("|", "\\|") if c.character else "-"
            lines.append(f"| {c.rule_id} | {c.severity} | {char} | {desc} | {loc_a} | {loc_b} |")

        lines.append("\n## 修复建议\n")
        lines.append("| Rule ID | Conflict Description | Suggestion | Priority |")
        lines.append("|---|---|---|---|")
        for s in self.fix_suggestions.suggestions:
            c_desc = s.conflict_description.replace("|", "\\|")
            sugg = s.suggestion.replace("|", "\\|")
            lines.append(f"| {s.conflict_rule_id} | {c_desc} | {sugg} | {s.priority} |")

        return "\n".join(lines)
