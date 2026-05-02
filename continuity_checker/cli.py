#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import requests

from continuity_checker.checker import full_check
from continuity_checker.parsers import parse_file


class SimpleLLMClient:
    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    def __call__(self, prompt: str) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严谨的剧本连续性审查助手。所有回答必须遵循用户要求的 JSON 格式。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


_SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _report_to_text(report) -> str:
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("摘要 (Summary)")
    lines.append("=" * 60)
    lines.append(report.summary)
    lines.append("")

    conflicts = report.conflicts
    lines.append("=" * 60)
    lines.append(f"冲突详情 (共 {conflicts.total_count} 项)")
    lines.append("=" * 60)

    grouped: dict[str, list] = {}
    for c in conflicts.conflicts:
        grouped.setdefault(c.severity, []).append(c)

    for sev_key in sorted(grouped.keys(), key=lambda s: _SEVERITY_ORDER.get(s, 99)):
        group = grouped[sev_key]
        lines.append("")
        lines.append(f"── {sev_key} ({len(group)} 项) ──")
        for idx, c in enumerate(group, 1):
            lines.append(f"  [{idx}] rule_id   : {c.rule_id}")
            lines.append(f"      severity  : {c.severity}")
            if c.scene_ids:
                lines.append(f"      场景      : {', '.join(str(s) for s in c.scene_ids)}")
            if c.character_names:
                lines.append(f"      角色      : {', '.join(c.character_names)}")
            if c.asset_names:
                lines.append(f"      资产      : {', '.join(c.asset_names)}")
            lines.append(f"      描述      : {c.description}")
            if c.evidence:
                lines.append(f"      证据      : {c.evidence}")
            lines.append("")

    lines.append("=" * 60)
    lines.append("修复建议 (Fix Suggestions)")
    lines.append("=" * 60)
    for idx, s in enumerate(report.fix_suggestions.suggestions, 1):
        lines.append(f"  [{idx}] conflict_rule_id : {s.conflict_rule_id}")
        lines.append(f"      冲突描述         : {s.conflict_description}")
        lines.append(f"      suggestion     : {s.suggestion}")
        lines.append(f"      priority       : {s.priority}")
        lines.append("")

    return "\n".join(lines)


def _build_llm_client() -> SimpleLLMClient:
    api_key = os.environ.get("CONTINUITY_API_KEY")
    if not api_key:
        print(
            "错误: 未设置环境变量 CONTINUITY_API_KEY。\n"
            "请通过 export CONTINUITY_API_KEY=<your_key> 设置后再运行。",
            file=sys.stderr,
        )
        sys.exit(1)

    api_base = os.environ.get(
        "CONTINUITY_API_BASE", "https://open.bigmodel.cn/api/paas/v4"
    )
    model = os.environ.get("CONTINUITY_MODEL", "glm-5.1")
    return SimpleLLMClient(api_key=api_key, api_base=api_base, model=model)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="juben-check",
        description="剧本连续性检查工具 — 检测剧本中的设定冲突与资产不一致",
    )
    parser.add_argument("file", metavar="FILE", help="剧本文件路径（支持 .pdf / .docx / .txt）")
    parser.add_argument(
        "--mode",
        choices=["full", "assets_only", "conflicts_only"],
        default="full",
        help="检查模式（默认: full）",
    )
    parser.add_argument(
        "--severity",
        choices=["all", "high", "high_medium"],
        default="all",
        help="严重度过滤（默认: all）",
    )
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="剧本语言（默认: zh）")
    parser.add_argument("--semantic", action="store_true", default=False, help="启用语义审查")
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="输出文件路径（支持 .json / .txt，默认打印到 stdout）",
    )

    args = parser.parse_args(argv)

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"错误: 文件不存在 — {args.file}", file=sys.stderr)
        sys.exit(1)

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(
            f"错误: 不支持的文件格式 '{ext}'，仅支持 {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        script_text = parse_file(str(file_path))
    except Exception as e:
        print(f"错误: 解析文件失败 — {e}", file=sys.stderr)
        sys.exit(1)

    if not script_text or not script_text.strip():
        print("错误: 文件内容为空", file=sys.stderr)
        sys.exit(1)

    llm_client = _build_llm_client()

    try:
        report = full_check(
            script_text=script_text,
            script_language=args.lang,
            llm_client=llm_client,
            enable_semantic_review=args.semantic,
            check_mode=args.mode,
            severity_filter=args.severity,
        )
    except Exception as e:
        print(f"错误: 检查执行失败 — {e}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path:
        out_ext = Path(output_path).suffix.lower()
        content = report.to_json() if out_ext == ".json" else _report_to_text(report)
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(content, encoding="utf-8")
        except OSError as e:
            print(f"错误: 写入输出文件失败 — {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(_report_to_text(report))


if __name__ == "__main__":
    main()
