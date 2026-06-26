#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Final Premium report UX pass.

This script only changes HTML wording and final report presentation. It does
not touch prediction, training, validation truth, or ranking logic.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "reports"

METRICS_COLLAPSE_STYLE = """
    .metrics-collapsible{margin:0 0 16px;padding:10px;background:rgba(255,255,255,.72);border:1px solid var(--line-soft);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.03)}
    .metrics-collapse-head{display:flex;align-items:center;justify-content:space-between;gap:14px;min-height:42px;padding:2px 4px 2px 8px}
    .metrics-collapse-title{display:flex;align-items:baseline;gap:10px;min-width:0}
    .metrics-collapse-title span{color:var(--muted);font-size:12px;line-height:1.35;font-weight:600;white-space:nowrap}
    .metrics-collapse-title strong{color:var(--ink);font-size:16px;line-height:1.2;font-weight:700;white-space:nowrap}
    .metrics-toggle{width:34px;height:34px;min-width:34px;border:1px solid #1d1d1f;border-radius:50%;background:#1d1d1f;color:#fff;display:inline-flex;align-items:center;justify-content:center;padding:0;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.10);transition:background .18s ease,color .18s ease,border-color .18s ease,transform .18s ease}
    .metrics-toggle:hover{background:#000;border-color:#000;transform:translateY(-1px)}
    .metrics-toggle span{display:block;font-size:22px;line-height:1;font-weight:500;transform:translateY(-1px)}
    .metrics-collapsible .metrics{margin:10px 0 0}
    .metrics-collapsible.collapsed .metrics{display:none}
    @media(max-width:560px){.metrics-collapse-title{flex-direction:column;align-items:flex-start;gap:2px}.metrics-collapse-head{padding-left:4px}}
"""

METRICS_COLLAPSE_SCRIPT = """
    (function premiumMetricsCollapse(){
      const box = document.querySelector('.metrics-collapsible');
      if (!box) return;
      const btn = box.querySelector('.metrics-toggle');
      if (!btn) return;
      const icon = btn.querySelector('span');
      const sync = () => {
        const collapsed = box.classList.contains('collapsed');
        btn.setAttribute('aria-expanded', String(!collapsed));
        btn.setAttribute('title', collapsed ? '展开指标' : '折叠指标');
        btn.setAttribute('aria-label', collapsed ? '展开指标' : '折叠指标');
        if (icon) icon.textContent = collapsed ? '+' : '-';
      };
      btn.addEventListener('click', () => {
        box.classList.toggle('collapsed');
        sync();
      });
      sync();
    })();
"""


def iter_report_files(report_dir: Path) -> list[Path]:
    return sorted(report_dir.glob("premium_*.html")) + sorted(report_dir.glob("premium_latest.html"))


def _plain(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value or "")).strip()


def _num_from_text(value: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", _plain(value))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _clean_separators(value: str) -> str:
    value = html.unescape(_plain(value)).replace(";", "；")
    value = re.sub(r"\s*；\s*", "；", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ；")


def _brier_state(value: float | None) -> str:
    if value is None:
        return "当前-"
    if value <= 0.15:
        return "当前较好"
    if value <= 0.25:
        return "当前可用"
    return "当前偏弱"


def _ic_state(value: float | None) -> str:
    if value is None:
        return "当前-"
    if value >= 0.20:
        return "当前较好"
    if value > 0:
        return "当前有效"
    if value < 0:
        return "当前反向"
    return "接近无效"


def simplify_current_truth_note(text: str) -> str:
    text = re.sub(
        r"，命中率 ([^；<]+)；验证口径：T日收盘涨停=命中；T\+1 \d{8} 用于后续接力验证。",
        r"，命中率 \1。",
        text,
    )
    text = re.sub(
        r"，等待T日收盘后的市场真值，当前不计入命中率；验证口径：T日收盘涨停=命中；T\+1 \d{8} 用于后续接力验证。",
        "，等待T日收盘后的市场真值，当前不计入命中率。",
        text,
    )
    return text


def enhance_calibration_card(text: str) -> str:
    pattern = re.compile(
        r'(<div class="metric"><span>概率校准质量</span><strong>)(.*?)(</strong><small>)(.*?)(</small></div>)',
        re.S,
    )

    def repl(match: re.Match[str]) -> str:
        strong = match.group(2)
        old_note = _clean_separators(match.group(4))
        brier = _num_from_text(strong)
        if "越低越好" in old_note:
            note = old_note
        else:
            note = f"Brier/ECE越低越好；最佳0，最差1；{_brier_state(brier)}；{old_note}"
        return f"{match.group(1)}{strong}{match.group(3)}{html.escape(note, quote=True)}{match.group(5)}"

    return pattern.sub(repl, text)


def enhance_rank_ic_card(text: str, label: str) -> str:
    pattern = re.compile(
        r'(<div class="metric"><span>' + re.escape(label) + r'</span><strong>)(.*?)(</strong><small>)(.*?)(</small></div>)',
        re.S,
    )

    def repl(match: re.Match[str]) -> str:
        strong = match.group(2)
        old_note = _clean_separators(match.group(4))
        ic_value = _num_from_text(strong)
        if "越高越好" in old_note:
            note = old_note
        else:
            note = f"越高越好；>0有效，<0反向；{_ic_state(ic_value)}；{old_note}"
        return f"{match.group(1)}{strong}{match.group(3)}{html.escape(note, quote=True)}{match.group(5)}"

    return pattern.sub(repl, text)


def inject_metrics_collapse_style(text: str) -> str:
    if ".metrics-collapsible" in text:
        return text
    return text.replace("\n  </style>", f"{METRICS_COLLAPSE_STYLE}\n  </style>", 1)


def wrap_metrics_block(text: str) -> str:
    if 'class="metrics-collapsible' in text:
        return text

    pattern = re.compile(r'(?P<metrics><div class="metrics">[\s\S]*?</div>)\s*(?=<div class="toolbar">)', re.S)

    def repl(match: re.Match[str]) -> str:
        metrics = match.group("metrics").replace(
            '<div class="metrics">',
            '<div class="metrics" id="premium-metrics-grid">',
            1,
        )
        return (
            '<div class="metrics-collapsible collapsed" aria-label="关键指标">\n'
            '      <div class="metrics-collapse-head">\n'
            '        <div class="metrics-collapse-title"><span>Report Metrics</span><strong>关键指标</strong></div>\n'
            '        <button class="metrics-toggle" type="button" aria-expanded="false" aria-controls="premium-metrics-grid" aria-label="展开指标" title="展开指标"><span aria-hidden="true">+</span></button>\n'
            '      </div>\n'
            f'      {metrics}\n'
            '    </div>\n    '
        )

    return pattern.sub(repl, text, count=1)


def inject_metrics_collapse_script(text: str) -> str:
    if "premiumMetricsCollapse" in text:
        return text
    if "</script>" in text:
        return text.replace("\n  </script>", f"{METRICS_COLLAPSE_SCRIPT}\n  </script>", 1)
    return text.replace("\n</body>", f"\n  <script>{METRICS_COLLAPSE_SCRIPT}\n  </script>\n</body>", 1)


def collapse_metrics_block(text: str) -> str:
    text = inject_metrics_collapse_style(text)
    text = wrap_metrics_block(text)
    text = inject_metrics_collapse_script(text)
    return text


def clarify_html(text: str) -> str:
    text = simplify_current_truth_note(text)
    text = enhance_calibration_card(text)
    text = enhance_rank_ic_card(text, "涨停Rank IC")
    text = enhance_rank_ic_card(text, "T+1收益Rank IC")
    text = collapse_metrics_block(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--files", nargs="*", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    paths = args.files if args.files else iter_report_files(args.report_dir)
    changed = 0
    seen: set[Path] = set()
    for path in paths:
        if not path.is_absolute():
            path = ROOT / path
        if path in seen or not path.exists():
            continue
        seen.add(path)
        old = path.read_text(encoding="utf-8")
        new = clarify_html(old)
        if new != old:
            path.write_text(new, encoding="utf-8")
            changed += 1
            if args.verbose:
                print(f"clarified {path}")
    if args.verbose:
        print(f"changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
