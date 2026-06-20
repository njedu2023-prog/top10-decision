#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Apply the clean Premium report UI and keep table headers in English.'''

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / 'docs' / 'reports'

PREMIUM_STYLE = '''<style>
    :root{color-scheme:light;--page:#f5f5f7;--panel:#ffffff;--panel-soft:#fbfbfd;--ink:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--line-soft:#ececf0;--accent:#c4251a;--warn:#b26a00;--shadow:0 8px 24px rgba(0,0,0,.06)}
    *{box-sizing:border-box}
    html{scroll-behavior:smooth;background:var(--page)}
    body{margin:0;min-width:320px;font-family:-apple-system,BlinkMacSystemFont,SF Pro SC,SF Pro Text,PingFang SC,Microsoft YaHei,Segoe UI,Arial,sans-serif;color:var(--ink);background:var(--page);-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
    header{position:sticky;top:0;z-index:10;padding:18px 28px 14px;background:rgba(251,251,253,.88);border-bottom:1px solid rgba(210,210,215,.72);backdrop-filter:saturate(180%) blur(18px)}
    .topbar{display:flex;align-items:center;justify-content:space-between;gap:18px;max-width:1440px;margin:0 auto}
    .kicker{color:var(--muted);font-weight:600;font-size:12px;line-height:1.35;letter-spacing:0}
    h1{margin:5px 0 0;font-size:24px;line-height:1.18;letter-spacing:0;font-weight:700}
    .status-pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:7px 12px;color:var(--muted);font-size:13px;line-height:1;white-space:nowrap;background:rgba(255,255,255,.8)}
    .status-pill b{color:var(--ink)}
    main{padding:20px 28px 40px;max-width:1440px;margin:0 auto}
    .report-nav{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:0 0 16px;padding:10px;background:rgba(255,255,255,.72);border:1px solid var(--line-soft);border-radius:8px}
    .nav-actions,.date-chips,.tabs{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
    .nav-btn,.date-chip,.tab-btn{border:1px solid transparent;background:transparent;color:#1d1d1f;text-decoration:none;border-radius:999px;padding:8px 13px;font-size:13px;line-height:1;cursor:pointer;transition:background .18s ease,color .18s ease,border-color .18s ease}
    .nav-btn.nav-icon{width:42px;height:42px;min-width:42px;padding:0;display:inline-flex;align-items:center;justify-content:center;font-size:22px;font-weight:600}
    .nav-btn.nav-icon span{display:block;transform:translateY(-1px)}
    .nav-btn:hover,.date-chip:hover,.tab-btn:hover{background:#fff;border-color:var(--line-soft)}
    .nav-btn.primary,.date-chip.active,.tab-btn.active{color:#fff;background:#1d1d1f;border-color:#1d1d1f;font-weight:600}
    .nav-btn.disabled{color:#a1a1a6;background:transparent;cursor:not-allowed}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(236px,1fr));gap:12px;margin-bottom:16px}
    .metric{background:var(--panel);border:1px solid var(--line-soft);border-radius:8px;padding:16px 17px;min-height:104px;box-shadow:var(--shadow)}
    .metric-wide{grid-column:span 2}
    .metric span{display:block;color:var(--muted);font-size:11.5px;line-height:1.35;font-weight:600}
    .metric strong{display:block;margin-top:9px;color:var(--ink);font-size:21px;line-height:1.2;font-weight:700}
    .metric small{display:block;margin-top:8px;color:var(--muted);font-size:11.5px;line-height:1.42}
    .metric-verify-current small{color:#424245}
    .metric-lines{margin-top:10px;display:grid;gap:8px}
    .metric-line{display:grid;grid-template-columns:minmax(0,1fr) auto auto;align-items:center;gap:12px;padding:9px 0;border-top:1px solid var(--line-soft)}
    .metric-line:first-child{border-top:0;padding-top:0}
    .metric-line span{color:#424245;font-weight:600;line-height:1.3}
    .metric-line strong{margin:0;font-size:18px;color:var(--accent);font-variant-numeric:tabular-nums}
    .metric-line small{margin:0;color:var(--muted);font-size:12px;white-space:nowrap}
    .toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px;background:rgba(255,255,255,.86);border:1px solid var(--line-soft);border-radius:8px;margin-bottom:14px;box-shadow:0 4px 16px rgba(0,0,0,.03)}
    .hint{color:var(--muted);font-size:13px;line-height:1.35}
    section{background:var(--panel);border:1px solid var(--line-soft);border-radius:8px;margin-top:14px;overflow:hidden;box-shadow:var(--shadow)}
    section.hidden{display:none}
    .section-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:17px 18px;border-bottom:1px solid var(--line-soft);background:var(--panel)}
    h2{margin:0;font-size:18px;letter-spacing:0;font-weight:700}
    .badge{border:1px solid var(--line-soft);border-radius:999px;padding:5px 10px;font-size:12px;color:var(--muted);white-space:nowrap;background:var(--panel-soft)}
    .table-wrap{overflow:auto;width:100%;max-height:72vh}
    table{width:100%;border-collapse:collapse;min-width:1100px}
    th,td{padding:11px 13px;border-bottom:1px solid var(--line-soft);text-align:left;white-space:nowrap;font-size:13px;line-height:1.35;vertical-align:top}
    th{background:#f7f7fa;color:#55555b;font-weight:700;position:sticky;top:0;z-index:2}
    td{color:#1d1d1f}
    tbody tr:nth-child(even) td{background:#fbfbfd}
    tbody tr:hover td{background:#fff7f5}
    th:first-child,td:first-child{position:sticky;left:0;z-index:1;background:#fff}
    tbody tr:nth-child(even) td:first-child{background:#fbfbfd}
    tbody tr:hover td:first-child{background:#fff7f5}
    th:first-child{z-index:3}
    .num{font-variant-numeric:tabular-nums;font-weight:650}
    .good{color:var(--accent)}
    .mid{color:var(--warn)}
    .quiet{color:#5f6368}
    .explain{padding:16px 18px 18px;color:#424245;line-height:1.7}
    .explain div{padding:8px 0;border-bottom:1px solid var(--line-soft)}
    .explain div:last-of-type{border-bottom:0}
    .explain ul{margin:8px 0 0;padding-left:18px}
    .verify-table-wrap{overflow:auto;width:100%}
    .verify-table{min-width:0;width:100%;table-layout:auto}
    .verify-table th,.verify-table td{white-space:normal;font-size:13px;line-height:1.55;padding:12px 14px}
    .verify-table th{position:static;top:auto;width:260px;background:#f7f7fa;color:#55555b}
    .verify-table td{color:#424245;word-break:break-word}
    .verify-table th:first-child,.verify-table td:first-child{position:static;left:auto;background:inherit}
    .verify-table tbody tr:nth-child(even) th,.verify-table tbody tr:nth-child(even) td{background:#fbfbfd}
    .verify-table tbody tr:hover th,.verify-table tbody tr:hover td{background:#fff7f5}
    .empty{margin:0;padding:16px 18px;color:var(--muted)}
    .footnote{color:var(--muted);font-size:12px;margin:14px 0 0;line-height:1.5}
    @media(max-width:900px){header{position:static}header,main{padding-left:16px;padding-right:16px}.topbar,.report-nav,.toolbar{align-items:flex-start;flex-direction:column}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metric-wide{grid-column:1 / -1}h1{font-size:24px}}
    @media(max-width:560px){.metrics{grid-template-columns:1fr}.metric-line{grid-template-columns:1fr auto}.metric-line small{grid-column:1 / -1}.section-head{align-items:flex-start;flex-direction:column}}
  </style>'''

HEADER_MAP = {
    '排名': 'Rank', '代码': 'Code', '名称': 'Name', '姓名': 'Name',
    '板块': 'Sector', '部门': 'Sector', 'D日收盘': 'D Close', 'D 关': 'D Close',
    '分组': 'Bucket', '桶': 'Bucket', 'T涨停概率': 'T-Up',
    'T强度': 'T-Strength', 'T力量': 'T-Strength', 'T攻击': 'T-Attack',
    'T攻击分': 'T-Attack', 'T+1上涨概率': 'T1-Up', 'T+1承接分': 'T1-Accept',
    'T1-接受': 'T1-Accept', 'T+1接力分': 'T1-Relay', 'T1中继': 'T1-Relay',
    '总分': 'Score', '分数': 'Score', '门槛原因': 'Gate', '门': 'Gate',
    'T日竞价动作': 'T Auction Action', 'T 拍卖行动': 'T Auction Action',
    '价格': 'Price', 'T+1卖出计划': 'T+1 Sell Plan', 'T+1 销售计划': 'T+1 Sell Plan',
}

DATE_LABELS = {
    'd': ('D日分析日期', 'D Analysis Date'),
    't': ('T日竞价买入日期', 'T Auction Buy Date'),
    't1': ('T+1择时卖出日期', 'T+1 Timing Exit Date'),
}


def iter_report_files(report_dir: Path) -> list[Path]:
    return sorted(report_dir.glob('premium_*.html')) + sorted(report_dir.glob('premium_latest.html'))


def normalize_nav_arrows(text: str) -> str:
    replacements = [
        (
            r'<a class="nav-btn" href="([^"]+)">(Previous Report|上一份报告)</a>',
            r'<a class="nav-btn nav-icon" href="\1" aria-label="上一份报告" title="上一份报告"><span aria-hidden="true">&#8592;</span></a>',
        ),
        (
            r'<span class="nav-btn disabled">(Previous Report|上一份报告)</span>',
            r'<span class="nav-btn nav-icon disabled" aria-label="上一份报告" title="上一份报告"><span aria-hidden="true">&#8592;</span></span>',
        ),
        (
            r'<a class="nav-btn" href="([^"]+)">(Next Report|下一份报告)</a>',
            r'<a class="nav-btn nav-icon" href="\1" aria-label="下一份报告" title="下一份报告"><span aria-hidden="true">&#8594;</span></a>',
        ),
        (
            r'<span class="nav-btn disabled">(Next Report|下一份报告)</span>',
            r'<span class="nav-btn nav-icon disabled" aria-label="下一份报告" title="下一份报告"><span aria-hidden="true">&#8594;</span></span>',
        ),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text


def trim_long_decimals(text: str) -> str:
    return re.sub(r'(?<![\d.])(-?\d+\.\d{4})\d+', r'\1', text)


def _strip_tags(value: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', value or '')).strip()


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _extract_metric_value(text: str, labels: tuple[str, ...]) -> str:
    label_alt = '|'.join(re.escape(x) for x in labels)
    pattern = re.compile(
        r'<div class="metric[^\"]*">\s*<span>(?:' + label_alt + r')</span>\s*<strong>(.*?)</strong>',
        re.S,
    )
    match = pattern.search(text)
    return _strip_tags(match.group(1)) if match else '-'


def _extract_report_dates(text: str) -> tuple[str, str, str]:
    d = _extract_metric_value(text, DATE_LABELS['d'])
    t = _extract_metric_value(text, DATE_LABELS['t'])
    t1 = _extract_metric_value(text, DATE_LABELS['t1'])
    return d, t, t1


def _parse_hits(value: str) -> tuple[int, int]:
    match = re.search(r'(\d+)\s*/\s*(\d+)', _strip_tags(value))
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _parse_rate(value: str) -> str:
    plain = _strip_tags(value)
    match = re.search(r'(?:^|\s)(-|\d+(?:\.\d+)?%)', plain)
    return match.group(1) if match else '-'


def _truth_note(topn: int, rate: str, hits: int, total: int, d: str, t: str, t1: str) -> str:
    if total > 0 and rate != '-':
        return (
            f'TOP{topn}：D {d} -> T {t}，T日收盘涨停命中 {hits}/{total}，命中率 {rate}；'
            f'验证口径：T日收盘涨停=命中；T+1 {t1} 用于后续接力验证。'
        )
    return (
        f'TOP{topn}：D {d} -> T {t}，等待T日收盘后的市场真值，当前不计入命中率；'
        f'验证口径：T日收盘涨停=命中；T+1 {t1} 用于后续接力验证。'
    )


def _status_note(raw: str, d: str, t: str) -> str:
    plain = _strip_tags(raw)
    low = plain.lower()
    if any(x in low for x in ('t_daily_not_ready', 'not_ready', '404', '未就绪', '未找到')):
        return f'T日真值未就绪：D {d} 预测 T {t}，等待T日收盘后的市场真值数据；当前报告不计入实时命中率。'
    if plain in {'正常', 'ok'} or low == 'ok':
        return f'已验证：D {d} 预测 T {t}，已使用T日真实行情回填。'
    return plain


def enhance_current_limitup_truth(text: str) -> str:
    d, t, t1 = _extract_report_dates(text)

    current_card = re.compile(
        r'<div class="metric[^\"]*">\s*<span>(?:当前TOP10涨停命中率|Current TOP10 Hit Rate)</span>'
        r'\s*<strong>(.*?)</strong>\s*<small>(.*?)</small>\s*</div>',
        re.S,
    )

    def card_repl(match: re.Match[str]) -> str:
        value = _parse_rate(match.group(1))
        hits, total = _parse_hits(match.group(2))
        note = _truth_note(10, value, hits, total, d, t, t1)
        return (
            '<div class="metric metric-wide metric-verify-current">'
            '<span>当前TOP10涨停命中率</span>'
            f'<strong>{_esc(value)}</strong><small>{_esc(note)}</small></div>'
        )

    text = current_card.sub(card_repl, text, count=1)

    row_specs = [
        (10, ('当前TOP10涨停预测命中率', 'Current TOP10 limit-up prediction hit rate')),
        (20, ('当前TOP20涨停预测命中率', 'Current TOP20 limit-up prediction hit rate')),
    ]
    for topn, labels in row_specs:
        label_alt = '|'.join(re.escape(x) for x in labels)
        row_pattern = re.compile(r'(<tr><th>(?:' + label_alt + r')</th><td>)(.*?)(</td></tr>)', re.S)

        def row_repl(match: re.Match[str], n: int = topn) -> str:
            value = _parse_rate(match.group(2))
            hits, total = _parse_hits(match.group(2))
            return f'{match.group(1)}{_esc(_truth_note(n, value, hits, total, d, t, t1))}{match.group(3)}'

        text = row_pattern.sub(row_repl, text)

    status_pattern = re.compile(r'(<tr><th>(?:验证状态|Validation status)</th><td>)(.*?)(</td></tr>)', re.S)
    text = status_pattern.sub(lambda m: f'{m.group(1)}{_esc(_status_note(m.group(2), d, t))}{m.group(3)}', text)
    return text


def convert_validation_panel_to_table(text: str) -> str:
    pattern = re.compile(
        r'(<section id="verify-panel"[^>]*>.*?<div class="explain">)(.*?)(</div>\s*</section>)',
        re.S,
    )

    def repl(match: re.Match[str]) -> str:
        body = match.group(2)
        if 'verify-table' in body:
            return match.group(0)

        rows: list[tuple[str, str]] = []
        for item in re.findall(r'<div>(.*?)</div>', body, flags=re.S):
            item = re.sub(r'\s+', ' ', item).strip()
            if not item:
                continue
            if '：' in item:
                label, value = item.split('：', 1)
            elif ':' in item:
                label, value = item.split(':', 1)
            else:
                label, value = '说明', item
            rows.append((label.strip(), value.strip()))

        for li in re.findall(r'<li>(.*?)</li>', body, flags=re.S):
            li = re.sub(r'\s+', ' ', li).strip()
            if li:
                rows.append(('备注', li))

        if not rows:
            return match.group(0)

        body_rows = ''.join(f'<tr><th>{label}</th><td>{value}</td></tr>' for label, value in rows)
        table = (
            '<div class="verify-table-wrap"><table class="verify-table">'
            '<thead><tr><th>项目</th><th>内容</th></tr></thead>'
            f'<tbody>{body_rows}</tbody></table></div>'
        )
        return f'{match.group(1)}{table}{match.group(3)}'

    return pattern.sub(repl, text)


def restyle_html(text: str) -> str:
    text = re.sub(r'<html[^>]*>', '<html lang=\"zh-CN\">', text, count=1, flags=re.I)
    text = re.sub(r'<style>.*?</style>', PREMIUM_STYLE, text, count=1, flags=re.S)
    for old, new in HEADER_MAP.items():
        text = text.replace(f'<th>{old}</th>', f'<th>{new}</th>')
    text = normalize_nav_arrows(text)
    text = trim_long_decimals(text)
    text = convert_validation_panel_to_table(text)
    return enhance_current_limitup_truth(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-dir', type=Path, default=REPORT_DIR)
    parser.add_argument('--files', nargs='*', type=Path, default=None)
    parser.add_argument('--verbose', action='store_true')
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
        old = path.read_text(encoding='utf-8')
        new = restyle_html(old)
        if new != old:
            path.write_text(new, encoding='utf-8')
            changed += 1
            if args.verbose:
                print(f'styled {path}')
    if args.verbose:
        print(f'changed={changed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
