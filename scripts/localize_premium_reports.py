#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Post-process Premium HTML reports for zh-CN pages with English table headers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "reports"

TEXT_REPLACEMENTS = [
    ('<html lang="en">', '<html lang="zh-CN">'),
    ('Premium V4 Quant Engine', 'Premium V4 量化引擎'),
    ('Validation <b>PENDING</b>', '验证状态 <b>待验证</b>'),
    ('Validation <b>READY</b>', '验证状态 <b>已验证</b>'),
    ('>PENDING<', '>待验证<'),
    ('>READY<', '>已验证<'),
    ('Report navigation', '报告导航'),
    ('Previous Report', '上一份报告'),
    ('Latest Report', '最新报告'),
    ('Next Report', '下一份报告'),
    ('D Analysis Date', 'D日分析日期'),
    ('Uses post-close data from D only', '只使用D日收盘后可见数据'),
    ('T Auction Buy Date', 'T日竞价买入日期'),
    ('Strict A-share trading calendar', '严格按中国A股交易日历'),
    ('T+1 Timing Exit Date', 'T+1择时卖出日期'),
    ('Continuation and validation date', '接力验证和择时卖出日期'),
    ('Current TOP10 Hit Rate', '当前TOP10涨停命中率'),
    ('Historical TOP10 Limit-up Hit Rate', '历史TOP10累计涨停命中率'),
    ('Head Historical Limit-up Hit Rate', '头部历史累计涨停命中率'),
    ('TOP1 Historical Limit-up Hit Rate', 'TOP1 历史累计涨停命中率'),
    ('TOP3 Historical Limit-up Hit Rate', 'TOP3 历史累计涨停命中率'),
    ('TOP5 Historical Limit-up Hit Rate', 'TOP5 历史累计涨停命中率'),
    ('Head Historical Up Rate', '头部历史累计上涨率'),
    ('TOP1 Historical Up Rate', 'TOP1 历史累计上涨率'),
    ('TOP3 Historical Up Rate', 'TOP3 历史累计上涨率'),
    ('TOP5 Historical Up Rate', 'TOP5 历史累计上涨率'),
    ('T close &gt; D close', 'T日收盘&gt;D日收盘'),
    ('20D TOP10 Hit Rate', '近20日TOP10命中率'),
    ('5D ', '近5日 '),
    ('60D ', '近60日 '),
    ('20D ', '近20日 '),
    ('Probability Calibration Quality', '概率校准质量'),
    ('Limit-up Rank IC', '涨停Rank IC'),
    ('T+1 Return Rank IC', 'T+1收益Rank IC'),
    ('Adaptive Ranking Weights', '自适应排序权重'),
    ('Professional Gate', '专业门槛'),
    ('Tier Effectiveness', '分层有效性'),
    ('D Market Sentiment', 'D日市场情绪'),
    ('List switcher', '列表切换'),
    ('TOP10 Execution List', 'TOP10 执行名单'),
    ('TOP20 Watch List', 'TOP20 观察名单'),
    ('Validation & Learning', '验证与学习'),
    ('TOP10: Highest T-day Limit-up Probability', 'TOP10: T日涨停概率最高'),
    ('Core Execution List', '核心执行名单'),
    ('TOP20: T+1 Continuation Candidates', 'TOP20: T+1接力观察候选'),
    ('Extended Watch List', '扩展观察名单'),
    ('Validation status:', '验证状态：'),
    ('Current TOP10 limit-up prediction hit rate:', '当前TOP10涨停预测命中率：'),
    ('Current TOP20 limit-up prediction hit rate:', '当前TOP20涨停预测命中率：'),
    ('Historical TOP1 / TOP3 / TOP5 cumulative limit-up hit rate:', '历史TOP1 / TOP3 / TOP5累计涨停命中率：'),
    ('Historical TOP1 / TOP3 / TOP5 cumulative up rate:', '历史TOP1 / TOP3 / TOP5累计上涨率：'),
    ('Historical TOP10 cumulative limit-up hit rate:', '历史TOP10累计涨停命中率：'),
    ('Historical TOP20 cumulative limit-up hit rate:', '历史TOP20累计涨停命中率：'),
    ('Rolling TOP10 hit rate:', '滚动TOP10命中率：'),
    ('Probability calibration:', '概率校准：'),
    ('Adaptive ranking weights:', '自适应排序权重：'),
    ('T+1 weight is reduced when T+1 Rank IC is negative, then increases gradually after it turns positive.', '当T+1 Rank IC为负时，系统会降低T+1权重；转正后再逐步提高。'),
    ('Tier hit/return summary:', '分层命中/收益摘要：'),
    ('D-day market sentiment:', 'D日市场情绪：'),
    ('Historical statistics sample:', '历史统计样本：'),
    ('Model version:', '模型版本：'),
    ('generated at:', '生成时间：'),
    ('This report is generated automatically by Premium. Previous/next navigation is based on historical HTML report dates already present in the repository.', '本报告由 Premium 自动生成。上一页/下一页导航基于仓库中已有的历史HTML报告日期。'),
    ('>ELIGIBLE<', '>合格池<'),
    ('>WATCH<', '>观察池<'),
    ('>EXCLUDED<', '>排除池<'),
    ('>ok<', '>正常<'),
    ('no_ready_limitup_rows', '没有可验证的涨停样本'),
    ('verify_table_empty', '验证表为空'),
    ('limitup_actual_missing', '缺少涨停真值字段'),
    ('t_daily_not_ready', 'T日行情未就绪'),
    ('t_daily_missing_ts_code', 'T日行情缺少代码字段'),
    ('missing_D_or_T_price', '缺少D日或T日价格'),
    ('professional_score_rule_guarded', '专业评分规则保护模式'),
    ('model_validated_professional_score', '模型验证通过排序模式'),
    ('disabled_validation_not_pass', '模型验证未通过，未接管排序'),
    ('rank_enabled_validation_pass', '模型验证通过，可接管排序'),
    ('fetch_remote_failed', '远端拉取失败'),
    ('http_get_failed', 'HTTP请求失败'),
    ('body_snippet', '响应片段'),
    ('Not Found', '未找到'),
    ('archive_rebuilt', '归档已重建'),
    ('history_source', '历史样本来源'),
    ('truth_refresh', '真值刷新'),
    ('; rank ', '；排序 '),
    ('; model ', '；模型 '),
    ('Excluded ', '排除 '),
    (' eligible / ', ' 合格 / '),
    (' watch', ' 观察'),
    ('Limit-up ', '涨停 '),
    ('limit-up ', '涨停 '),
    ('Strength ', '强度 '),
    ('strength ', '强度 '),
    ('Execution ', '执行 '),
    ('execution ', '执行 '),
    ('Up ratio ', '上涨占比 '),
    ('up ratio ', '上涨占比 '),
    ('strong stocks ', '强势股 '),
    ('sentiment ', '情绪 '),
    ('samples ', '样本 '),
    ('samples=', '样本='),
    ('days=', '天数='),
    ('valid trading days ', '有效交易日 '),
    ('valid days ', '有效天数 '),
    ('trading days ', '交易日 '),
    ('positive rate ', '正值率 '),
    ('Spearman mean ', 'Spearman均值 '),
    ('source:', '来源：'),
    ('status:', '状态：'),
    ('status=', '状态='),
    ('ret=', '收益='),
    ('url=', 'URL='),
    ('True', '是'),
    ('False', '否'),
]

TABLE_HEADERS_EN_TO_CN = {
    'Rank': '排名',
    'Code': '代码',
    'Name': '名称',
    'Sector': '板块',
    'D Close': 'D收盘',
    'Bucket': '候选池',
    'T-Up': 'T涨停概率',
    'T-Strength': 'T强度',
    'T-Attack': 'T日攻击力',
    'T1-Up': 'T+1上涨概率',
    'T1-Accept': 'T+1承接力',
    'T1-Relay': '接力综合分',
    'Score': '综合分',
    'Gate': '筛选原因',
    'T Auction Action': 'T竞价动作',
    'Price': '买入价',
    'T+1 Sell Plan': 'T+1卖出计划',
}


def localize_html(text: str) -> str:
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    for old, new in TABLE_HEADERS_EN_TO_CN.items():
        text = text.replace(f'<th>{old}</th>', f'<th>{new}</th>')
    text = re.sub(r'(\d+) days', r'\1日', text)
    text = text.replace(':天数=', '：天数=').replace(';样本=', '；样本=').replace(';t1_', '；t1_')
    text = text.replace(', 没有', '，没有').replace(', 交易日', '，交易日').replace(', T日', '，T日')
    text = text.replace('涨停Rank IC:', '涨停Rank IC：').replace('T+1收益Rank IC:', 'T+1收益Rank IC：')
    text = text.replace('状态： ok', '状态： 正常')
    text = text.replace('<title>Premium 涨停 Relay Forecast', '<title>Premium TOP 10')
    text = text.replace('<title>Premium Limit-up Relay Forecast', '<title>Premium TOP 10')
    text = text.replace('<h1>Premium 涨停 Relay TOP10 / TOP20</h1>', '<h1>Premium TOP 10</h1>')
    text = text.replace('<h1>Premium Limit-up Relay TOP10 / TOP20</h1>', '<h1>Premium TOP 10</h1>')
    return text


def iter_report_files(report_dir: Path) -> list[Path]:
    return sorted(report_dir.glob('premium_*.html')) + sorted(report_dir.glob('premium_latest.html'))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-dir', type=Path, default=REPORT_DIR)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    changed = 0
    seen: set[Path] = set()
    for path in iter_report_files(args.report_dir):
        if path in seen or not path.exists():
            continue
        seen.add(path)
        old = path.read_text(encoding='utf-8')
        new = localize_html(old)
        if new != old:
            path.write_text(new, encoding='utf-8')
            changed += 1
            if args.verbose:
                print(f'localized {path}')
    if args.verbose:
        print(f'changed={changed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
