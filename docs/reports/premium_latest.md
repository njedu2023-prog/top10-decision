# Premium（溢价预测）V3（E_ret_plus / Close[T+2] 分布预测）

> 注：T 为本次预测的**基准交易日**（使用 Close[T]）；T+2 为**预测到期交易日**（预测 Close[T+2] 的分布）。

- 预测日（T）：**20260609**
- 预测到期日（T+2）：**20260611**
- 周期：**2 个交易日（T→T+2）**
- 生成时间：2026-06-11T17:03:05Z
- 模型版本：**premium_v2**

## 预测表（Top30）

<div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>操作排名</th>
      <th>代码</th>
      <th>名称</th>
      <th>收盘价</th>
      <th>T+2预期收益</th>
      <th>预期价格区间</th>
      <th>T+2上涨概率</th>
      <th>模型置信度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>002263.SZ</td>
      <td>大东南</td>
      <td>4.03</td>
      <td>+10.94%</td>
      <td>4.32 ~ 4.62，中位 4.47</td>
      <td>15.62%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>2</td>
      <td>000509.SZ</td>
      <td>华塑控股</td>
      <td>4.80</td>
      <td>+10.35%</td>
      <td>5.12 ~ 5.48，中位 5.30</td>
      <td>14.13%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>3</td>
      <td>002225.SZ</td>
      <td>濮耐股份</td>
      <td>4.58</td>
      <td>+10.35%</td>
      <td>4.89 ~ 5.23，中位 5.05</td>
      <td>5.36%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>4</td>
      <td>600611.SH</td>
      <td>大众交通</td>
      <td>4.48</td>
      <td>+9.67%</td>
      <td>4.75 ~ 5.08，中位 4.91</td>
      <td>11.97%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>5</td>
      <td>600226.SH</td>
      <td>亨通股份</td>
      <td>7.43</td>
      <td>+9.14%</td>
      <td>7.84 ~ 8.39，中位 8.11</td>
      <td>22.71%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>6</td>
      <td>002515.SZ</td>
      <td>金字火腿</td>
      <td>7.36</td>
      <td>+9.14%</td>
      <td>7.77 ~ 8.31，中位 8.03</td>
      <td>29.40%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>7</td>
      <td>600487.SH</td>
      <td>亨通光电</td>
      <td>105.02</td>
      <td>+8.91%</td>
      <td>110.59 ~ 118.30，中位 114.38</td>
      <td>13.06%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>8</td>
      <td>688141.SH</td>
      <td>杰华特</td>
      <td>103.70</td>
      <td>+8.91%</td>
      <td>109.20 ~ 116.82，中位 112.94</td>
      <td>12.27%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>9</td>
      <td>002484.SZ</td>
      <td>江海股份</td>
      <td>91.98</td>
      <td>+8.91%</td>
      <td>96.86 ~ 103.62，中位 100.18</td>
      <td>5.78%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>10</td>
      <td>600183.SH</td>
      <td>生益科技</td>
      <td>147.47</td>
      <td>+8.89%</td>
      <td>155.26 ~ 166.09，中位 160.58</td>
      <td>17.77%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>11</td>
      <td>002585.SZ</td>
      <td>双星新材</td>
      <td>12.20</td>
      <td>+8.75%</td>
      <td>12.83 ~ 13.72，中位 13.27</td>
      <td>2.69%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>12</td>
      <td>002051.SZ</td>
      <td>中工国际</td>
      <td>12.23</td>
      <td>+8.75%</td>
      <td>12.86 ~ 13.76，中位 13.30</td>
      <td>11.40%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>13</td>
      <td>603020.SH</td>
      <td>爱普股份</td>
      <td>11.64</td>
      <td>+8.75%</td>
      <td>12.24 ~ 13.09，中位 12.66</td>
      <td>3.16%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>14</td>
      <td>600110.SH</td>
      <td>诺德股份</td>
      <td>11.81</td>
      <td>+8.75%</td>
      <td>12.42 ~ 13.28，中位 12.84</td>
      <td>7.54%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>15</td>
      <td>000037.SZ</td>
      <td>深南电A</td>
      <td>11.39</td>
      <td>+8.62%</td>
      <td>11.96 ~ 12.80，中位 12.37</td>
      <td>15.84%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>16</td>
      <td>600186.SH</td>
      <td>莲花控股</td>
      <td>11.54</td>
      <td>+8.62%</td>
      <td>12.12 ~ 12.96，中位 12.53</td>
      <td>11.37%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>17</td>
      <td>605069.SH</td>
      <td>正和生态</td>
      <td>11.34</td>
      <td>+8.62%</td>
      <td>11.91 ~ 12.74，中位 12.32</td>
      <td>30.22%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>18</td>
      <td>002141.SZ</td>
      <td>贤丰控股</td>
      <td>3.33</td>
      <td>+8.40%</td>
      <td>3.49 ~ 3.73，中位 3.61</td>
      <td>18.68%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>19</td>
      <td>002584.SZ</td>
      <td>西陇科学</td>
      <td>7.88</td>
      <td>+8.33%</td>
      <td>8.25 ~ 8.83，中位 8.54</td>
      <td>2.28%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>20</td>
      <td>603330.SH</td>
      <td>天洋新材</td>
      <td>8.53</td>
      <td>+8.33%</td>
      <td>8.93 ~ 9.56，中位 9.24</td>
      <td>8.61%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>21</td>
      <td>002769.SZ</td>
      <td>普路通</td>
      <td>8.47</td>
      <td>+8.33%</td>
      <td>8.87 ~ 9.49，中位 9.18</td>
      <td>15.07%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>22</td>
      <td>002354.SZ</td>
      <td>天娱数科</td>
      <td>8.06</td>
      <td>+8.33%</td>
      <td>8.44 ~ 9.03，中位 8.73</td>
      <td>31.37%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>23</td>
      <td>600719.SH</td>
      <td>大连热电</td>
      <td>9.05</td>
      <td>+8.24%</td>
      <td>9.47 ~ 10.13，中位 9.80</td>
      <td>10.16%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>24</td>
      <td>603065.SH</td>
      <td>宿迁联盛</td>
      <td>10.40</td>
      <td>+8.24%</td>
      <td>10.88 ~ 11.64，中位 11.26</td>
      <td>85.77%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>25</td>
      <td>600903.SH</td>
      <td>贵州燃气</td>
      <td>9.14</td>
      <td>+8.24%</td>
      <td>9.56 ~ 10.23，中位 9.89</td>
      <td>8.04%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>26</td>
      <td>300411.SZ</td>
      <td>金盾股份</td>
      <td>10.00</td>
      <td>+8.24%</td>
      <td>10.46 ~ 11.20，中位 10.82</td>
      <td>15.07%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>27</td>
      <td>002129.SZ</td>
      <td>TCL中环</td>
      <td>9.81</td>
      <td>+8.24%</td>
      <td>10.27 ~ 10.98，中位 10.62</td>
      <td>20.53%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>28</td>
      <td>002137.SZ</td>
      <td>实益达</td>
      <td>10.79</td>
      <td>+8.24%</td>
      <td>11.29 ~ 12.08，中位 11.68</td>
      <td>12.10%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>29</td>
      <td>002057.SZ</td>
      <td>中钢天源</td>
      <td>10.74</td>
      <td>+8.24%</td>
      <td>11.24 ~ 12.02，中位 11.62</td>
      <td>7.58%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>30</td>
      <td>600198.SH</td>
      <td>大唐电信</td>
      <td>9.34</td>
      <td>+8.24%</td>
      <td>9.77 ~ 10.46，中位 10.11</td>
      <td>6.55%</td>
      <td>高（1.000）</td>
    </tr>
  </tbody>
</table></div>

## 验证表（Top30）

<div><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>操作排名</th>
      <th>代码</th>
      <th>名称</th>
      <th>收盘价</th>
      <th>T+2预期收益</th>
      <th>预期价格区间</th>
      <th>T+2上涨概率</th>
      <th>模型置信度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>002263.SZ</td>
      <td>大东南</td>
      <td>4.03</td>
      <td>+10.94%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>2</td>
      <td>000509.SZ</td>
      <td>华塑控股</td>
      <td>4.80</td>
      <td>+10.35%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>3</td>
      <td>002225.SZ</td>
      <td>濮耐股份</td>
      <td>4.58</td>
      <td>+10.35%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>4</td>
      <td>600611.SH</td>
      <td>大众交通</td>
      <td>4.48</td>
      <td>+9.67%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>5</td>
      <td>600226.SH</td>
      <td>亨通股份</td>
      <td>7.43</td>
      <td>+9.14%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>6</td>
      <td>002515.SZ</td>
      <td>金字火腿</td>
      <td>7.36</td>
      <td>+9.14%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>7</td>
      <td>600487.SH</td>
      <td>亨通光电</td>
      <td>105.02</td>
      <td>+8.91%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>8</td>
      <td>688141.SH</td>
      <td>杰华特</td>
      <td>103.70</td>
      <td>+8.91%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>9</td>
      <td>002484.SZ</td>
      <td>江海股份</td>
      <td>91.98</td>
      <td>+8.91%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>10</td>
      <td>600183.SH</td>
      <td>生益科技</td>
      <td>147.47</td>
      <td>+8.89%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>11</td>
      <td>002585.SZ</td>
      <td>双星新材</td>
      <td>12.20</td>
      <td>+8.75%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>12</td>
      <td>002051.SZ</td>
      <td>中工国际</td>
      <td>12.23</td>
      <td>+8.75%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>13</td>
      <td>603020.SH</td>
      <td>爱普股份</td>
      <td>11.64</td>
      <td>+8.75%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>14</td>
      <td>600110.SH</td>
      <td>诺德股份</td>
      <td>11.81</td>
      <td>+8.75%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>15</td>
      <td>000037.SZ</td>
      <td>深南电A</td>
      <td>11.39</td>
      <td>+8.62%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>16</td>
      <td>600186.SH</td>
      <td>莲花控股</td>
      <td>11.54</td>
      <td>+8.62%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>17</td>
      <td>605069.SH</td>
      <td>正和生态</td>
      <td>11.34</td>
      <td>+8.62%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>18</td>
      <td>002141.SZ</td>
      <td>贤丰控股</td>
      <td>3.33</td>
      <td>+8.40%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>19</td>
      <td>002584.SZ</td>
      <td>西陇科学</td>
      <td>7.88</td>
      <td>+8.33%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>20</td>
      <td>603330.SH</td>
      <td>天洋新材</td>
      <td>8.53</td>
      <td>+8.33%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>21</td>
      <td>002769.SZ</td>
      <td>普路通</td>
      <td>8.47</td>
      <td>+8.33%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>22</td>
      <td>002354.SZ</td>
      <td>天娱数科</td>
      <td>8.06</td>
      <td>+8.33%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>23</td>
      <td>600719.SH</td>
      <td>大连热电</td>
      <td>9.05</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>24</td>
      <td>603065.SH</td>
      <td>宿迁联盛</td>
      <td>10.40</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>25</td>
      <td>600903.SH</td>
      <td>贵州燃气</td>
      <td>9.14</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>26</td>
      <td>300411.SZ</td>
      <td>金盾股份</td>
      <td>10.00</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>27</td>
      <td>002129.SZ</td>
      <td>TCL中环</td>
      <td>9.81</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>28</td>
      <td>002137.SZ</td>
      <td>实益达</td>
      <td>10.79</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>29</td>
      <td>002057.SZ</td>
      <td>中钢天源</td>
      <td>10.74</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>30</td>
      <td>600198.SH</td>
      <td>大唐电信</td>
      <td>9.34</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
  </tbody>
</table></div>

- 命中（旧口径：actual_ret>0）：18/30（60.00%）
- 覆盖率（V2：in_p10，r_actual ∈ [p05,p95]）：53.33%
- 覆盖率（V2：in_p50，r_actual ∈ [p25,p75]）：6.67%
- MAE（V2：|err_r_p50|）：0.097983
- MAE（V2：|err_close_p50|）：2.461740

## E_ret_plus / EHX 验证摘要

- Raw MAE：5.4812%
- Plus MAE：10.0787%
- MAE 改善：-4.5975%
- Plus 优于 Raw 比例：17.24%

## 字段说明（V3 人类操作口径）

- 操作排名：优先使用 E_ret_plus 排名。
- T+2预期收益：EHX 残差增强后的 E_ret_plus。
- 预期价格区间：使用 T+2 价格分位 p25 ~ p75，并展示 p50 中位价。
- T+2上涨概率：预测到期日 T+2 收盘上涨概率。
- 模型置信度：EHX 置信度标签及置信分。

> 注：E_ret原始值、EHX修正值、EHX来源、Raw/Plus误差等工程审计字段仍保留在 CSV 与验证摘要中，主表不再展开展示。

---
## 审计（Factor Packs / Degrade）
- degrade_mode: **full**
- packs_used: `Pack0, Pack1, Pack2`
- packs_missing: `-`

### notes
- 01. Pack0 baseline always on
- 02. Pack1 enabled: market cache found -> /home/runner/work/top10-decision/top10-decision/data/market/daily_20260609.csv
- 03. Pack2 enabled (soft mode, fixed)

