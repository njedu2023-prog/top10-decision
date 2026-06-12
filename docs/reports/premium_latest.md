# Premium（溢价预测）V3（E_ret_plus / Close[T+2] 分布预测）

> 注：T 为本次预测的**基准交易日**（使用 Close[T]）；T+2 为**预测到期交易日**（预测 Close[T+2] 的分布）。

- 预测日（T）：**20260609**
- 预测到期日（T+2）：**20260611**
- 周期：**2 个交易日（T→T+2）**
- 生成时间：2026-06-12T12:33:44Z
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
      <td>600186.SH</td>
      <td>莲花控股</td>
      <td>11.54</td>
      <td>+10.27%</td>
      <td>12.30 ~ 13.16，中位 12.72</td>
      <td>11.37%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>2</td>
      <td>002051.SZ</td>
      <td>中工国际</td>
      <td>12.23</td>
      <td>+10.27%</td>
      <td>13.04 ~ 13.95，中位 13.49</td>
      <td>11.40%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>3</td>
      <td>600110.SH</td>
      <td>诺德股份</td>
      <td>11.81</td>
      <td>+10.27%</td>
      <td>12.59 ~ 13.47，中位 13.02</td>
      <td>7.54%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>4</td>
      <td>603020.SH</td>
      <td>爱普股份</td>
      <td>11.64</td>
      <td>+10.27%</td>
      <td>12.41 ~ 13.28，中位 12.84</td>
      <td>3.16%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>5</td>
      <td>002585.SZ</td>
      <td>双星新材</td>
      <td>12.20</td>
      <td>+10.27%</td>
      <td>13.01 ~ 13.91，中位 13.45</td>
      <td>2.69%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>6</td>
      <td>605069.SH</td>
      <td>正和生态</td>
      <td>11.34</td>
      <td>+9.85%</td>
      <td>12.04 ~ 12.88，中位 12.46</td>
      <td>30.22%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>7</td>
      <td>000037.SZ</td>
      <td>深南电A</td>
      <td>11.39</td>
      <td>+9.85%</td>
      <td>12.10 ~ 12.94，中位 12.51</td>
      <td>15.84%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>8</td>
      <td>600183.SH</td>
      <td>生益科技</td>
      <td>147.47</td>
      <td>+8.79%</td>
      <td>155.12 ~ 165.94，中位 160.44</td>
      <td>17.77%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>9</td>
      <td>600487.SH</td>
      <td>亨通光电</td>
      <td>105.02</td>
      <td>+8.61%</td>
      <td>110.28 ~ 117.97，中位 114.06</td>
      <td>13.06%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>10</td>
      <td>688141.SH</td>
      <td>杰华特</td>
      <td>103.70</td>
      <td>+8.61%</td>
      <td>108.89 ~ 116.49，中位 112.63</td>
      <td>12.27%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>11</td>
      <td>000509.SZ</td>
      <td>华塑控股</td>
      <td>4.80</td>
      <td>+8.54%</td>
      <td>5.04 ~ 5.39，中位 5.21</td>
      <td>14.13%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>12</td>
      <td>002225.SZ</td>
      <td>濮耐股份</td>
      <td>4.58</td>
      <td>+8.54%</td>
      <td>4.81 ~ 5.14，中位 4.97</td>
      <td>5.36%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>13</td>
      <td>002263.SZ</td>
      <td>大东南</td>
      <td>4.03</td>
      <td>+8.50%</td>
      <td>4.23 ~ 4.52，中位 4.37</td>
      <td>15.62%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>14</td>
      <td>000158.SZ</td>
      <td>常山北明</td>
      <td>15.30</td>
      <td>+8.35%</td>
      <td>16.03 ~ 17.15，中位 16.58</td>
      <td>11.80%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>15</td>
      <td>600458.SH</td>
      <td>时代新材</td>
      <td>15.73</td>
      <td>+8.35%</td>
      <td>16.48 ~ 17.63，中位 17.04</td>
      <td>22.13%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>16</td>
      <td>003036.SZ</td>
      <td>泰坦股份</td>
      <td>81.95</td>
      <td>+8.33%</td>
      <td>85.83 ~ 91.82，中位 88.78</td>
      <td>8.97%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>17</td>
      <td>002859.SZ</td>
      <td>洁美科技</td>
      <td>88.33</td>
      <td>+8.26%</td>
      <td>92.46 ~ 98.91，中位 95.63</td>
      <td>9.24%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>18</td>
      <td>000920.SZ</td>
      <td>沃顿科技</td>
      <td>12.98</td>
      <td>+8.18%</td>
      <td>13.58 ~ 14.52，中位 14.04</td>
      <td>21.44%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>19</td>
      <td>002515.SZ</td>
      <td>金字火腿</td>
      <td>7.36</td>
      <td>+7.89%</td>
      <td>7.68 ~ 8.21，中位 7.94</td>
      <td>29.40%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>20</td>
      <td>603335.SH</td>
      <td>迪生力</td>
      <td>7.68</td>
      <td>+7.89%</td>
      <td>8.01 ~ 8.57，中位 8.29</td>
      <td>10.29%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>21</td>
      <td>600226.SH</td>
      <td>亨通股份</td>
      <td>7.43</td>
      <td>+7.89%</td>
      <td>7.75 ~ 8.29，中位 8.02</td>
      <td>22.71%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>22</td>
      <td>002354.SZ</td>
      <td>天娱数科</td>
      <td>8.06</td>
      <td>+7.89%</td>
      <td>8.41 ~ 8.99，中位 8.70</td>
      <td>31.37%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>23</td>
      <td>002584.SZ</td>
      <td>西陇科学</td>
      <td>7.88</td>
      <td>+7.89%</td>
      <td>8.22 ~ 8.79，中位 8.50</td>
      <td>2.28%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>24</td>
      <td>002484.SZ</td>
      <td>江海股份</td>
      <td>91.98</td>
      <td>+7.83%</td>
      <td>95.90 ~ 102.59，中位 99.19</td>
      <td>5.78%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>25</td>
      <td>600596.SH</td>
      <td>新安股份</td>
      <td>14.94</td>
      <td>+7.71%</td>
      <td>15.56 ~ 16.64，中位 16.09</td>
      <td>11.18%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>26</td>
      <td>001369.SZ</td>
      <td>双欣材料</td>
      <td>14.88</td>
      <td>+7.71%</td>
      <td>15.50 ~ 16.58，中位 16.03</td>
      <td>8.21%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>27</td>
      <td>605318.SH</td>
      <td>法狮龙</td>
      <td>105.96</td>
      <td>+7.65%</td>
      <td>110.29 ~ 117.98，中位 114.07</td>
      <td>9.35%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>28</td>
      <td>603163.SH</td>
      <td>圣晖集成</td>
      <td>106.54</td>
      <td>+7.65%</td>
      <td>110.89 ~ 118.63，中位 114.69</td>
      <td>15.51%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>29</td>
      <td>600903.SH</td>
      <td>贵州燃气</td>
      <td>9.14</td>
      <td>+7.51%</td>
      <td>9.50 ~ 10.16，中位 9.83</td>
      <td>8.04%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>30</td>
      <td>603065.SH</td>
      <td>宿迁联盛</td>
      <td>10.40</td>
      <td>+7.51%</td>
      <td>10.81 ~ 11.57，中位 11.18</td>
      <td>85.77%</td>
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
      <td>600186.SH</td>
      <td>莲花控股</td>
      <td>11.54</td>
      <td>+10.27%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>2</td>
      <td>002051.SZ</td>
      <td>中工国际</td>
      <td>12.23</td>
      <td>+10.27%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>3</td>
      <td>600110.SH</td>
      <td>诺德股份</td>
      <td>11.81</td>
      <td>+10.27%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>4</td>
      <td>603020.SH</td>
      <td>爱普股份</td>
      <td>11.64</td>
      <td>+10.27%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>5</td>
      <td>002585.SZ</td>
      <td>双星新材</td>
      <td>12.20</td>
      <td>+10.27%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>6</td>
      <td>605069.SH</td>
      <td>正和生态</td>
      <td>11.34</td>
      <td>+9.85%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>7</td>
      <td>000037.SZ</td>
      <td>深南电A</td>
      <td>11.39</td>
      <td>+9.85%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>8</td>
      <td>600183.SH</td>
      <td>生益科技</td>
      <td>147.47</td>
      <td>+8.79%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>9</td>
      <td>600487.SH</td>
      <td>亨通光电</td>
      <td>105.02</td>
      <td>+8.61%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>10</td>
      <td>688141.SH</td>
      <td>杰华特</td>
      <td>103.70</td>
      <td>+8.61%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>11</td>
      <td>000509.SZ</td>
      <td>华塑控股</td>
      <td>4.80</td>
      <td>+8.54%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>12</td>
      <td>002225.SZ</td>
      <td>濮耐股份</td>
      <td>4.58</td>
      <td>+8.54%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>13</td>
      <td>002263.SZ</td>
      <td>大东南</td>
      <td>4.03</td>
      <td>+8.50%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>14</td>
      <td>000158.SZ</td>
      <td>常山北明</td>
      <td>15.30</td>
      <td>+8.35%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>15</td>
      <td>600458.SH</td>
      <td>时代新材</td>
      <td>15.73</td>
      <td>+8.35%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>16</td>
      <td>003036.SZ</td>
      <td>泰坦股份</td>
      <td>81.95</td>
      <td>+8.33%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>17</td>
      <td>002859.SZ</td>
      <td>洁美科技</td>
      <td>88.33</td>
      <td>+8.26%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>18</td>
      <td>000920.SZ</td>
      <td>沃顿科技</td>
      <td>12.98</td>
      <td>+8.18%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>19</td>
      <td>002515.SZ</td>
      <td>金字火腿</td>
      <td>7.36</td>
      <td>+7.89%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>20</td>
      <td>603335.SH</td>
      <td>迪生力</td>
      <td>7.68</td>
      <td>+7.89%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>21</td>
      <td>600226.SH</td>
      <td>亨通股份</td>
      <td>7.43</td>
      <td>+7.89%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>22</td>
      <td>002354.SZ</td>
      <td>天娱数科</td>
      <td>8.06</td>
      <td>+7.89%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>23</td>
      <td>002584.SZ</td>
      <td>西陇科学</td>
      <td>7.88</td>
      <td>+7.89%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>24</td>
      <td>002484.SZ</td>
      <td>江海股份</td>
      <td>91.98</td>
      <td>+7.83%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>25</td>
      <td>600596.SH</td>
      <td>新安股份</td>
      <td>14.94</td>
      <td>+7.71%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>26</td>
      <td>001369.SZ</td>
      <td>双欣材料</td>
      <td>14.88</td>
      <td>+7.71%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>27</td>
      <td>605318.SH</td>
      <td>法狮龙</td>
      <td>105.96</td>
      <td>+7.65%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>28</td>
      <td>603163.SH</td>
      <td>圣晖集成</td>
      <td>106.54</td>
      <td>+7.65%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>29</td>
      <td>600903.SH</td>
      <td>贵州燃气</td>
      <td>9.14</td>
      <td>+7.51%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>30</td>
      <td>603065.SH</td>
      <td>宿迁联盛</td>
      <td>10.40</td>
      <td>+7.51%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
  </tbody>
</table></div>

- 命中（旧口径：actual_ret>0）：16/30（53.33%）
- 覆盖率（V2：in_p10，r_actual ∈ [p05,p95]）：43.33%
- 覆盖率（V2：in_p50，r_actual ∈ [p25,p75]）：13.33%
- MAE（V2：|err_r_p50|）：0.098721
- MAE（V2：|err_close_p50|）：3.703509

## E_ret_plus / EHX 验证摘要

- Raw MAE：5.9556%
- Plus MAE：10.1753%
- MAE 改善：-4.2196%
- Plus 优于 Raw 比例：23.33%

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

