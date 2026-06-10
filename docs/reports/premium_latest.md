# Premium（溢价预测）V3（E_ret_plus / Close[T+2] 分布预测）

> 注：T 为本次预测的**基准交易日**（使用 Close[T]）；T+2 为**预测到期交易日**（预测 Close[T+2] 的分布）。

- 预测日（T）：**20260608**
- 预测到期日（T+2）：**20260610**
- 周期：**2 个交易日（T→T+2）**
- 生成时间：2026-06-10T15:36:48Z
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
      <td>000517.SZ</td>
      <td>荣安地产</td>
      <td>2.33</td>
      <td>+12.00%</td>
      <td>2.52 ~ 2.70，中位 2.61</td>
      <td>26.52%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>2</td>
      <td>601588.SH</td>
      <td>北辰实业</td>
      <td>2.22</td>
      <td>+12.00%</td>
      <td>2.40 ~ 2.57，中位 2.49</td>
      <td>13.34%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>3</td>
      <td>600539.SH</td>
      <td>狮头股份</td>
      <td>15.40</td>
      <td>+12.00%</td>
      <td>16.68 ~ 17.84，中位 17.25</td>
      <td>10.94%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>4</td>
      <td>600721.SH</td>
      <td>百花医药</td>
      <td>7.79</td>
      <td>+12.00%</td>
      <td>8.44 ~ 9.02，中位 8.72</td>
      <td>18.52%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>5</td>
      <td>600403.SH</td>
      <td>大有能源</td>
      <td>8.95</td>
      <td>+12.00%</td>
      <td>9.69 ~ 10.37，中位 10.02</td>
      <td>16.98%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>6</td>
      <td>002164.SZ</td>
      <td>宁波东力</td>
      <td>14.66</td>
      <td>+12.00%</td>
      <td>15.87 ~ 16.98，中位 16.42</td>
      <td>9.99%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>7</td>
      <td>603877.SH</td>
      <td>太平鸟</td>
      <td>14.06</td>
      <td>+12.00%</td>
      <td>15.22 ~ 16.29，中位 15.75</td>
      <td>15.45%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>8</td>
      <td>603135.SH</td>
      <td>中重科技</td>
      <td>15.43</td>
      <td>+12.00%</td>
      <td>16.71 ~ 17.87，中位 17.28</td>
      <td>8.63%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>9</td>
      <td>000068.SZ</td>
      <td>华控赛格</td>
      <td>3.92</td>
      <td>+12.00%</td>
      <td>4.24 ~ 4.54，中位 4.39</td>
      <td>9.87%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>10</td>
      <td>002354.SZ</td>
      <td>天娱数科</td>
      <td>7.33</td>
      <td>+12.00%</td>
      <td>7.94 ~ 8.49，中位 8.21</td>
      <td>10.64%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>11</td>
      <td>603669.SH</td>
      <td>灵康药业</td>
      <td>4.48</td>
      <td>+12.00%</td>
      <td>4.85 ~ 5.19，中位 5.02</td>
      <td>3.59%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>12</td>
      <td>002490.SZ</td>
      <td>山东墨龙</td>
      <td>8.23</td>
      <td>+12.00%</td>
      <td>8.91 ~ 9.53，中位 9.22</td>
      <td>12.91%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>13</td>
      <td>603278.SH</td>
      <td>大业股份</td>
      <td>14.14</td>
      <td>+12.00%</td>
      <td>15.31 ~ 16.38，中位 15.84</td>
      <td>15.41%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>14</td>
      <td>000785.SZ</td>
      <td>居然智家</td>
      <td>2.72</td>
      <td>+12.00%</td>
      <td>2.95 ~ 3.15，中位 3.05</td>
      <td>6.58%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>15</td>
      <td>603500.SH</td>
      <td>祥和实业</td>
      <td>13.45</td>
      <td>+12.00%</td>
      <td>14.56 ~ 15.58，中位 15.06</td>
      <td>9.94%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>16</td>
      <td>002982.SZ</td>
      <td>湘佳股份</td>
      <td>13.48</td>
      <td>+12.00%</td>
      <td>14.60 ~ 15.62，中位 15.10</td>
      <td>7.39%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>17</td>
      <td>600505.SH</td>
      <td>西昌电力</td>
      <td>14.92</td>
      <td>+12.00%</td>
      <td>16.16 ~ 17.28，中位 16.71</td>
      <td>5.79%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>18</td>
      <td>000970.SZ</td>
      <td>中科三环</td>
      <td>12.63</td>
      <td>+12.00%</td>
      <td>13.68 ~ 14.63，中位 14.15</td>
      <td>14.86%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>19</td>
      <td>300105.SZ</td>
      <td>龙源技术</td>
      <td>7.18</td>
      <td>+12.00%</td>
      <td>7.77 ~ 8.32，中位 8.04</td>
      <td>11.58%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>20</td>
      <td>300376.SZ</td>
      <td>易事特</td>
      <td>5.94</td>
      <td>+11.79%</td>
      <td>6.42 ~ 6.87，中位 6.64</td>
      <td>10.79%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>21</td>
      <td>002613.SZ</td>
      <td>北玻股份</td>
      <td>5.46</td>
      <td>+11.79%</td>
      <td>5.90 ~ 6.31，中位 6.10</td>
      <td>5.48%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>22</td>
      <td>000759.SZ</td>
      <td>中百集团</td>
      <td>6.27</td>
      <td>+11.79%</td>
      <td>6.78 ~ 7.25，中位 7.01</td>
      <td>10.57%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>23</td>
      <td>600488.SH</td>
      <td>津药药业</td>
      <td>5.69</td>
      <td>+11.79%</td>
      <td>6.15 ~ 6.58，中位 6.36</td>
      <td>11.52%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>24</td>
      <td>002421.SZ</td>
      <td>达实智能</td>
      <td>5.73</td>
      <td>+11.79%</td>
      <td>6.19 ~ 6.63，中位 6.41</td>
      <td>36.58%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>25</td>
      <td>603616.SH</td>
      <td>韩建河山</td>
      <td>6.63</td>
      <td>+11.79%</td>
      <td>7.17 ~ 7.67，中位 7.41</td>
      <td>80.25%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>26</td>
      <td>600825.SH</td>
      <td>新华传媒</td>
      <td>5.45</td>
      <td>+11.79%</td>
      <td>5.89 ~ 6.30，中位 6.09</td>
      <td>8.60%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>27</td>
      <td>001696.SZ</td>
      <td>宗申动力</td>
      <td>15.95</td>
      <td>+11.58%</td>
      <td>17.21 ~ 18.41，中位 17.80</td>
      <td>7.58%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>28</td>
      <td>600280.SH</td>
      <td>中央商场</td>
      <td>3.74</td>
      <td>+11.25%</td>
      <td>4.02 ~ 4.30，中位 4.16</td>
      <td>18.89%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>29</td>
      <td>603186.SH</td>
      <td>华正新材</td>
      <td>136.03</td>
      <td>+10.21%</td>
      <td>144.95 ~ 155.06，中位 149.92</td>
      <td>9.33%</td>
      <td>高（1.000）</td>
    </tr>
    <tr>
      <td>30</td>
      <td>300197.SZ</td>
      <td>节能铁汉</td>
      <td>3.02</td>
      <td>+10.09%</td>
      <td>3.21 ~ 3.44，中位 3.32</td>
      <td>14.26%</td>
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
      <td>000517.SZ</td>
      <td>荣安地产</td>
      <td>2.33</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>2</td>
      <td>601588.SH</td>
      <td>北辰实业</td>
      <td>2.22</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>3</td>
      <td>600539.SH</td>
      <td>狮头股份</td>
      <td>15.40</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>4</td>
      <td>600721.SH</td>
      <td>百花医药</td>
      <td>7.79</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>5</td>
      <td>600403.SH</td>
      <td>大有能源</td>
      <td>8.95</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>6</td>
      <td>002164.SZ</td>
      <td>宁波东力</td>
      <td>14.66</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>7</td>
      <td>603877.SH</td>
      <td>太平鸟</td>
      <td>14.06</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>8</td>
      <td>603135.SH</td>
      <td>中重科技</td>
      <td>15.43</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>9</td>
      <td>000068.SZ</td>
      <td>华控赛格</td>
      <td>3.92</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>10</td>
      <td>002354.SZ</td>
      <td>天娱数科</td>
      <td>7.33</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>11</td>
      <td>603669.SH</td>
      <td>灵康药业</td>
      <td>4.48</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>12</td>
      <td>002490.SZ</td>
      <td>山东墨龙</td>
      <td>8.23</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>13</td>
      <td>603278.SH</td>
      <td>大业股份</td>
      <td>14.14</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>14</td>
      <td>000785.SZ</td>
      <td>居然智家</td>
      <td>2.72</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>15</td>
      <td>603500.SH</td>
      <td>祥和实业</td>
      <td>13.45</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>16</td>
      <td>002982.SZ</td>
      <td>湘佳股份</td>
      <td>13.48</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>17</td>
      <td>600505.SH</td>
      <td>西昌电力</td>
      <td>14.92</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>18</td>
      <td>000970.SZ</td>
      <td>中科三环</td>
      <td>12.63</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>19</td>
      <td>300105.SZ</td>
      <td>龙源技术</td>
      <td>7.18</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>20</td>
      <td>300376.SZ</td>
      <td>易事特</td>
      <td>5.94</td>
      <td>+11.79%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>21</td>
      <td>002613.SZ</td>
      <td>北玻股份</td>
      <td>5.46</td>
      <td>+11.79%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>22</td>
      <td>000759.SZ</td>
      <td>中百集团</td>
      <td>6.27</td>
      <td>+11.79%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>23</td>
      <td>600488.SH</td>
      <td>津药药业</td>
      <td>5.69</td>
      <td>+11.79%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>24</td>
      <td>002421.SZ</td>
      <td>达实智能</td>
      <td>5.73</td>
      <td>+11.79%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>25</td>
      <td>603616.SH</td>
      <td>韩建河山</td>
      <td>6.63</td>
      <td>+11.79%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>26</td>
      <td>600825.SH</td>
      <td>新华传媒</td>
      <td>5.45</td>
      <td>+11.79%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>27</td>
      <td>001696.SZ</td>
      <td>宗申动力</td>
      <td>15.95</td>
      <td>+11.58%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>28</td>
      <td>600280.SH</td>
      <td>中央商场</td>
      <td>3.74</td>
      <td>+11.25%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>29</td>
      <td>603186.SH</td>
      <td>华正新材</td>
      <td>136.03</td>
      <td>+10.21%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
    <tr>
      <td>30</td>
      <td>300197.SZ</td>
      <td>节能铁汉</td>
      <td>3.02</td>
      <td>+10.09%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
    </tr>
  </tbody>
</table></div>

- 命中（旧口径：actual_ret>0）：8/30（26.67%）
- 覆盖率（V2：in_p10，r_actual ∈ [p05,p95]）：16.67%
- 覆盖率（V2：in_p50，r_actual ∈ [p25,p75]）：6.67%
- MAE（V2：|err_r_p50|）：0.163573
- MAE（V2：|err_close_p50|）：1.751794

## E_ret_plus / EHX 验证摘要

- Raw MAE：9.3751%
- Plus MAE：16.6880%
- MAE 改善：-7.3130%
- Plus 优于 Raw 比例：16.67%

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
- 02. Pack1 enabled: market cache found -> /home/runner/work/top10-decision/top10-decision/data/market/daily_20260608.csv
- 03. Pack2 enabled (soft mode, fixed)

