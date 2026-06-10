# Premium（溢价预测）V3（E_ret_plus / Close[T+2] 分布预测）

> 注：T 为本次预测的**基准交易日**（使用 Close[T]）；T+2 为**预测到期交易日**（预测 Close[T+2] 的分布）。

- 预测日（T）：**20260608**
- 预测到期日（T+2）：**20260610**
- 周期：**2 个交易日（T→T+2）**
- 生成时间：2026-06-10T15:09:09Z
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
      <th>操作结论</th>
      <th>风险提示</th>
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
      <td>优先观察</td>
      <td>无明显风险</td>
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
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>3</td>
      <td>000785.SZ</td>
      <td>居然智家</td>
      <td>2.72</td>
      <td>+12.00%</td>
      <td>2.95 ~ 3.15，中位 3.05</td>
      <td>6.58%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率偏低；高收益低胜率</td>
    </tr>
    <tr>
      <td>4</td>
      <td>603669.SH</td>
      <td>灵康药业</td>
      <td>4.48</td>
      <td>+10.61%</td>
      <td>4.79 ~ 5.13，中位 4.96</td>
      <td>3.59%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率偏低；高收益低胜率</td>
    </tr>
    <tr>
      <td>5</td>
      <td>603877.SH</td>
      <td>太平鸟</td>
      <td>14.06</td>
      <td>+10.13%</td>
      <td>14.97 ~ 16.02，中位 15.48</td>
      <td>15.45%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>6</td>
      <td>603500.SH</td>
      <td>祥和实业</td>
      <td>13.45</td>
      <td>+10.13%</td>
      <td>14.32 ~ 15.32，中位 14.81</td>
      <td>9.94%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率偏低；高收益低胜率</td>
    </tr>
    <tr>
      <td>7</td>
      <td>002982.SZ</td>
      <td>湘佳股份</td>
      <td>13.48</td>
      <td>+10.13%</td>
      <td>14.35 ~ 15.36，中位 14.85</td>
      <td>7.39%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率偏低；高收益低胜率</td>
    </tr>
    <tr>
      <td>8</td>
      <td>000970.SZ</td>
      <td>中科三环</td>
      <td>12.63</td>
      <td>+9.83%</td>
      <td>13.41 ~ 14.35，中位 13.87</td>
      <td>14.86%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>9</td>
      <td>600403.SH</td>
      <td>大有能源</td>
      <td>8.95</td>
      <td>+8.86%</td>
      <td>9.42 ~ 10.08，中位 9.74</td>
      <td>16.98%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>10</td>
      <td>688549.SH</td>
      <td>中巨芯-U</td>
      <td>19.19</td>
      <td>+8.61%</td>
      <td>20.15 ~ 21.56，中位 20.84</td>
      <td>1.10%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率偏低；高收益低胜率</td>
    </tr>
    <tr>
      <td>11</td>
      <td>603778.SH</td>
      <td>国晟科技</td>
      <td>18.95</td>
      <td>+8.61%</td>
      <td>19.90 ~ 21.29，中位 20.58</td>
      <td>21.15%</td>
      <td>高（1.000）</td>
      <td>优先观察</td>
      <td>无明显风险</td>
    </tr>
    <tr>
      <td>12</td>
      <td>600280.SH</td>
      <td>中央商场</td>
      <td>3.74</td>
      <td>+8.53%</td>
      <td>3.92 ~ 4.20，中位 4.06</td>
      <td>18.89%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>13</td>
      <td>603186.SH</td>
      <td>华正新材</td>
      <td>136.03</td>
      <td>+8.33%</td>
      <td>142.48 ~ 152.42，中位 147.37</td>
      <td>9.33%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率偏低；高收益低胜率</td>
    </tr>
    <tr>
      <td>14</td>
      <td>000068.SZ</td>
      <td>华控赛格</td>
      <td>3.92</td>
      <td>+8.24%</td>
      <td>4.10 ~ 4.39，中位 4.24</td>
      <td>9.87%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率偏低；高收益低胜率</td>
    </tr>
    <tr>
      <td>15</td>
      <td>002490.SZ</td>
      <td>山东墨龙</td>
      <td>8.23</td>
      <td>+8.15%</td>
      <td>8.61 ~ 9.21，中位 8.90</td>
      <td>12.91%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>16</td>
      <td>002354.SZ</td>
      <td>天娱数科</td>
      <td>7.33</td>
      <td>+8.15%</td>
      <td>7.66 ~ 8.20，中位 7.93</td>
      <td>10.64%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>17</td>
      <td>300105.SZ</td>
      <td>龙源技术</td>
      <td>7.18</td>
      <td>+8.15%</td>
      <td>7.51 ~ 8.03，中位 7.77</td>
      <td>11.58%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>18</td>
      <td>600721.SH</td>
      <td>百花医药</td>
      <td>7.79</td>
      <td>+8.15%</td>
      <td>8.15 ~ 8.71，中位 8.42</td>
      <td>18.52%</td>
      <td>高（1.000）</td>
      <td>高收益低胜率，谨慎</td>
      <td>上涨概率一般；高收益低胜率</td>
    </tr>
    <tr>
      <td>19</td>
      <td>000833.SZ</td>
      <td>粤桂股份</td>
      <td>23.68</td>
      <td>+7.65%</td>
      <td>24.65 ~ 26.37，中位 25.49</td>
      <td>16.35%</td>
      <td>高（1.000）</td>
      <td>稳健观察</td>
      <td>上涨概率一般</td>
    </tr>
    <tr>
      <td>20</td>
      <td>301302.SZ</td>
      <td>华如科技</td>
      <td>30.12</td>
      <td>+7.65%</td>
      <td>31.35 ~ 33.54，中位 32.42</td>
      <td>12.39%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率一般</td>
    </tr>
    <tr>
      <td>21</td>
      <td>603929.SH</td>
      <td>亚翔集成</td>
      <td>187.10</td>
      <td>+7.50%</td>
      <td>194.46 ~ 208.02，中位 201.13</td>
      <td>14.96%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率一般</td>
    </tr>
    <tr>
      <td>22</td>
      <td>300197.SZ</td>
      <td>节能铁汉</td>
      <td>3.02</td>
      <td>+7.38%</td>
      <td>3.14 ~ 3.35，中位 3.24</td>
      <td>14.26%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率一般</td>
    </tr>
    <tr>
      <td>23</td>
      <td>000620.SZ</td>
      <td>盈新发展</td>
      <td>3.17</td>
      <td>+7.33%</td>
      <td>3.29 ~ 3.52，中位 3.40</td>
      <td>13.44%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率一般</td>
    </tr>
    <tr>
      <td>24</td>
      <td>920578.BJ</td>
      <td>巨能股份</td>
      <td>22.75</td>
      <td>+7.27%</td>
      <td>23.59 ~ 25.24，中位 24.40</td>
      <td>10.27%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率一般</td>
    </tr>
    <tr>
      <td>25</td>
      <td>001696.SZ</td>
      <td>宗申动力</td>
      <td>15.95</td>
      <td>+7.06%</td>
      <td>16.51 ~ 17.66，中位 17.08</td>
      <td>7.58%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率偏低</td>
    </tr>
    <tr>
      <td>26</td>
      <td>603135.SH</td>
      <td>中重科技</td>
      <td>15.43</td>
      <td>+6.91%</td>
      <td>15.95 ~ 17.06，中位 16.50</td>
      <td>8.63%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率偏低</td>
    </tr>
    <tr>
      <td>27</td>
      <td>600505.SH</td>
      <td>西昌电力</td>
      <td>14.92</td>
      <td>+6.91%</td>
      <td>15.42 ~ 16.50，中位 15.95</td>
      <td>5.79%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率偏低</td>
    </tr>
    <tr>
      <td>28</td>
      <td>600539.SH</td>
      <td>狮头股份</td>
      <td>15.40</td>
      <td>+6.91%</td>
      <td>15.92 ~ 17.03，中位 16.46</td>
      <td>10.94%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率一般</td>
    </tr>
    <tr>
      <td>29</td>
      <td>603278.SH</td>
      <td>大业股份</td>
      <td>14.14</td>
      <td>+6.76%</td>
      <td>14.60 ~ 15.61，中位 15.10</td>
      <td>15.41%</td>
      <td>高（1.000）</td>
      <td>稳健观察</td>
      <td>上涨概率一般</td>
    </tr>
    <tr>
      <td>30</td>
      <td>002164.SZ</td>
      <td>宁波东力</td>
      <td>14.66</td>
      <td>+6.76%</td>
      <td>15.13 ~ 16.19，中位 15.65</td>
      <td>9.99%</td>
      <td>高（1.000）</td>
      <td>谨慎观察</td>
      <td>上涨概率偏低</td>
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
      <th>操作结论</th>
      <th>风险提示</th>
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
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
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
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>3</td>
      <td>000785.SZ</td>
      <td>居然智家</td>
      <td>2.72</td>
      <td>+12.00%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>4</td>
      <td>603669.SH</td>
      <td>灵康药业</td>
      <td>4.48</td>
      <td>+10.61%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>方向命中</td>
      <td>未命中核心区间；Plus未优于Raw</td>
    </tr>
    <tr>
      <td>5</td>
      <td>603877.SH</td>
      <td>太平鸟</td>
      <td>14.06</td>
      <td>+10.13%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>6</td>
      <td>603500.SH</td>
      <td>祥和实业</td>
      <td>13.45</td>
      <td>+10.13%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>7</td>
      <td>002982.SZ</td>
      <td>湘佳股份</td>
      <td>13.48</td>
      <td>+10.13%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>8</td>
      <td>000970.SZ</td>
      <td>中科三环</td>
      <td>12.63</td>
      <td>+9.83%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>未命中核心区间</td>
    </tr>
    <tr>
      <td>9</td>
      <td>600403.SH</td>
      <td>大有能源</td>
      <td>8.95</td>
      <td>+8.86%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>10</td>
      <td>688549.SH</td>
      <td>中巨芯-U</td>
      <td>19.19</td>
      <td>+8.61%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>11</td>
      <td>603778.SH</td>
      <td>国晟科技</td>
      <td>18.95</td>
      <td>+8.61%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>无明显风险</td>
    </tr>
    <tr>
      <td>12</td>
      <td>600280.SH</td>
      <td>中央商场</td>
      <td>3.74</td>
      <td>+8.53%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>13</td>
      <td>603186.SH</td>
      <td>华正新材</td>
      <td>136.03</td>
      <td>+8.33%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>14</td>
      <td>000068.SZ</td>
      <td>华控赛格</td>
      <td>3.92</td>
      <td>+8.24%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>15</td>
      <td>002490.SZ</td>
      <td>山东墨龙</td>
      <td>8.23</td>
      <td>+8.15%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>16</td>
      <td>002354.SZ</td>
      <td>天娱数科</td>
      <td>7.33</td>
      <td>+8.15%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>17</td>
      <td>300105.SZ</td>
      <td>龙源技术</td>
      <td>7.18</td>
      <td>+8.15%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>18</td>
      <td>600721.SH</td>
      <td>百花医药</td>
      <td>7.79</td>
      <td>+8.15%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>方向命中</td>
      <td>未命中核心区间；Plus未优于Raw</td>
    </tr>
    <tr>
      <td>19</td>
      <td>000833.SZ</td>
      <td>粤桂股份</td>
      <td>23.68</td>
      <td>+7.65%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>无明显风险</td>
    </tr>
    <tr>
      <td>20</td>
      <td>301302.SZ</td>
      <td>华如科技</td>
      <td>30.12</td>
      <td>+7.65%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>无明显风险</td>
    </tr>
    <tr>
      <td>21</td>
      <td>603929.SH</td>
      <td>亚翔集成</td>
      <td>187.10</td>
      <td>+7.50%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>22</td>
      <td>300197.SZ</td>
      <td>节能铁汉</td>
      <td>3.02</td>
      <td>+7.38%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>23</td>
      <td>000620.SZ</td>
      <td>盈新发展</td>
      <td>3.17</td>
      <td>+7.33%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>24</td>
      <td>920578.BJ</td>
      <td>巨能股份</td>
      <td>22.75</td>
      <td>+7.27%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>方向命中</td>
      <td>未命中核心区间；Plus未优于Raw</td>
    </tr>
    <tr>
      <td>25</td>
      <td>001696.SZ</td>
      <td>宗申动力</td>
      <td>15.95</td>
      <td>+7.06%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>验证较好</td>
      <td>Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>26</td>
      <td>603135.SH</td>
      <td>中重科技</td>
      <td>15.43</td>
      <td>+6.91%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>27</td>
      <td>600505.SH</td>
      <td>西昌电力</td>
      <td>14.92</td>
      <td>+6.91%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>28</td>
      <td>600539.SH</td>
      <td>狮头股份</td>
      <td>15.40</td>
      <td>+6.91%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>29</td>
      <td>603278.SH</td>
      <td>大业股份</td>
      <td>14.14</td>
      <td>+6.76%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
    <tr>
      <td>30</td>
      <td>002164.SZ</td>
      <td>宁波东力</td>
      <td>14.66</td>
      <td>+6.76%</td>
      <td>-</td>
      <td>-</td>
      <td>高</td>
      <td>模型高估</td>
      <td>实际收益为负；Plus误差偏大；分布未命中P10</td>
    </tr>
  </tbody>
</table></div>

- 命中（旧口径：actual_ret>0）：12/30（40.00%）
- 覆盖率（V2：in_p10，r_actual ∈ [p05,p95]）：23.33%
- 覆盖率（V2：in_p50，r_actual ∈ [p25,p75]）：10.00%
- MAE（V2：|err_r_p50|）：0.130428
- MAE（V2：|err_close_p50|）：2.367635

## E_ret_plus / EHX 验证摘要

- Raw MAE：10.1395%
- Plus MAE：13.3488%
- MAE 改善：-3.2093%
- Plus 优于 Raw 比例：30.00%

## 字段说明（V3 人类操作口径）

- 操作排名：优先使用 E_ret_plus 排名。
- T+2预期收益：EHX 残差增强后的 E_ret_plus。
- 预期价格区间：使用 T+2 价格分位 p25 ~ p75，并展示 p50 中位价。
- T+2上涨概率：预测到期日 T+2 收盘上涨概率。
- 模型置信度：EHX 置信度标签及置信分。
- 操作结论：面向人类执行的简化判断，不等于强制买入。
- 风险提示：根据上涨概率、E_ret_plus、验证误差与分布命中情况生成。

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

