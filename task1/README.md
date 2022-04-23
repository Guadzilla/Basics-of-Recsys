# 任务1：推荐系统基础

- 阅读推荐系统在工业落地的链接：
  - [推荐系统整体架构及算法流程详解](https://mp.weixin.qq.com/s/WXcfdzz7vts9UYBVxWs3AA)
  - [美团旅游推荐系统的演进](https://tech.meituan.com/2017/03/24/travel-recsys.html)
  - [阿里智能推荐AIRec](https://www.alibabacloud.com/zh/product/airec)
- 思考 & 回答以下问题，并将回答记录到博客
  - 推荐系统与常见的结构化问题的区别是什么？
  - 如何评价推荐系统「推荐」的准不准？
  - 推荐系统一般分为召回 & 排序，为什么这样划分？

## 推荐系统与常见的结构化问题的区别是什么？

结构化数据即表格数据（tabular data），绝大多数数据都是表格数据。

推荐系统有user信息，item信息，这些数据虽然都是结构化的，但是推荐系统涉及到user和item的交互，所以不仅仅是一条结构化数据预测一个分类or一个数值那么简单，还需要从交互中抽象出用户兴趣，例如将用户交互建模为序列，这就不是结构化问题了。

## 如何评价推荐系统「推荐」的准不准？

常用的评价指标有：召回率Recall，准确率Precision，覆盖率Coverage，新颖度Popularity。

召回率Recall：正确推荐的商品占所有应该推荐的商品的比例，即应该推荐的推荐了多少。公式描述：对用户u推荐N个物品（$R(u)$），令用户在测试集上喜欢的物品集合为$T(u)$，则
$$
Recall=\frac{\sum_u|R(u) \cap T(u)|}{\sum_u |T(u)|}
$$
准确率Precision：正确推荐的商品占推荐的商品列表的比例，即有多少推荐对了。公式描述：
$$
Precision=\frac{\sum_u|R(u) \cap T(u)|}{\sum_u |R(u)|}
$$
覆盖率Coverage：推荐的商品占所有商品的比例，即推荐的商品覆盖了多少所有商品。反映发掘长尾的能力。
$$
Coverage = \frac{\bigcup_u R(u)}{|I|} \ \  , \ \bigcup:并集
$$
新颖度Popularity：刻画推荐物品的平均流行度，平均流行度（Popularity）越高，新颖度越低。$Popularity(x)$定义为$x$在所有用户序列中出现的次数，出现次数越多，流行度越高。
$$
Popularity= \sum _u \sum _ { i \in R(u) } \log (Popularity(i)+1)
$$

AUC曲线：AUC（Area Under Curve），ROC曲线下与坐标轴围成的面积。在讲AUC前需要理解混淆矩阵，召回率，精确率，ROC曲线等概念。

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/v2-a253b01cf7f141b9ad11eefdf3cf58d3_1440w.jpg)

根据混淆矩阵的定义，以另一种形式定义召回率和精确率：
$$
Recall = \frac{TP}{TP+FN} \\
Precision = \frac{TP}{TP+FP}
$$
ROC曲线的横坐标为假阳性率（False Positive Rate, FPR），$FPR=\frac{FP}{FP+TN}$，N是真实负样本的个数， FP是N个负样本中被分类器预测为正样本的个数。**FPRate的意义是所有真实类别为0的样本中，预测类别为1的比例。**

纵坐标为真阳性率（True Positive Rate, TPR），$TPR=\frac{TP}{TP+FN}$，P是真实正样本的个数，TP是P个正样本中被分类器预测为正样本的个数。**TPRate的意义是所有真实类别为1的样本中，预测类别为1的比例。**

![img](https://camo.githubusercontent.com/07a924ef2229334f903f1ba3e5cd17115a16159dcb1756cda93232b3cf998c0d/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2f4a6176616175632e706e67)

AUC的计算方法同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价。对ROC更细致的解释：[如何理解机器学习和统计中的AUC？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/39840928/answer/241440370)

下面是部分评价指标的代码实现。

```python
# 评价指标:召回率、准确率
def Metric(train, test, N, all_recommend_list):  # N:推荐N个物品
    hit = 0
    recall_all = 0      # recall 的分母
    precision_all = 0   # precision 的分母
    for user in train.keys():
        tu = test[user]
        rank = all_recommend_list[user][0:N]
        for item, pui in rank:
            if item in tu:
                hit += 1
        recall_all += len(tu)
        precision_all += N
    recall = hit / (recall_all * 1.0)
    precision = hit / (precision_all * 1.0)
    return recall, precision

# 评价指标：覆盖率
def Coverage(train, test, N, all_recommend_list):  # N:推荐N个物品
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        for item, pui in rank:
            recommend_items.add(item)
    coverage = len(recommend_items) / (len(all_items) * 1.0)
    return coverage


# 评价指标：新颖度
def Popularity(train, test, N, recommend_res):	# N:推荐N个物品
    item_popularity = dict()
    for user, items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    popularity = 0
    n = 0
    for user in train.keys():
        rank = recommend_res[user][0:N]
        for item, pui in rank:
            popularity += math.log(1 + item_popularity[item])
            n += 1
    popularity /= n * 1.0
    return popularity
```





## 推荐系统一般分为召回 & 精排，为什么这样划分？

> 物品集量级非常大，召回先选出一部分候选物品，再对这一部分候选物品做精排，计算开销相比于对所有物品做精排大大降低了。

相关知识点：召回、排序（精排）。

### 召回

从海量的物品中，先筛选出一小部分物品作为推荐的候选集。

#### 召回的目的

当用户和物品量比较大时，如果直接精排（计算预测得分）复杂度会非常高。计算预测得分通常是计算用户和物品的向量内积，假设 user 和 item 的 embedding 维度都是 D ，用户数为 M ，物品数为 N ，那么计算这个得分的复杂度就是 $O(D^2) *O(MN)$。当 M 和 N 都是百万量级、亿量级时，计算开销会非常大。

如果可以先从海量的物品中，先筛选出一小部分用户最可能喜欢的物品（召回），例如先选出 N/100 的物品，那么复杂度就是 $\frac{O(D^2) *O(MN)}{100}$ ，降低为原来的一百分之一，计算效率更高了。实际场景中，做热销召回的量级可能是百级，这样一来从百万量级的物品数降低到百量级的物品数，计算开销大大降低！另一方面，大量内容中真正的精品只是少数，对所有内容都计算将非常的低效，会浪费大量资源和时间。

#### 召回的重要性

虽然精排模型一直是优化的重点，但召回模型也非常的重要，因为如果召回的内容不对，怎么精排都是错误的。

#### 召回的方法

1. 热销召回：将一段时间内的热门内容召回。
2. 协同召回：基于用户与用户行为的相似性推荐，可以很好的突破一定的限制，发现用户潜在的兴趣偏好。
3. 标签召回：根据每个用户的行为，构建标签，并根据标签召回内容。
4. 时间召回：将一段时间内最新的内容召回，在新闻视频等有时效性的领域常用。是常见的几种召回方法。

#### 多路召回

一开始我们可能有成千上万的物品，首先要由召回（也叫触发，recall）来挖掘出原则上任何用户有可能感兴趣的东西。这个环节是入口。有时候，单独的召回可能难以做到照顾所有方面，这个时候就需要多路召回。所谓的“多路召回”策略，就是指采用不同的策略、特征或简单模型，分别召回一部分候选集，然后把候选集混合在一起供后续排序模型使用。下图只是一个多路召回的例子，也就是说可以使用多种不同的策略来获取用户排序的候选商品集合，**而具体使用哪些召回策略其实是与业务强相关的**，针对不同的任务就会有对于该业务真实场景下需要考虑的召回规则。例如视频推荐，召回规则可以是“热门视频”、“导演召回”、“演员召回”、“最近上映“、”流行趋势“、”类型召回“等等。

<img src="https://camo.githubusercontent.com/5194f61aac70bfec14ede3fa6b27aed0670f2cd59b5e9ad688f1f526a4c5f658/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303731373230313431313336372e706e67237069635f63656e746572" alt="img" style="zoom:67%;" />

#### Embedding召回



### 精排

排序负责将多个召回策略的结果进行个性化排序。

#### 精排的重要性

精排是最纯粹的排序，也是最纯粹的机器学习模块。它的目标只有一个，就是**根据手头所有的信息输出最准**的预测。精排一直是优化的重点。召回的物品中，筛选出用户最感兴趣的物品，进一步做出个性化排序，才最终达到推荐的目的。

#### 精排模型

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/icTMNdGHpfJYqcAFSwiaWKjeqTweM9aJrNKqZVvMn2GZvoDTnPHjYMVywvGicII8P9d4nMjib5Jia8kGlDbicibTGSPlQ/640?wx_fmt=jpeg&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:67%;" />

