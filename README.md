# Basics of Recsys
[Coggle 30 Days of ML（22年4月）](https://coggle.club/blog/30days-of-ml-202204)

# 任务介绍

**在本次学习中我们将学习推荐系统的基础操作，包括协同过滤、矩阵分解和向量召回等基础内容。**

- 任务1：推荐系统基础
  - 阅读推荐系统在工业落地的链接：
    - [推荐系统整体架构及算法流程详解](https://mp.weixin.qq.com/s/WXcfdzz7vts9UYBVxWs3AA)
    - [美团旅游推荐系统的演进](https://tech.meituan.com/2017/03/24/travel-recsys.html)
    - [阿里智能推荐AIRec](https://www.alibabacloud.com/zh/product/airec)
  - 思考 & 回答以下问题，并将回答记录到博客
    - 推荐系统与常见的结构化问题的区别是什么？
    - 如何评价推荐系统「推荐」的准不准？
    - 推荐系统一般分为召回 & 排序，为什么这样划分？
- 任务2：Movienles介绍
  - 下载并读取Movielens 1M数据集（用户、电影、评分）
  - 统计如下指标：
    - 总共包含多少用户？
    - 总共包含多个电影？
    - 平均每个用户对多少个用户进行了评分？
    - 每部电影 & 每个用户的平均评分是？
  - 如果你来进行划分数据集为训练和验证，你会如何划分？
- 任务3：协同过滤基础
  - [阅读协同过滤教程](https://github.com/datawhalechina/fun-rec/blob/master/docs/第一章 推荐系统基础/1.1 基础推荐算法/1.1.2 协同过滤.md)
  - 编写代码计算两个用户的相似度
  - 编写代码计算两个物品的相似度
- 任务4：协同过滤进阶
  - 编写User-CF代码，通过用户相似度得到电影推荐
  - 编写Item-CF代码，通过物品相似度得到电影推荐
  - 进阶：如果不使用矩阵乘法，你能使用倒排索引实现上述计算吗？
- 任务5：矩阵分解SVD
  - 阅读[矩阵分解基础教程](https://github.com/datawhalechina/fun-rec/blob/master/docs/第一章 推荐系统基础/1.1 基础推荐算法/1.1.2 协同过滤.md)，[代码实现](https://alyssaq.github.io/2015/20150426-simple-movie-recommender-using-svd/)
  - 编写SVD用于电影推荐的流程
  - 比较SVD与协同过滤的精度，哪一个模型的RMSE评分更低？
- 任务6：Slope One
  - 阅读[Slope One基础原理](https://blog.csdn.net/xidianliutingting/article/details/51916578)
  - 编写Slope One用于电影推荐的流程
  - 比较Slope One、SVD、协同过滤的精度，哪一个模型的RMSE评分更低？
- 任务7：词向量基础
  - 学习[word2vec基础](https://cloud.tencent.com/developer/article/1486055)
  - 将用户历史观看电影转为列表数据（一个用户一个列表）
  - 使用gensim训练word2vec，然后对用户完成聚类
- 任务8：向量召回基础
  - 基于任务7的基础上，使用编码后的用户向量，计算用户相似度。
  - 参考User-CF的过程，通过用户相似度得到电影推荐
- 任务9：多路召回实践
  - 基于任务3、任务5、任务6、任务7、任务8，总共5个召回模型，进行多路召回。
  - 可以考虑对每个召回模型的物品打分进行相加，也可以加权求和。
  - 分别计算每个模型 & 多路召回模型的Top10、Top20、Top50的命中率。

## 学习资料

https://github.com/datawhalechina/fun-rec

# 总结

关注到这个打卡活动时已经4月22日，比较晚了，为了在四月完成打卡，整体上做得比较匆忙。中间有很多问题还没有解决，在此罗列出来：

1. 任务4，协同过滤进阶中，ItemCF 和 UserCF 用 5 分制评分计算皮尔逊相似度时会出现 nan ，直接把 nan 置为 0 了（0 表示不相关）；维度为 1 的向量（标量）无法计算皮尔逊相似度，直接置为 1 。可能不合理。另外 5 分制的算法跑完结果都很差，可能有bug。
2. 任务5，用 pytorch 实现了 MF，没有负采样时效果很差，加入负采样以后效果有提示，但还没有仔细调参，实验结果一般般。在`task5/MF_pytorch.py` 实现的是 5 分制下的推荐，最后在任务9 里用的是 0-1 评分制下的推荐，还没整合。
3. 任务6，SlopeOne 5 分制的推荐效果很差（可能有bug），0-1评分制没有意义。
4. 任务9，因为各模型在 5 分制评分下效果都不好，所以最后用了各模型的 0-1 评分版本。又因为用的是 0-1 评分制，所以 SlopeOne 算法没有参与最后的多路召回。
5. 代码注释和博文里很多词混用："用户/user"、"评分过/交互过/看过/购买过/点击过''、"物品/商品/电影/item"，"MF/SVD"，"向量/embedding"。

有空再解决这些问题，最近好忙啊~~
