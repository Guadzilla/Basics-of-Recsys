# Basics of Recsys
[Coggle 30 Days of ML（22年4月）](https://coggle.club/blog/30days-of-ml-202204)

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

### 学习资料

https://github.com/datawhalechina/fun-rec
