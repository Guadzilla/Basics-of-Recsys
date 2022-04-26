# 任务6：Slope One

- 阅读[Slope One基础原理](https://blog.csdn.net/xidianliutingting/article/details/51916578)
- 编写Slope One用于电影推荐的流程
- 比较Slope One、SVD、协同过滤的精度，哪一个模型的RMSE评分更低？

代码地址： https://github.com/Guadzilla/recpre

## slope one 算法

1.示例引入

我们可以这么认为，商品间受欢迎的差异从某种程度上是固定的，比如所有人都喜欢海底捞火锅，但对赛百味的喜爱程度一般。此时小明对海底捞火锅的评分为4，对赛百味的评分为2；而小吴对海底捞火锅的评分为5，对赛百味的评分为3。尽管两个人评分的习惯上不同，小明平均打的分都高，但是对两个物品来说，他们之间的评分差值是不变的，即 $5-3=4-2$ 。

![image-20220426192311847](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220426192311847.png)

Slope one 的思路大抵如此，现在假设我们想预测 Alice 对物品 2 的评分，已有的是别的用户对物品 1 和对物品 2 的评分和 Alice 对物品 1 的评分。

从物品评分偏差的角度，我们可以求出物品 1 和物品 2 之间的评分偏差，即 $R_{1,2}=\frac{(3-1)+(4-3)+(3-3)+(1-5)}{4}=-0.25$，再用 Alice 对物品 1 的评分减去这个偏差，即 $p_{Alice,2}=r_{Alice,1}-R_{1,2}=5-(-0.25)=5.25$，把它作为 Alice 对物品 2 的预测评分。这就是Slope one 算法最简单的场景。

## slope one 算法思想

Slope One 算法是由 Daniel Lemire 教授在 2005 年提出的一个 **Item-Based** 的协同过滤推荐算法。和其它类似算法相比, 它的最大优点在于算法很简单, 易于实现, 执行效率高, 同时推荐的准确性相对较高。
Slope One算法是基于不同物品之间的评分差的线性算法，预测用户对物品评分的个性化算法。主要分为三步：

Step1: 计算物品之间的评分差的均值，记为物品间的评分偏差(两物品同时被评分)；

![这里写图片描述](https://img-blog.csdn.net/20160715114006473)

Step2:根据物品间的评分偏差和用户的历史评分，预测用户对未评分的物品的评分。

![这里写图片描述](https://img-blog.csdn.net/20160715114054480)

Step3:将预测评分排序，取topN对应的物品推荐给用户。

**举例：**
假设有100个人对物品A和物品B打分了，R(AB)表示这100个人对A和B打分的平均偏差;有1000个人对物品B和物品C打分了， R(CB)表示这1000个人对C和B打分的平均偏差；

![这里写图片描述](https://img-blog.csdn.net/20160715114619049)

## slope one 的代码实现

1.准备数据

![在这里插入图片描述](https://camo.githubusercontent.com/68d8995d1a9bacf4e58fa39359de71cbb99e3bf5abc5174bd1033e8b93fdae81/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303832373135303233373932312e706e67237069635f63656e746572)

```python
# 定义数据集
def loadData():
    rating_data={1: {'A': 5, 'B': 3, 'C': 4, 'D': 4},
                 2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
                 3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
                 4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
                 5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
              }
    return rating_data	
users_rating = loadData()
users_rating
"""
{1: {'A': 5, 'B': 3, 'C': 4, 'D': 4},
 2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
 3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
 4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
 5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}}
"""
```

2.建立倒排索引

```python
# 建立倒排索引
items_rating = {}
for user,ratings in users_rating.items():
    for item,rating in ratings.items():
        if item not in items_rating:
            items_rating[item] = {}
        if user not in items_rating[item]:
            items_rating[item][user] = 0
        items_rating[item][user] = rating
items_rating
"""
{'A': {1: 5, 2: 3, 3: 4, 4: 3, 5: 1},
 'B': {1: 3, 2: 1, 3: 3, 4: 3, 5: 5},
 'C': {1: 4, 2: 2, 3: 4, 4: 1, 5: 5},
 'D': {1: 4, 2: 3, 3: 3, 4: 5, 5: 2},
 'E': {2: 3, 3: 5, 4: 4, 5: 1}}
"""
```

3.计算物品间评分偏差矩阵

```python
# 计算物品间评分偏差
Ratings_diffs = {}	# 评分偏差矩阵
N_set = {}			# 物品对间共同被评分次数
for itemx,itemx_history in items_rating.items():
    if itemx not in Ratings_diffs:
        Ratings_diffs[itemx] = {}
        N_set[itemx] = {}
    for itemy,itemy_history in items_rating.items():
        if itemx != itemy:
            Ratings_diffs[itemx][itemy] = 0
            N_set[itemx][itemy] = 0
            for x_user in itemx_history:
                if x_user in itemy_history:
                    Ratings_diffs[itemx][itemy] += items_rating[itemy][x_user] - items_rating[itemx][x_user]
                    N_set[itemx][itemy] += 1
for itemx,ys in Ratings_diffs.items():
    for itemy,rating in ys.items():
        Ratings_diffs[itemx][itemy] /= N_set[itemx][itemy]

Ratings_diffs,N_set
"""
({'A': {'B': -0.2, 'C': 0.0, 'D': 0.2, 'E': 0.5},
  'B': {'A': 0.2, 'C': 0.2, 'D': 0.4, 'E': 0.25},
  'C': {'A': 0.0, 'B': -0.2, 'D': 0.2, 'E': 0.25},
  'D': {'A': -0.2, 'B': -0.4, 'C': -0.2, 'E': 0.0},
  'E': {'A': -0.5, 'B': -0.25, 'C': -0.25, 'D': 0.0}},
 {'A': {'B': 5, 'C': 5, 'D': 5, 'E': 4},
  'B': {'A': 5, 'C': 5, 'D': 5, 'E': 4},
  'C': {'A': 5, 'B': 5, 'D': 5, 'E': 4},
  'D': {'A': 5, 'B': 5, 'C': 5, 'E': 4},
  'E': {'A': 4, 'B': 4, 'C': 4, 'D': 4}})
"""
```

4.预测评分

```python
# 预测评分
# 首先找出Alice交互过的物品哪些与要预测的物品有过”共同被统一用户评分“的经历，即存在倒排索引Ratings_item[x][y]
A_history = users_rating[1]
candidate_items = set()
for item in Ratings_diffs['E']:
    if item in A_history:
        candidate_items.add(item)
weighted_score = 0
weighted_sum = 0
for item in Ratings_diffs['E']:
    weighted_sum += N_set['E'][item]
    weighted_score += Ratings_diffs['E'][item] * N_set['E'][item]
predict = weighted_score/weighted_sum
predict
"""
-0.25
"""
```

## slope one使用场景

该算法适用于物品更新不频繁，数量相对较稳定并且物品数目明显小于用户数的场景。依赖用户的用户行为日志和物品偏好的相关内容。
优点：
1.算法简单，易于实现，执行效率高；
2.可以发现用户潜在的兴趣爱好；
缺点：
依赖用户行为，存在冷启动问题和稀疏性问题。

## 比较Slope One、SVD、协同过滤的精度，哪一个模型的RMSE评分更低？



| Model     | RMSE    |
| --------- | ------- |
| UserCF    | 3.80369 |
| ItemCF    | 3.74319 |
| SVD(MF)   | 3.75332 |
| Slope One | 3.91533 |



# 参考资料：

https://blog.csdn.net/xidianliutingting/article/details/51916578
