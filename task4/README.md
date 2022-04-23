# 任务4：协同过滤进阶

- 编写User-CF代码，通过用户相似度得到电影推荐
- 编写Item-CF代码，通过物品相似度得到电影推荐
- 进阶：如果不使用矩阵乘法，你能使用倒排索引实现上述计算吗？

## 基于用户的协同过滤

### UserCF原理介绍

基于用户的协同过滤算法(UserCF)的假设是：**相似用户的兴趣也相似**。所以，当一个用户A需要个性化推荐的时候， 我们可以先找到和他有相似兴趣的其他用户， 然后把那些用户喜欢的， 而用户A没有听说过的物品推荐给A。

<img src="https://camo.githubusercontent.com/4cf6c62c1f1e533dffdd89db9bc045b560c95444a4a6e61b7b2d49cf28703f06/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2fe59bbee78987696d6167652d32303231303632393233323534303238392e706e67" alt="image-20210629232540289"  />

**UserCF算法主要包括两个步骤：**

1. 找到和目标用户兴趣相似的用户集合
2. 找到这个集合中的用户喜欢的， 且目标用户没有听说过的物品推荐给目标用户。



上面的两个步骤中， 第一个步骤里面， 我们会基于前面给出的相似性度量的方法找出与目标用户兴趣相似的用户， 而第二个步骤里面， 如何基于相似用户喜欢的物品来对目标用户进行推荐呢？ 这个要依赖于目标用户对相似用户喜欢的物品的一个喜好程度， 那么如何衡量这个程度大小呢？ 为了更好理解上面的两个步骤， 下面拿一个具体的例子把两个步骤具体化。



**以下图为例，此例将会用于本文各种算法中**

[![image-20210629232622758](https://camo.githubusercontent.com/38b1abf510c90bc3a8850a09ba19e39ea11b1f83b916f0432cc29fdaed665e5c/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2f254535253942254245254537253839253837696d6167652d32303231303632393233323632323735382e706e67)](https://camo.githubusercontent.com/38b1abf510c90bc3a8850a09ba19e39ea11b1f83b916f0432cc29fdaed665e5c/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2f254535253942254245254537253839253837696d6167652d32303231303632393233323632323735382e706e67)

给用户推荐物品的过程可以**形象化为一个猜测用户对商品进行打分的任务**，上面表格里面是5个用户对于5件物品的一个打分情况，可以理解为用户对物品的喜欢程度

应用UserCF算法的两个步骤：

1. 首先根据前面的这些打分情况(或者说已有的用户向量）计算一下Alice和用户1， 2， 3， 4的相似程度， 找出与Alice最相似的n个用户
2. 根据这n个用户对物品5的评分情况和与Alice的相似程度会猜测出Alice对物品5的评分， 如果评分比较高的话， 就把物品5推荐给用户Alice， 否则不推荐。

关于第一个步骤， 上面已经给出了计算两个用户相似性的方法， 这里不再过多赘述， 这里主要解决第二个问题， 如何产生最终结果的预测。

**最终结果的预测**

根据上面的几种方法， 我们可以计算出向量之间的相似程度， 也就是可以计算出Alice和其他用户的相近程度， 这时候我们就可以选出与Alice最相近的前n个用户， 基于他们对物品5的评价猜测出Alice的打分值， 那么是怎么计算的呢？

这里常用的方式之一是**利用用户相似度和相似用户的评价加权平均获得用户的评价预测**， 用下面式子表示：
$$
R_{\mathrm{u}, \mathrm{p}}=\frac{\sum_{\mathrm{s} \in S}\left(w_{\mathrm{u}, \mathrm{s}} \cdot R_{\mathrm{s}, \mathrm{p}}\right)}{\sum_{\mathrm{s} \in S} w_{\mathrm{u}, \mathrm{s}}}
$$
这个式子里面， 权重$w_{u,s}$是用户$u$和用户$s$的相似度， $R_{s,p}$是用户$s$对物品$p$的评分。

还有一种方式如下：
$$
P_{i, j}=\bar{R}_{i}+\frac{\sum_{k=1}^{n}\left(S_{i, k}\left(R_{k, j}-\bar{R}_{k}\right)\right)}{\sum_{k=1}^{n} S_{i, k}}
$$
这种方式考虑的更加全面， 依然是用户相似度作为权值， 但后面不单纯是其他用户对物品的评分， 而是**该物品的评分与此用户的所有评分的差值进行加权平均， 这时候考虑到了有的用户内心的评分标准不一的情况**， 即有的用户喜欢打高分， 有的用户喜欢打低分的情况。

所以这一种计算方式更为推荐。下面的计算将使用这个方式。这里的$S_{i,k}$与上面的$w_{u,s}$的意思是类似的，表示的是用户i和用户k之间的相似度。

在获得用户$u$对不同物品的评价预测后， 最终的推荐列表根据预测评分进行排序得到。 至此，基于用户的协同过滤算法的推荐过程完成。

根据上面的问题， 下面手算一下：

目标: 猜测Alice对物品5的得分：

1. **计算Alice与其他用户的相似度（这里使用皮尔逊相关系数）**:

```python
# 定义数据集， 也就是那个表格， 注意这里我们采用字典存放数据， 因为实际情况中数据是非常稀疏的， 很少有情况是现在这样
def loadData():
    ratings={'Alice': {'item1': 5, 'item2': 3, 'item3': 4, 'item4': 4},
           'user1': {'item1': 3, 'item2': 1, 'item3': 2, 'item4': 3, 'item5': 3},
           'user2': {'item1': 4, 'item2': 3, 'item3': 4, 'item4': 3, 'item5': 5},
           'user3': {'item1': 3, 'item2': 3, 'item3': 1, 'item4': 5, 'item5': 4},
           'user4': {'item1': 1, 'item2': 5, 'item3': 5, 'item4': 2, 'item5': 1}
          }
    return ratings
ratings = loadData()
ratings = pd.DataFrame(ratings).T
ratings
```

![image-20220422153425763](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220422153425763.png)

```python
# 取出用户向量
Alice = ratings.loc['Alice',:'item4']
user1 = ratings.loc['user1',:'item4']
user2 = ratings.loc['user2',:'item4']
user3 = ratings.loc['user3',:'item4']
user4 = ratings.loc['user4',:'item4']

# 定义皮尔逊相似度
from scipy.stats import pearsonr
def pearsonrSim(x,y):
    """
    皮尔森相似度
    """
    return pearsonr(x,y)[0]

# 计算Alice和其它用户的相似度
Alice_user1_similarity = pearsonrSim(Alice,user1)
Alice_user2_similarity = pearsonrSim(Alice,user2)
Alice_user3_similarity = pearsonrSim(Alice,user3)
Alice_user4_similarity = pearsonrSim(Alice,user4)
Alice_user1_similarity,Alice_user2_similarity,Alice_user3_similarity,Alice_user4_similarity

# 输出相似度
(0.8528028654224415, 0.7071067811865475, 0.0, -0.7921180343813393)
```

从这里看出, Alice用户1和用户2,用户3,用户4的相似度是0.85, 0.7, 0, -0.79。 所以如果n=2， 找到与Alice最相近的两个用户是用户1， 和Alice的相似度是0.85， 用户2， 和Alice相似度是0.7。

2. **根据相似度用户计算Alice对物品5的最终得分** 用户1对物品5的评分是3， 用户2对物品5的打分是5， 那么根据上面的计算公式， 可以计算出Alice对物品5的最终得分是 

$$
P_{Alice, 物品5}=\bar{R}_{Alice}+\frac{\sum_{k=1}^{2}\left(S_{Alice,user k}\left(R_{userk, 物品5}-\bar{R}_{userk}\right)\right)}{\sum_{k=1}^{2} S_{Alice, userk}}=4+\frac{0.85*(3-2.4)+0.7*(5-3.8)}{0.85+0.7}=4.87
$$

![image-20220422154709719](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220422154709719.png)



3. **根据用户评分对用户进行推荐** 这时候， 我们就得到了Alice对物品5的得分是4.87， 根据Alice的打分对物品排个序从大到小：$$物品1>物品5>物品3=物品4>物品2$$ 这时候，如果要向Alice推荐2款产品的话， 我们就可以推荐物品1和物品5给Alice

至此， 基于用户的协同过滤算法原理介绍完毕。

### UserCF代码实现

这里简单的通过编程实现上面的案例，为后面的大作业做一个热身， 梳理一下上面的过程其实就是三步： 计算用户相似性矩阵、得到前n个相似用户、计算最终得分。

所以我们下面的程序也是分为这三步：

1. **首先， 先把数据表给建立起来** 这里采用字典的方式， 之所以没有用pandas， 是因为上面举得这个例子其实是个个例， 在真实情况中， 我们知道， 用户对物品的打分情况并不会这么完整， 会存在大量的空值， 所以矩阵会很稀疏， 这时候用DataFrame， 会有大量的NaN。故这里用字典的形式存储。 用两个字典， 第一个字典是物品-用户的评分映射， 键是物品1-5， 用A-E来表示， 每一个值又是一个字典， 表示的是每个用户对该物品的打分。 第二个字典是用户-物品的评分映射， 键是上面的五个用户， 用1-5表示， 值是该用户对每个物品的打分。

```python
# 定义数据集， 也就是那个表格， 注意这里我们采用字典存放数据， 因为实际情况中数据是非常稀疏的， 很少有情况是现在这样
def loadData():
    items={'A': {1: 5, 2: 3, 3: 4, 4: 3, 5: 1},
           'B': {1: 3, 2: 1, 3: 3, 4: 3, 5: 5},
           'C': {1: 4, 2: 2, 3: 4, 4: 1, 5: 5},
           'D': {1: 4, 2: 3, 3: 3, 4: 5, 5: 2},
           'E': {2: 3, 3: 5, 4: 4, 5: 1}
          }
    users={1: {'A': 5, 'B': 3, 'C': 4, 'D': 4},
           2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
           3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
           4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
           5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
          }
    return items,users

items, users = loadData()
item_df = pd.DataFrame(items).T
user_df = pd.DataFrame(users).T
```

2. **计算用户相似性矩阵** 这个是一个共现矩阵, 5*5，行代表每个用户， 列代表每个用户， 值代表用户和用户的相关性，这里的思路是这样， 因为要求用户和用户两两的相关性， 所以需要用双层循环遍历用户-物品评分数据， 当不是同一个用户的时候， 我们要去遍历物品-用户评分数据， 在里面去找这两个用户同时对该物品评过分的数据放入到这两个用户向量中。 因为正常情况下会存在很多的NAN， 即可能用户并没有对某个物品进行评分过， 这样的不能当做用户向量的一部分， 没法计算相似性。 还是看代码吧， 感觉不太好描述：

```python
"""计算用户相似性矩阵"""
similarity_matrix = pd.DataFrame(-1 * np.ones((len(users), len(users))), index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])

for userx in users:
    for usery in users:
        userxVec=[]
        useryVec=[]
        if userx == usery:
            continue
        else:
            userx_history = users[userx].keys()
            usery_history = users[usery].keys()
            intersection = set(userx_history).intersection(usery_history) # 用户x和用户y行为历史的交集，否则有nan无法计算相似性
            for i in intersection:
                userxVec.append(users[userx][i])
                useryVec.append(users[usery][i])
            similarity_matrix[userx][usery]=np.corrcoef(np.array(userxVec),np.array(useryVec))[0][1]
```

得到如下user相似性矩阵：

![image-20220422161751278](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220422161751278.png)

注意相似度矩阵的初始值为-1，因为皮尔逊相关系数的取值为[-1,1]。

3. **计算前n个相似的用户**

```python
"""计算前n个相似的用户"""
n = 2
similar_users = dict()
for user in users:
    similar_users[user] = similarity_matrix[user].sort_values(ascending=False)[:n].index.tolist()
    
similar_users
{1: [2, 3], 2: [1, 4], 3: [1, 2], 4: [2, 1], 5: [3, 4]}
```

经计算，与用户1最相似的2个用户分别是 用户2 和 用户3 。

4. **计算最终得分**

这里就是上面的那个公式：
$$
P_{Alice, 物品5}=\bar{R}_{Alice}+\frac{\sum_{k=1}^{2}\left(S_{Alice,user k}\left(R_{userk, 物品5}-\bar{R}_{userk}\right)\right)}{\sum_{k=1}^{2} S_{Alice, userk}}
$$

```python
"""计算最后得分,用户1对物品E的预测评分"""

# 计算所有用户平均评分
user_mean_rating = dict()
for user in users:
    user_mean = np.mean([value for value in users[user].values()])
    user_mean_rating[user] = user_mean
# 计算预测得分
weighted_scores = 0.
corr_values_sum = 0.
for user in similar_users[1]:
    weighted_scores += similarity_matrix[1][user]
    corr_values_sum += similarity_matrix[1][user] * (users[user]['E'] - user_mean_rating[user])
predict = user_mean_rating[1] + corr_values_sum/weighted_scores

print(f'用户1对物品E的预测评分为 {predict:.2f} ')

用户1对物品E的预测评分为 4.87
```



计算结果如下：

<img src="https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220422164842510.png" alt="image-20220422164842510" style="zoom:67%;" />

### UserCF的缺点

UserCF算法存在两个重大问题：

1. 数据稀疏性。 一个大型的电子商务推荐系统一般有非常多的物品，用户可能买的其中不到1%的物品，不同用户之间买的物品重叠性较低，导致算法无法找到一个用户的邻居，即偏好相似的用户。**这导致UserCF不适用于那些正反馈获取较困难的应用场景**(如酒店预订， 大件商品购买等低频应用)
2. 算法扩展性。 基于用户的协同过滤需要维护用户相似度矩阵以便快速的找出Topn相似用户， 该矩阵的存储开销非常大，存储空间随着用户数量的增加而增加，**不适合用户数据量大的情况使用**。

由于UserCF技术上的两点缺陷， 导致很多电商平台并没有采用这种算法， 而是采用了ItemCF算法实现最初的推荐系统。

## 基于物品的协同过滤

### ItemCF原理介绍

基于物品的协同过滤(ItemCF)的基本思想是预先根据所有用户的历史偏好数据计算物品之间的相似性，然后把与用户喜欢的物品相类似的物品推荐给用户。比如物品a和c非常相似，因为喜欢a的用户同时也喜欢c，而用户A喜欢a，所以把c推荐给用户A。**ItemCF算法并不利用物品的内容属性计算物品之间的相似度， 主要通过分析用户的行为记录计算物品之间的相似度， 该算法认为， 物品a和物品c具有很大的相似度是因为喜欢物品a的用户大都喜欢物品c**。

**和UserCF类似，ItemCF算法主要包括两个步骤：**

- 计算物品之间的相似度
- 根据物品的相似度和用户的历史行为给用户生成推荐列表（购买了该商品的用户也经常购买的其他商品）



这里直接还是拿上面Alice的那个例子来看。

![image-20210629232622758](https://camo.githubusercontent.com/38b1abf510c90bc3a8850a09ba19e39ea11b1f83b916f0432cc29fdaed665e5c/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2f254535253942254245254537253839253837696d6167652d32303231303632393233323632323735382e706e67)

如果想知道Alice对物品5打多少分， 基于物品的协同过滤算法会这么做：

1. 首先计算一下物品5和物品1， 2， 3， 4之间的相似性(它们也是向量的形式， 每一列的值就是它们的向量表示， 因为ItemCF认为如果物品a和物品c具有很大的相似度，那么是因为喜欢物品a的用户大都喜欢物品c， 所以就可以基于每个用户对该物品的打分或者说喜欢程度来向量化物品)
2. 找出与物品5最相近的n个物品（取n=2）
3. 根据Alice对最相近的n个物品的打分去计算对物品5的打分情况，加入评分偏置的预测公式如下：

$$
P_{Alice, 物品5}=\bar{R}_{物品5}+\frac{\sum_{k=1}^{2}\left(S_{物品5,物品 k}\left(R_{Alice, 物品k}-\bar{R}_{物品k}\right)\right)}{\sum_{k=1}^{2} S_{物品k, 物品5}}
$$



**下面我们就可以具体计算一下，猜测Alice对物品5的打分：**

首先是步骤1：计算物品5和其它物品之间的相似度。

![image-20220422193745961](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220422193745961.png)

```python
# 获取item向量
item5 = ratings.loc['user1':,'item5'].values.tolist()
item4 = ratings.loc['user1':,'item4'].values.tolist()
item3 = ratings.loc['user1':,'item3'].values.tolist()
item2 = ratings.loc['user1':,'item2'].values.tolist()
item1 = ratings.loc['user1':,'item1'].values.tolist()

# 计算item相似度
item51_similarity = pearsonrSim(item5,item1)
item52_similarity = pearsonrSim(item5,item2)
item53_similarity = pearsonrSim(item5,item3)
item54_similarity = pearsonrSim(item5,item4)
[x.round(2) for x in (item51_similarity,item52_similarity,item53_similarity,item54_similarity)]

# 输出相似度
[0.97, -0.48, -0.43, 0.58]
```

步骤2：对相似度进行排序，选择最靠前的n=2个物品：item1和item4

步骤3：下面根据公式计算Alice对物品5的打分
$$
P_{Alice, 物品5}=\bar{R}_{物品5}+\frac{\sum_{k=1}^{2}\left(S_{物品5,物品 k}\left(R_{Alice, 物品k}-\bar{R}_{物品k}\right)\right)}{\sum_{k=1}^{2} S_{物品k, 物品5}}=\frac{13}{4}+\frac{0.97*(5-3.2)+0.58*(4-3.4)}{0.97+0.58}=4.6
$$
这时候依然可以向Alice推荐物品5。

下面也是简单编程实现一下， 和上面的差不多：

### ItemCF代码实现

```python
"""计算物品的相似矩阵"""
similarity_matrix = pd.DataFrame(-1 * np.ones((len(items), len(items))), index=['A', 'B', 'C', 'D', 'E'], columns=['A', 'B', 'C', 'D', 'E'])

# 遍历每条物品-用户评分数据
for itemx in items:
    for itemy in items:
        itemxVec = []
        itemyVec = []
        if itemx == itemy:
            continue
        else:
            itemx_history = set(items[itemx].keys())
            itemy_history = set(items[itemy].keys())
            intersection = itemx_history.intersection(itemy_history)  # 求交集，同时对两个物品都打分的用户，才有意义
            for i in intersection:
                itemxVec.append(items[itemx][i])
                itemyVec.append(items[itemy][i])
            similarity_matrix[itemx][itemy] = pearsonrSim(itemxVec,itemyVec).round(2)
            # similarity_matrix[itemx][itemy] = np.corrcoef(np.array(itemxVec),np.array(itemyVec))[0][1] 两种计算方式等价
```

得到物品相似度矩阵：

![image-20220422195802322](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220422195802322.png)

```python
"""计算前n个相似的用户"""
n = 2
similar_items = dict()
for item in items:
    similar_items[item] = similarity_matrix[item].sort_values(ascending=False)[:n].index.tolist()
    
similar_items['E']

# 与E最相似的2个物品是A和D
['A', 'D']
```



```python
"""计算最后得分,用户1对物品E的预测评分"""

# 计算物品平均打分情况
item_ratings_mean = dict()
for item,rating in items.items():
    item_ratings_mean[item] = np.mean([value for value in rating.values()])

weighted_scores = 0.
corr_values_sum = 0.

for item in similar_items['E']:
    weighted_scores += similarity_matrix['E'][item]
    corr_values_sum += similarity_matrix['E'][item] * (users[1][item] -  item_ratings_mean[item])

predict = item_ratings_mean['E'] + corr_values_sum/weighted_scores
print(f'用户1对物品E的预测得分为 {predict:.2f}')
```

### 9. 协同过滤算法的问题分析

协同过滤算法存在的问题之一就是**泛化能力弱**， 即协同过滤无法将两个物品相似的信息推广到其他物品的相似性上。 导致的问题是**热门物品具有很强的头部效应， 容易跟大量物品产生相似， 而尾部物品由于特征向量稀疏， 导致很少被推荐**。 比如下面这个例子：

![image-20220422203637025](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220422203637025.png)

A, B, C, D是物品， 看右边的物品共现矩阵， 可以发现物品D与A、B、C的相似度比较大， 所以很有可能将D推荐给用过A、B、C的用户。 但是物品D与其他物品相似的原因是因为D是一件热门商品， 系统无法找出A、B、C之间相似性的原因是其特征太稀疏， 缺乏相似性计算的直接数据。 所以这就是协同过滤的天然缺陷：**推荐系统头部效应明显， 处理稀疏向量的能力弱**。

为了解决这个问题， 同时增加模型的泛化能力，2006年，**矩阵分解技术(Matrix Factorization,MF**)被提出， 该方法在协同过滤共现矩阵的基础上， 使用更稠密的隐向量表示用户和物品， 挖掘用户和物品的隐含兴趣和隐含特征， 在一定程度上弥补协同过滤模型处理稀疏矩阵能力不足的问题。

## 进阶：如果不使用矩阵乘法，你能使用倒排索引实现上述计算吗？

使用余弦相似度的协同过滤算法可以使用，具体见python文件。

集合角度的余弦相似度计算如下：


$$
sim_{uv}=\frac{|N(u) \cap N(v)|}{\sqrt{|N(u)|\cdot|N(v)|}}
$$

如果只建立 user 对 item 的索引，形式如： `{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}`，计算两两用户的交互交集时，比较麻烦。建立倒排表形如：`{item_id1: {user_id1, user_id2, ... , user_idn}, item_id2: ...}` ，只需要对每个 item 遍历，就可以统计两两用户的交互交集。代码如下：


```python
# 建立item->users倒排表
# 倒排表的格式为: {item_id1: {user_id1, user_id2, ... , user_idn}, item_id2: ...} 也就是每个item对应有那些用户有过点击
# 建立倒排表的目的就是为了更方便的统计用户之间共同交互的商品数量
item_users = {}
for uid, items in tqdm(tra_users.items()): # 遍历每一个用户的数据,其中包含了该用户所有交互的item
    for item in items: # 遍历该用户的所有item, 给这些item对应的用户列表添加对应的uid
        if item not in item_users:
            item_users[item] = set()
            item_users[item].add(uid)
```

