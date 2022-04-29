# 任务7：词向量基础

- 学习[word2vec基础](https://cloud.tencent.com/developer/article/1486055)
- 将用户历史观看电影转为列表数据（一个用户一个列表）
- 使用gensim训练word2vec，然后对用户完成聚类

代码地址： https://github.com/Guadzilla/recpre

# 学习资料

来源 | Analytics Vidhya 【磐创AI导读】：这篇文章主要介绍了如何使用word2vec构建推荐系统。想要获取更多的机器学习、深度学习资源，欢迎大家点击上方蓝字关注我们的公众号：磐创AI。

## **概览**

- 如今，推荐引擎无处不在，人们希望数据科学家知道如何构建一个推荐引擎
- Word2vec是一个非常流行的词嵌入，用于执行各种NLP任务
- 我们将使用word2vec来构建我们自己的推荐系统。就让我们来看看NLP和推荐引擎是如何结合的吧！

完整的代码可以从这里下载：

> https://github.com/prateekjoshi565/recommendation_system/blob/master/recommender_2.ipynb

## 　**介绍**

老实说，你在亚马逊上有注意到网站为你推荐的内容吗（Recommended for you部分)? 自从几年前我发现机器学习可以增强这部分内容以来，我就迷上了它。每次登录Amazon时，我都会密切关注该部分。

Netflix、谷歌、亚马逊、Flipkart等公司花费数百万美元完善他们的推荐引擎是有原因的，因为这是一个强大的信息获取渠道并且提高了消费者的体验。

让我用一个最近的例子来说明这种作用。我去了一个很受欢迎的网上市场购买一把躺椅，那里有各种各样的躺椅，我喜欢其中的大多数并点击了查看了一把人造革手动躺椅。

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/aurj300r0x.jpeg)

请注意页面上显示的不同类型的信息，图片的左半部分包含了不同角度的商品图片。右半部分包含有关商品的一些详细信息和部分类似的商品。

而这是我最喜欢的部分，该网站正在向我推荐类似的商品，这为我节省了手动浏览类似躺椅的时间。

在本文中，我们将构建自己的推荐系统。但是我们将从一个独特的视角来处理这个问题。我们将使用一个NLP概念--Word2vec,向用户推荐商品。如果你觉得这个教程让你有点小期待，那就让我们开始吧！

在文中，我会提及一些概念。我建议可以看一下以下这两篇文章来快速复习一下

> [理解神经网络:](https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/?utm_source=blog&utm_medium=how-to-build-recommendation-system-word2vec-python)
>
> [构建推荐引擎的综合指南 ](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/?utm_source=blog&utm_medium=how-to-build-recommendation-system-word2vec-python)

## **word2vec - 词的向量表示**

我们知道机器很难处理原始文本数据。事实上，除了数值型数据，机器几乎不可能处理其他类型的数据。因此，以向量的形式表示文本几乎一直是所有NLP任务中最重要的步骤。

在这个方向上，最重要的步骤之一就是使用 word2vec embeddings，它是在2013年引入NLP社区的并彻底改变了NLP的整个发展。

事实证明，这些 embeddings在单词类比和单词相似性等任务中是最先进的。word2vec embeddings还能够实现像 `King - man +woman ~= Queen`之类的任务，这是一个非常神奇的结果。

有两种⁠word2vec模型——Continuous Bag of Words模型和Skip-Gram模型。在本文中，我们将使用Skip-Gram模型。

首先让我们了解word2vec向量或者说embeddings是怎么计算的。

## **如何获得word2vec embeddings?**

word2vec模型是一个简单的神经网络模型，其只有一个隐含层，该模型的任务是预测句子中每个词的近义词。然而，我们的目标与这项任务无关。我们想要的是一旦模型被训练好，通过模型的**隐含层学习到的权重**。然后可以将这些权重用作单词的embeddings。

让我举个例子来说明word2vec模型是如何工作的。请看下面这句话:

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/39q26nmjqb.png)

假设单词“teleport”(用黄色高亮显示)是我们的输入单词。它有一个大小为2的上下文窗口。这意味着我们只考虑输入单词两边相邻的两个单词作为邻近的单词。

*注意:上下文窗口的大小不是固定的，可以根据我们的需要进行更改。*

现在，任务是逐个选择邻近的单词(上下文窗口中的单词)，并给出词汇表中每个单词成为选中的邻近单词的概率。这听起来应该挺直观的吧？

让我们再举一个例子来详细了解整个过程。

#### **准备训练数据**

我们需要一个标记数据集来训练神经网络模型。这意味着数据集应该有一组输入和对应输入的输出。在这一点上，你可能有一些问题，像:

- 在哪里可以找到这样的数据集?
- 这个数据集包含什么?
- 这个数据有多大?

等等。

然而我要告诉你的是：我们可以轻松地创建自己的标记数据来训练word2vec模型。下面我将演示如何从任何文本生成此数据集。让我们使用一个句子并从中创建训练数据。

**第一步**: 黄色高亮显示的单词将作为输入，绿色高亮显示的单词将作为输出单词。我们将使用2个单词的窗口大小。让我们从第一个单词作为输入单词开始。

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/uqzmaohs5l.png)

所以，关于这个输入词的训练样本如下:

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/jikh7horxy.png)

**第二步**: 接下来，我们将第二个单词作为输入单词。上下文窗口也会随之移动。现在，邻近的单词是“we”、“become”和“what”。

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/rrq9cfy6m9.png)

新的训练样本将会被添加到之前的训练样本中，如下所示:

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/q4sl5f7xlo.png)

我们将重复这些步骤，直到最后一个单词。最后，这句话的完整训练数据如下:

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/djmiyovdx5.png)

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/sflnd9gu81.png)

我们从一个句子中抽取了27个训练样本，这是我喜欢处理非结构化数据的许多方面之一——凭空创建了一个标记数据集。

#### **获得 word2vec Embeddings**

现在，假设我们有一堆句子，我们用同样的方法从这些句子中提取训练样本。我们最终将获得相当大的训练数据。

假设这个数据集中有5000个惟一的单词，我们希望为每个单词创建大小为100维的向量。然后，对于下面给出的word2vec架构:

- V = 5000(词汇量)
- N = 100(隐藏单元数量或单词embeddings长度)

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/78ugqeahim.png)

输入将是一个**热编码向量**，而输出层将给出词汇表中**每个单词都在其附近的概率**。

一旦对该模型进行训练，我们就可以很容易地提取学习到的权值矩阵 

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/455ulptdh8.png)

x N，并用它来提取单词向量:

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/n01f88stii.png)

正如你在上面看到的，权重矩阵的形状为5000 x 100。这个矩阵的第一行对应于词汇表中的第一个单词，第二个对应于第二个单词，以此类推。

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/9sffijf6gc.png)

这就是我们如何通过word2vec得到固定大小的词向量或embeddings。这个数据集中相似的单词会有相似的向量，即指向相同方向的向量。例如，单词“car”和“jeep”有类似的向量:

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/b5dbjph3bi.png)

这是对word2vec如何在NLP中使用的高级概述。

在我们开始构建推荐系统之前，让我问你一个问题。如何将word2vec用于非nlp任务，如商品推荐?我相信自从你读了这篇文章的标题后，你就一直在想这个问题。让我们一起解出这个谜题。

## **在非文本数据上应用word2vec模型**

你能猜到word2vec用来创建文本向量表示的自然语言的基本特性吗?

是**文本的顺序性**。每个句子或短语都有一个单词序列。如果没有这个顺序，我们将很难理解文本。试着解释下面这句话:

> “these most been languages deciphered written of have already”

这个句子没有顺序，我们很难理解它，这就是为什么在任何自然语言中，单词的顺序是如此重要。正是这个特性让我想到了其他不像文本具有顺序性质的数据。

其中一类数据是**消费者在电子商务网站的购买行为**。大多数时候，消费者的购买行为都有一个模式，例如，一个从事体育相关活动的人可能有一个类似的在线购买模式:

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/e7v1rjzihf.png)

如果我们可以用向量表示每一个商品，那么我们可以很容易地找到相似的商品。因此，如果用户在网上查看一个商品，那么我们可以通过使用商品之间的向量相似性评分轻松地推荐类似商品。

但是我们如何得到这些商品的向量表示呢?我们可以用word2vec模型来得到这些向量吗?

答案当然是可以的! 把消费者的购买历史想象成一句话，而把商品想象成这句话的单词:

![img](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/v2vl4bkiqr.png)

更进一步，让我们研究在线零售数据，并使用word2vec构建一个推荐系统。

## **案例研究:使用Python中的word2vec进行在线商品推荐**

现在让我们再一次确定我们的问题和需求：

我们被要求创建一个系统，根据消费者过去的购买行为，自动向电子商务网站的消费者推荐一定数量的商品。

我们将使用一个在线零售数据集，你可以从这个链接下载:

> https://archive.ics.uci.edu/ml/machine-learning-databases/00352/

详细代码见：[recpre/word2vec_demo.ipynb at master · Guadzilla/recpre (github.com)](https://github.com/Guadzilla/recpre/blob/master/task7/word2vec_demo.ipynb) 因为和原文差不多，就不重复介绍了。

# 在 Movielens 数据集上用 Word2Vec 对用户聚类

见 word2vec_cluster.ipynb

### 导入相关包

```python
import os
import pandas as pd
import numpy as np
import random
import umap
import umap.plot
from tqdm import tqdm
from gensim.models import Word2Vec 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')	
```

### 加载数据集

最后要得到：所有用户的列表 `users=[user1,user2,...]`，语料库`corpus=[[item1,item2,...],[item2,item5,...],...]`，和商品字典：`items_dict={item1:"item1的描述", item2:"item2的描述",...}`

这里的语料库其实就是每个用户的购买序列组成的列表，我们把每个商品看作一个词，用户的一个购买序列看作一句话，通过这种方式构建语料库。

构建商品字典是为了方便后续查看相似物品信息。

```python
def load_data(file_path):
    df = pd.read_table('../dataset/ml-1m/ratings.dat', sep='::', names = ['userID','itemID','Rating','Zip-code'])
    movies = pd.read_table('../dataset/ml-1m/movies.dat',sep='::',names=['MovieID','Title','Genres'],encoding='ISO-8859-1')
    movies['content'] = movies['Title'] + '__' + movies['Genres']

    # 所有userID
    users = df["userID"].unique().tolist()

    # 存储user的购买历史,每个user对应一个list,每个list当作一句话,所有list作为语料库
    corpus = []
    for i in tqdm(users):
        temp = df[df["userID"] == i]["itemID"].tolist()
        corpus.append(temp)

    # 建立商品字典,方便后续查看相似物品信息
    items_dict = movies.groupby('MovieID')['content'].apply(list).to_dict()

    return users, corpus, items_dict
```

Movielens-1m 数据集很完整，没有缺失值要处理。

### 训练 Word2Vec

```python
# 训练word2vec模型
model = Word2Vec(window = 10, sg = 1, hs = 0, negative = 10, alpha=0.03, min_alpha=0.0007, seed = 14)
model.build_vocab(corpus, progress_per=200)
model.train(corpus, total_examples = model.corpus_count, epochs=10, report_delay=1)
# 模型训练完成, init_sims()提高内存运行效率
model.init_sims(replace=True)
```

### 查看模型参数

共有3416个 item embedding ,每个维度为100.

```python
# 打印模型
print(model)

# Word2Vec(vocab=3416, vector_size=100, alpha=0.03)
```

### 查看相似物品

首先提取所有向量

```python
# 提取向量
X = model.wv[model.wv.key_to_index.keys()]
```

查看 movies 信息，这里我们就选第一个电影 "Toy Story"，看看它的相似电影

```python
movies = pd.read_table('../dataset/ml-1m/movies.dat',sep='::',names=['MovieID','Title','Genres'],encoding='ISO-8859-1')
movies.head()
```

<img src="https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220427172027663.png" alt="image-20220427172027663" style="zoom:67%;" />

Toy Story 的 key 就是对应的 MovieID = 1，通过模型内置的字典找到它对应的索引为 29

```python
model.wv.key_to_index[1]

# 29
```

计算并返回与它相似的10部电影

```python
def similar_products(v, n = 10):
    """
    返回最相似的n个物品
    """
    # 为输入向量提取最相似的商品
    ms = model.wv.similar_by_vector(v, topn= n+1)[1:]
    # 提取相似产品的名称和相似度评分
    new_ms = []
    for j in ms:
        pair = (items_dict[j[0]][0], j[1])
        new_ms.append(pair)
    return new_ms   

for i in similar_products(X[29]):
    print(i[0])

"""
Aladdin (1992)__Animation|Children's|Comedy|Musical
Silence of the Lambs, The (1991)__Drama|Thriller
Train of Life (Train De Vie) (1998)__Comedy|Drama
Home Alone (1990)__Children's|Comedy
Jumanji (1995)__Adventure|Children's|Fantasy
Waiting to Exhale (1995)__Comedy|Drama
Ghost (1990)__Comedy|Romance|Thriller
Beavis and Butt-head Do America (1996)__Animation|Comedy
Tom and Huck (1995)__Adventure|Children's
Brady Bunch Movie, The (1995)__Comedy
"""
```

Toy Story 的类别是 Animation|Children's|Comedy ，与它相似的电影也大都是动画片、儿童电影，说明 word2vec 一定程度上的确给相似的电影学到了相似的embedding.

其中第2部 Silence of the Lambs , 现实中喜欢玩具总动员的大多是有冒险精神的大人，所以性格的另一面喜欢沉默的羔羊也是可以的（狡辩）

### 对物品和用户向量进行聚类

定义用来聚类和画图的函数

```python
# 两种画图方法
def visualize_emb(X):
    cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
    n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(10,9))
    plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], s=3, cmap='Spectral')
    plt.show()


def umap_plot_emb(X):
    cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
    n_components=2, random_state=42).fit(X)
    umap.plot.points(cluster_embedding)
```

#### 可视化物品向量

```python
visualize_emb(X)
umap_plot_emb(X)
```

<img src="https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220427164950421.png" alt="image-20220427164950421" style="zoom:50%;" /><img src="https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220427165014070.png" alt="image-20220427165014070" style="zoom: 41%;" />

居然类似一个圆？意思是接近随机生成吗？

#### 可视化用户向量

用户的向量可以用看过的电影的向量求均值表示，也可以对一部分求均值表示

```python
def aggregate_vectors(products):
    """
    返回用户向量
    """
    product_vec = []
    for i in products:
        try:
            product_vec.append(model.wv[i])
        except KeyError:
            continue
        return np.mean(product_vec, axis=0)
    
usersVec_all_item = []  # 对看过的所有电影的向量求均值
usersVec_last_10 = []   # 对看过的最后10个电影的向量求均值
for i in range(len(users)):
    usersVec_all_item.append(aggregate_vectors(corpus[i]))
    usersVec_last_10.append(aggregate_vectors(corpus[i][-10:]))
```

用户向量 = 看过的所有电影的向量的均值，可视化结果

```python
visualize_emb(usersVec_all_item)
umap_plot_emb(usersVec_all_item)
```

<img src="https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220427165406864.png" alt="image-20220427165406864" style="zoom:50%;" /><img src="https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220427165450274.png" alt="image-20220427165450274" style="zoom:41%;" />

用户向量 = 看过的最后10部电影的向量的均值，可视化结果

```python
visualize_emb(usersVec_last_10)
umap_plot_emb(usersVec_last_10)
```

<img src="https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220427165724232.png" alt="image-20220427165724232" style="zoom:50%;" /><img src="https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220427165735996.png" alt="image-20220427165735996" style="zoom:41%;" />

至此完成了利用word2vec训练得到物品的embedding，再用物品embedding表示用户embedding，最后对用户embedding聚类。

# 在 Movielens 数据集上用 Word2Vec 对用户进行推荐

见 word2vec_rec.py

如果要对用户推荐商品，就要比上面的简单对用户进行聚类稍微麻烦一点，因为需要划分训练集和验证集。划分数据集的代码如下：

对数据集进行划分，我们在这一步需要获取的是，训练集、测试集出现了哪些用户，训练集的语料库，以及商品字典。

```python

def load_data(file_path):
    data = pd.read_table('../dataset/ml-1m/ratings.dat', sep='::', names = ['userID','itemID','Rating','Zip-code'])
    movies = pd.read_table('../dataset/ml-1m/movies.dat',sep='::',names=['MovieID','Title','Genres'],encoding='ISO-8859-1')
    movies['content'] = movies['Title'] + '__' + movies['Genres']
    # 
    tra_data, val_data = train_test_split(data, test_size=0.2)
    users_train = tra_data['userID'].unique().tolist()
    users_valid = val_data['userID'].unique().tolist()

    """训练集"""
    # 存储消费者的购买历史
    train_users = {}
    train_corpus = []
    # 用 itemID 填充列表
    for i in tqdm(users_train):
        temp = tra_data[tra_data["userID"] == i]["itemID"].tolist()
        train_users[i] = temp
        train_corpus.append(temp)
    """验证集"""
    # 存储消费者的购买历史
    valid_users = {}
    valid_corpus = []
    # 用商品代码填充列表
    for i in tqdm(users_valid):
        temp = val_data[val_data["userID"] == i]["itemID"].tolist()
        valid_users[i] = temp
        valid_corpus.append(temp)
    """建立商品字典"""
    items_dict = movies.groupby('MovieID')['content'].apply(list).to_dict()

    return train_users, train_corpus, valid_users, valid_corpus, items_dict
```

其它步骤与前面的相似，具体看代码吧。

```python

train_users, train_corpus, valid_users, valid_corpus, items_dict = load_data('../dataset/ml-1m')
# train_users: {user1:[item1,item2,...],user2:[item2,item5,...],...}
# train_corpus: [[item1,item2,...],[item2,item5,...],...]
# 训练word2vec模型
model = Word2Vec(window = 10, sg = 1, hs = 0, negative = 10, alpha=0.03, min_alpha=0.0007, seed = 14)
model.build_vocab(train_corpus, progress_per=200)
model.train(train_corpus, total_examples = model.corpus_count, epochs=10, report_delay=1)
model.init_sims(replace=True)
# 打印模型
print(model)
# 提取向量
X = model.wv[model.wv.key_to_index.keys()]
# 可视化
#visualize_emb(X)



def similar_products_idx(v, n = 10):
    """
    返回最相似的n个物品
    """
    # 为输入向量提取最相似的商品
    ms = model.wv.similar_by_vector(v, topn= n+1)[1:]
    return ms  

def aggregate_vectors(products):
    """
    返回购买记录的平均向量
    """
    product_vec = []
    for i in products:
        try:
            product_vec.append(model.wv[i])
        except KeyError:
            continue
            return np.mean(product_vec, axis=0)
    
# 推荐TopN相似商品
rec_dict = {}
rel_dict = {}
for user in valid_users:    # valid_users:{user1:[item1,item2,...],user2:[item2,item5,...],...}
    if user not in train_users:
        continue
    user_vec = aggregate_vectors(train_users[user][-10:])
    similar_items = similar_products_idx(user_vec,10)
    rec_dict[user] = [x[0] for x in similar_items]
    rel_dict[user] = valid_users[user]

# evaluate
rec_eval(rec_dict,rel_dict,train_users)

```

评估结果：

```python
recall: 1.31
precision 4.34
coverage 71.29
Popularity 4.93
```

