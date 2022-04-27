# 任务8：向量召回基础

- 基于任务7的基础上，使用编码后的用户向量，计算用户相似度。
- 参考User-CF的过程，通过用户相似度得到电影推荐

代码地址： https://github.com/Guadzilla/recpre

任务 7 中，我们在 Movielens-1m 数据集上用 word2vec 训练得到了 电影的"词向量"（物品向量、item embedding，都是一个意思，下面都用 item embedding），再对用户所有看过的电影的 embedding 取均值作为”句向量“（用户向量，user embedding）。

在任务 7 的 word2vec_rec.py 里，实现了在 Movielens 数据集上用 Word2Vec 对用户进行推荐，主要步骤为：**划分数据集、训练word2vec模型、计算用户向量**、TopN推荐（计算相似物品）。

任务 8 就以它为基础，稍作修改，主要步骤为：**划分数据集、训练word2vec模型、计算用户向量**、计算用户相似度、建立向量库、构建用户相似性矩阵、利用UserCF算法进行TopN推荐。因为前三步骤和之前类似，不做过多介绍，这里重点介绍后四步。

## 计算用户相似度

计算用户相似度的思路是，利用 faiss 库建立向量库，存入所有 user embedding，再用 user embedding 对向量库进行检索，取出 TopK 个最相似的用户，用户后续的 UserCF 推荐。

因为用 faiss 构建向量库会对 userID 重新从0开始索引，取出结果也是根据重新索引后的 index 来取，所以有必要提前对训练集的 userID 重建索引，用一个字典映射来实现。

实际上在 load 数据集的时候已经对 userID 和 itemID 重新索引了，这里为什么还要再做一次索引呢？

答：为了方便索引，使得训练集和验证集可以直接对向量库进行索引。具体来说，划分数据集会造成训练集和验证集上的用户数量不一致，也就是说可能有一部分用户只在验证集出现，训练集里没有他。例如：全部数据集有10个用户，并且已经对他们从零开始编号，`userID=[0,1,2,3,4,5,6,7,8,9]`。随机划分数据集以后，训练集里可能就只有`userID = [0,2,3,4,5,6,7,8]` 这8个用户了，验证集里只有 `userID = [0,1,2,3,4,5,6,7,9]` 9个用户。此时如果直接把训练集的 user embedding 直接放到索引库里，faiss 为向量库构建的索引为`index = [0,1,2,3,4,5,6,7]`，对应关系是`index=0对应userID=0`，`index=1对应userID=2`，...，这个对应关系必须保存下来，否则验证集只有 userID ，无法定位到 userID 对应哪个索引，也就无法提取该 user embedding。如果用字典 `MAP(key=userID,value=index)`保存下来这个对应关系，在训练集上就可以用`MAP[userID]`作为索引从向量库提取 user embedding ；验证集上，首先把在验证集首次出现的用户单独保存，剩下`userID = [0,2,3,4,5,6,7]`，其余一样。

在这里使用了另一种思路，先对训练集 `userID = [0,2,3,4,5,6,7,8]` 重新索引成 `userID = [0,1,2,3,4,5,6,7]` ，并保存下这个字典`MAP(key=userID,value=index)`，再把验证集  `userID = [0,1,2,3,4,5,6,7,9]` 的 userID 都 MAP 到 index 上，其中 1 和 9 在训练集上没有出现，单独保存，剩下的`userID = [0,2,3,4,5,6,7]`再做映射，变成 `userID = [0,1,2,3,4,5,6]`，这样做方便在以后就可以直接用 userID 访问向量库。

两种方式都可以，第一种方式的缺点是频繁访问字典，但优点是不用 in place 修改数据；第二种方式正好相反，不用频繁访问字典，但是需要 in place 修改数据。个人习惯用第二种方式，一劳永逸。

下面是代码部分：

```python
TopK = 100
TopN = 10

# 对train_user重建索引, 求用户向量, 求重索引的new_train_users, 只需要一次循环就能做完
# 已有 train_users  # train_users: {user1:[item1,item2,...], user2:[item3,item4,...],...}
count = 0
MAP = dict()   # 对train_user重建索引,从0开始   MAP: {userID:faiss_idx, ...}
usersVec = []  # 求用户向量, aggregate_vectors():对看过的所有电影的向量求均值
new_train_users = {}    # new_train_users: {faiss_idx1:[item1,item2,...],faiss_idx2:[item3,item4,...],...}
for user,items in train_users.items():
    MAP[user] = count
    new_train_users[count] = items
    usersVec.append(aggregate_vectors(train_corpus[count]))
    count += 1
del train_users

# 对valid_users的userid按MAP重索引,得到new_valid_users: {faiss_idx1:[item1,item2,...],faiss_idx2:[item3,item4,...],...}
new_valid_users = {}
for user,items in valid_users.items():
    if user not in new_train_users:
        continue
    new_valid_users[MAP[user]] = items
del valid_users
```

## 建立向量索引库

要用 faiss 计算余弦相似度，需要注意的是 faiss 自带的两种常见相似算法是：`faiss.IndexFlatL2()`用来计算向量距离，和` faiss.IndexFlatIP()`用来计算向量内积。计算余弦相似度可以用向量内积形式，但前提是需要先把向量转成单位向量，faiss 自带了 `faiss.normalize_L2()`就是用来单位化向量的。

```python
# 建立用户向量索引库
import faiss
usersVec = np.array(usersVec)
normalize_L2(usersVec)   # 这是inplace操作 (计算余弦相似度,先正则化,再计算内积)
index = faiss.IndexFlatIP(100)
index.add(usersVec)
```

## 构建用户相似性矩阵

构建用户相似性的思路是用字典保存，因为矩阵太稀疏，浪费空间。

```python
# 找TopK相似用户
D, I = index.search(usersVec[list(new_valid_users.keys())], TopK+1) # TopK要+1,因为底下计算相似度会计算自身一次
similar_users_idxs = I
similar_users = {}  # similar_users: {faiss_idx1:{faiss_idx2:score,faiss_idx4:score,...},faiss_idx2:{...},...}
for idx,val_user in enumerate(new_valid_users):
    similar_users[val_user] = dict()
    for iidx,s_user in enumerate(similar_users_idxs[idx]):
        similar_users[val_user][s_user] = D[idx][iidx]
```

## 利用 UserCF 算法进行 TopN 推荐

UserCF 的思路，先找到和目标用户最相似的 TopK 个用户（相似度用于计算目标用户对陌生物品的得分）再计算这些用户看过的、且目标用户没看过的电影的得分，对有得分的物品进行排序，进行 TopN 推荐。

```python
# 推荐TopN相似商品
print('开始生成推荐列表...')
rec_dict = {}
rel_dict = new_valid_users
for val_user in tqdm(new_valid_users):   # new_valid_users: {faiss_idx1:[item1,item2,...],faiss_idx2:[item3,item4,...],...}
    rec_dict[val_user] = dict()
    for user in similar_users[val_user]:
        for item in new_train_users[user]:
            if item not in new_train_users[val_user]:
                if item not in rec_dict[val_user]:
                    rec_dict[val_user][item]=0
                rec_dict[val_user][item] += similar_users[val_user][user]
# rec_dict: {faiss_idx1:{item2:score,item4:score,...},faiss_idx2:{...},...}
# 先选出每个user的TopN "item-score" 对，再提出item到最后的推荐列表, 变换后的rec_dict: {faiss_idx1:[item2,item4,...],faiss_idx2:[item3,item4,...],...}
rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:TopN] for k, v in rec_dict.items()}
rec_dict = {k: list([x[0] for x in v]) for k, v in rec_dict.items()}
```

最后进行评估，验证集上实验结果：

```python
# evaluate
rec_eval(rec_dict,rel_dict,new_train_users)
"""
recall: 6.01
precision 19.91
coverage 9.38
Popularity 7.459
"""
```

完整代码见 [recpre/task8 at master · Guadzilla/recpre (github.com)](https://github.com/Guadzilla/recpre/tree/master/task8/word2vec_recall.py)
