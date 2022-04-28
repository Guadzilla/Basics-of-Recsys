# 任务9：多路召回实践

- 基于任务3、任务5、任务6、任务7、任务8，总共5个召回模型，进行多路召回。
- 可以考虑对每个召回模型的物品打分进行相加，也可以加权求和。
- 分别计算每个模型 & 多路召回模型的Top10、Top20、Top50的命中率。

代码地址： https://github.com/Guadzilla/recpre

根据前面完成的任务，5个召回模型分别是 ItemCF、UserCF、SVD、SlopeOne、Word2Vec。

**任务目标**：对用户是否评分做出预测，即评分只有0和1（数据集中用户评分过的电影评分都取1，未评分过的电影评分都取0），进行推荐 TopN 推荐。

这样的话 SlopeOne 没法做了，因为评分只有0和1，有评分的哪些物品的评分都是1，物品评分之间没有均差。当然也可以把任务改为预测5分制的评分；或者两阶段预测，先预测是否会评分，再预测会评多少分。那样都需要对模型做不少修改，暂时没有精力做，图省事就只实现了4个召回模型 ItemCF、UserCF、SVD、Word2Vec。

**进行多路召回（模型融合），步骤大致分为以下几步**：

1. 划分数据集并保存
2. 各模型读取数据进行训练
3. 各模型进行预测，保存评估指标、保存预测结果
4. 读取所有预测结果进行加权（取均值、或者训练得到权重），进行最终评估

## 划分数据集

多路召回，首先保证各个模型上数据集划分是一致的。所以考虑把划分数据集这一部分独立出来单独运行。划分完的数据要保存下来，方便后续各个模型加载并独立做实验。

```python
def load_and_save(load_path, save_path, test_rate = 0.1):
    """
    加载数据，分割训练集、验证集
    """
    data = pd.read_table(os.path.join(load_path,'ratings.dat'), sep='::', names = ['userID','itemID','Rating','Zip-code'])
    data = data[['userID','itemID']]
    uid_lbe = LabelEncoder()
    data['userID'] = uid_lbe.fit_transform(data['userID'])
    iid_lbe = LabelEncoder()
    data['itemID'] = iid_lbe.fit_transform(data['itemID'])

    train_data, valid_data = train_test_split(data,test_size=test_rate)
    
    train_users = train_data.groupby('userID')['itemID'].apply(list).to_dict()
    valid_users = valid_data.groupby('userID')['itemID'].apply(list).to_dict()

    train_data.to_csv('train_data.csv', index=False)
    valid_data.to_csv('valid_data.csv', index=False)
    pickle.dump(train_users, open(os.path.join(save_path,'train_users.txt'), 'wb'))
    pickle.dump(valid_users, open(os.path.join(save_path,'valid_users.txt'), 'wb'))

load_and_save(load_path='../ml-1m', save_path='./data', test_rate=0.1)
```

## 训练各模型

```python
ItemCF.py   	-->   基于物品的协同过滤算法
UserCF.py   	-->   基于用户的协同过滤算法
MF.py			-->   梯度下降矩阵分解算法
Word2Vec.py		-->	  word2vec算法
# 各模型细节就不展示了
```

## 保存各模型指标

各模型在验证集上的评估指标为：

![image-20220428195502828](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220428195502828.png)

保存各模型的预测结果，注意保存的是没有截断 TopN 的推荐列表：

```python
def save_rec_dict(save_path,rec_dict):
    pickle.dump(rec_dict, open(os.path.join(save_path,'word2vec_rec_dict.txt'), 'wb'))
```

保存的格式为字典：`Top50_rec_dict={uid1:{iid1:score,iid3:score,...}, uid2:{iid2:score,iid3:score,...},...}`

## 模型融合

观察各模型保存的预测得分结果，发现 MF 和另外几个算法得到的数据的值域不在同一区间，如果直接取均值的话， MF 的影响就很小了，所以先对每个用户的物品得分列表做 softmax 再取均值

![image-20220428202209856](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220428202209856.png)

```python
def load_rec_dict(file_path, model_name):
    with open(os.path.join(file_path,f'{model_name}_rec_dict.txt'),'rb') as f:
        Rec_dict = pickle.load(f)
    # Rec_dict:{uid1:[(iid1,score),(iid2,score),...],uid2:[(...),(...),...],...}
    new_rec_dict = {}
    for uid, iid_score_list in Rec_dict.items():
        new_rec_dict[uid] = dict()
        score_list = []
        iid_list = []
        for iid, score in iid_score_list.items():
            score_list.append(score)
            iid_list.append(iid)
        prob = softmax(score_list)      # 把物品评分转化成概率
        for iid, p in zip(iid_list, prob):
            new_rec_dict[uid][iid] = p
    return new_rec_dict
```

加权取平均融合：

```python
def mixup(*rec_dicts):
    new_rec_dict = {}
    num = {}    # 记录目标用户被推荐同一商品几次，后续做除数
    for rec_dict in tqdm(rec_dicts):
        for uid in rec_dict:
            if uid not in new_rec_dict:
                new_rec_dict[uid] = {}
                num[uid] = {}
            for iid, prob in rec_dict[uid].items():
                if iid not in new_rec_dict[uid]:
                    new_rec_dict[uid][iid] = 0
                    num[uid][iid] = 0
                new_rec_dict[uid][iid] += prob
                num[uid][iid] += 1
    for uid,iid_score in new_rec_dict.items():
        for iid in iid_score:
            new_rec_dict[uid][iid] /= num[uid][iid]
    return new_rec_dict

new_rec_dict = mixup(*models_rec_dict)
```

TopN 推荐：

```python
# TopN推荐
Top50_rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:50] for k, v in new_rec_dict.items()}
Top50_rec_dict = {k: list([x[0] for x in v]) for k, v in Top50_rec_dict.items()}
Top10_rec_dict = {k: v[:10] for k, v in Top50_rec_dict.items()}
Top20_rec_dict = {k: v[:20] for k, v in Top50_rec_dict.items()}

# 计算评价指标
def bag_eval(val_rec_items, val_user_items, trn_user_items):
    recall = Recall(val_rec_items, val_user_items)
    precision = Precision(val_rec_items, val_user_items)
    coverage = Coverage(val_rec_items, trn_user_items)
    popularity = Popularity(val_rec_items, trn_user_items)
    print(f'{recall}\t{precision}\t{coverage}\t{popularity}',end='\t')

print('Top 10,20,50:')
bag_eval(Top10_rec_dict,valid_users,train_users)
bag_eval(Top20_rec_dict,valid_users,train_users)
bag_eval(Top50_rec_dict,valid_users,train_users)
print('\nDone.')
```

实验结果：最优结果红色加粗，次优结果橙色加粗。前四行是单模型，后面的都是融合模型，取每个模型的首字母表示每个模型。

![image-20220428234536222](https://wjm-images.oss-cn-beijing.aliyuncs.com/img-hosting/image-20220428234536222.png)

居然是单模型UserCF最优...
