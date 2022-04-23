# 任务2：Movielens介绍

- 下载并读取Movielens 1M数据集（用户、电影、评分）
- 统计如下指标：
  - 总共包含多少用户？
  - 总共包含多个电影？
  - 平均每个用户对多少个电影进行了评分？
  - 每部电影 & 每个用户的平均评分是？
- 如果你来进行划分数据集为训练和验证，你会如何划分？

## 统计指标

- 总共包含 6040 个用户
- 总共包含 3883 部电影
- 平均每个用户对 165.6 部电影进行评分

其余细节见 notebook 。

## 划分数据集

参考项亮《推荐系统实践》，将用户行为数据集按照均匀分布随机分成 K 份，挑选一份作为测试集，剩下的 K-1 份作为训练集，进行 K 次实验，然后将 K 次评测指标的平均值作为最终评测指标 （即 K-fold 交叉验证）。

```python
def SplitData(data, K, i, seed):
    test = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0,K)==i:
            test.append([user,item])
        else:
            train.append([user,item])
    return train,test
```

