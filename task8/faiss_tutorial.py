import numpy as np


#================计算向量距离=================#
"""第一步: 得到向量(向量库和query)"""
d = 64                           # 向量维度
nb = 100000                      # index 向量库的向量数量
nq = 10000                       # 待检索的 query 向量的数量
np.random.seed(1234)             
xb = np.random.random((nb, d)).astype('float32')    
xb[:, 0] += np.arange(nb) / 1000.           # index 向量库的向量           
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.           # 待检索的 query 向量

"""第二步: 构建索引"""
import faiss                   
index = faiss.IndexFlatL2(d)   # 构建索引, 相似度方法为L2范数
print(index.is_trained)        # 输出为 True, 代表该类 index 不需要训练, 只需要 add 向量进去即可
index.add(xb)                  # 将向量库中的向量加入到 index 中
print(index.ntotal)            # 输出index中包含的向量总数, 为 100000 

"""第三步: 检索TopK相似向量"""
k = 4                          # TopK 的值
D, I = index.search(xb[:5], k) # 健全性检查
print(I)
print(D)
D, I = index.search(xq, k)     # index.search() 实际检索, xq 为待检索向量, 
                               # I: (Index) 与 query 最相似的 TopK 个向量的索引的 list, list长度为 query 个数
                               # D: (Distance) 对应的 query 和向量的距离

print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries


#================计算向量内积=================#
d = 64   # 向量维度
nb = 1000000  # 数据大小
k = 6 # 查询top 5个最近邻
np.random.seed(1000)
train_vectors = np.random.random((nb, d)).astype('float32')   # 产生随机数，维度为nb x d
print(train_vectors[0:1])

# 精确的内积搜索，对归一化向量计算余弦相似度
faiss.normalize_L2(train_vectors)   # 归一化
index = faiss.IndexFlatIP(d)  # 内积建立索引
index.add(train_vectors)  # 添加矩阵

D, I = index.search(train_vectors[:100], k)  # 基于索引，对前5行数据，进行K近邻查询
print(D[:5])  # 距离矩阵
print(I[:5])  # 索引矩阵

index = faiss.IndexFlatL2(d)
print(index.is_trained)
index.add(train_vectors) 
print(index.ntotal)  # 查看建立索引的向量数目

D, I = index.search(train_vectors[:100], k)  # 基于索引，对前5行数据，进行K近邻查询
print(D[:5])
print(I[:5])