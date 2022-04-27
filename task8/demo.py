import faiss
from faiss import normalize_L2
import numpy as np

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