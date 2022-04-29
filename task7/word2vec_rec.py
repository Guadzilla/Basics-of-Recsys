from audioop import rms
import os
import pandas as pd
import numpy as np
import random
import umap
from tqdm import tqdm
from gensim.models import Word2Vec 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from evaluate import rec_eval

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


def visualize_emb(X):
    cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
    n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(10,9))
    plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], s=3, cmap='Spectral')
    plt.show()



if __name__ == "__main__":
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
