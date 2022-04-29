import os
import pandas as pd
import numpy as np
import faiss
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


def users_similarity(user_emb):
    pass



if __name__ == "__main__":
    train_users, train_corpus, valid_users, valid_corpus, items_dict = load_data('../dataset/ml-1m')
    # train_users: {user1:[item1,item2,...],user2:[item2,item5,...],...}
    # train_corpus: [[item1,item2,...],[item2,item5,...],...]
    # 训练word2vec模型
    print('开始训练word2vec...')
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


    #=====================UserCF 推荐=======================#
    TopK = 100
    TopN = 10

    # 对train_user重建索引, 求用户向量, 求重索引的new_train_users, 只需要一次循环就能做完
    # 已有 train_users  # train_users: {user1:[item1,item2,...], user2:[item3,item4,...],...}
    count = 0
    MAP = dict()   # 对train_user重建索引,从0开始   MAP: {userID:faiss_idx, ...}
    usersVec = []  # 求用户向量, 即对看过的所有电影的向量求均值
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

    # 建立用户向量索引库
    import faiss
    usersVec = np.array(usersVec)
    faiss.normalize_L2(usersVec)   # 这是inplace操作 (计算余弦相似度,先正则化,再计算内积)
    index = faiss.IndexFlatIP(100)
    index.add(usersVec)

    # 找TopK相似用户
    D, I = index.search(usersVec[list(new_valid_users.keys())], TopK+1) # TopK要+1,因为底下计算相似度会计算自身一次
    similar_users_idxs = I
    similar_users = {}  # similar_users: {faiss_idx1:{faiss_idx2:score,faiss_idx4:score,...},faiss_idx2:{...},...}
    for idx,val_user in enumerate(new_valid_users):
        similar_users[val_user] = dict()
        for iidx,s_user in enumerate(similar_users_idxs[idx]):
            similar_users[val_user][s_user] = D[idx][iidx]

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
    
    # evaluate
    rec_eval(rec_dict,rel_dict,new_train_users)
