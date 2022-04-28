import os
import pandas as pd
import numpy as np
import pickle
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
    """
    加载数据
    """
    with open(os.path.join(file_path,'train_users.txt'), 'rb') as f1:
        train_users = pickle.load(f1)
    with open(os.path.join(file_path,'valid_users.txt'), 'rb') as f2:
        valid_users = pickle.load(f2) 

    train_corpus = []
    for uid,iid_list in train_users.items():
        train_corpus.append(iid_list)


    return train_users, train_corpus, valid_users


def save_rec_dict(save_path,rec_dict):
    pickle.dump(rec_dict, open(os.path.join(save_path,'Word2Vec_rec_dict.txt'), 'wb'))
    

if __name__ == "__main__":
    train_users, train_corpus, valid_users = load_data('./data')
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
    TopK = 50
    TopN = 10

    # 对train_user重建索引, 求用户向量, 求重索引的new_train_users, 只需要一次循环就能做完
    # 已有 train_users  # train_users: {user1:[item1,item2,...], user2:[item3,item4,...],...}
    count = 0
    MAP = dict()   # 对train_user重建索引,从0开始   MAP: {userID:faiss_idx, ...}
    """添加了inverse_MAP"""
    inverse_MAP = dict()    # inverse_MAP: {faiss_idx:userID, ...}
    usersVec = []  # 求用户向量, 即对看过的所有电影的向量求均值
    new_train_users = {}    # new_train_users: {faiss_idx1:[item1,item2,...],faiss_idx2:[item3,item4,...],...}
    for user,items in train_users.items():
        inverse_MAP[count] = user
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
    """把rec_dict、rel_dict的key还原成userID"""
    new_rec_dict = {}
    new_rel_dict = {}
    for faiss_idx, rec_list in rec_dict.items():
        userID = inverse_MAP[faiss_idx] # 从向量索引还原成userID
        new_rec_dict[userID] = rec_list
    for faiss_idx, rel_list in rel_dict.items():
        userID = inverse_MAP[faiss_idx] # 从向量索引还原成userID
        new_rel_dict[userID] = rel_list
    rec_dict = new_rec_dict
    rel_dict = new_rel_dict
    del new_rec_dict
    del new_rel_dict
    
    print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
    Top50_rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:50] for k, v in rec_dict.items()}
    Top50_rec_dict = {k: list([x[0] for x in v]) for k, v in Top50_rec_dict.items()}
    Top10_rec_dict = {k: v[:10] for k, v in Top50_rec_dict.items()}
    Top20_rec_dict = {k: v[:20] for k, v in Top50_rec_dict.items()}
    # evaluate
    print('Top 10:')
    rec_eval(Top10_rec_dict,rel_dict,new_train_users)
    print('Top 20:')
    rec_eval(Top20_rec_dict,rel_dict,new_train_users)
    print('Top 50:')
    rec_eval(Top50_rec_dict,rel_dict,new_train_users)
    save_rec_dict('./data',rec_dict)
    print('Done.')