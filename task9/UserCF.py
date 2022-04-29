import argparse
import pandas as pd
import numpy as np
import pickle
import math
import os 
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from collections import defaultdict
from tqdm import tqdm
from evaluate import rec_eval


def load_data(file_path):
    """
    加载数据
    """
    with open(os.path.join(file_path,'train_users.txt'), 'rb') as f1:
        train_users = pickle.load(f1)
    with open(os.path.join(file_path,'valid_users.txt'), 'rb') as f2:
        valid_users = pickle.load(f2) 

    return train_users, valid_users


def Cosine_UserCF(tra_users, val_users, K , N):
    """
    仅预测是否会评分，不包含具体rating
    """
    # 建立item->users倒排表
    # 倒排表的格式为: {item_id1: {user_id1, user_id2, ... , user_idn}, item_id2: ...} 也就是每个item对应有那些用户有过点击
    # 建立倒排表的目的就是为了更方便的统计用户之间共同交互的商品数量
    print('建立倒排表...')
    item_users = {}
    for uid, items in tqdm(tra_users.items()): # 遍历每一个用户的数据,其中包含了该用户所有交互的item
        for item in items: # 遍历该用户的所有item, 给这些item对应的用户列表添加对应的uid
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(uid)


    # 只要用户u,v共同交互过某个物品，它们之间的相似度就 +1  --->  协同过滤矩阵
    # 最后再除以 sqrt(N(u)*N(v))  --->  相似度矩阵


    # 计算用户协同过滤矩阵 
    # 即利用item-users倒排表统计用户之间交互的商品数量，用户协同过滤矩阵的表示形式为：sim = {user_id1: {user_id2: num1}, user_id3:{user_id4: num2}, ...}
    # 协同过滤矩阵是一个双层的字典，用来表示用户之间共同交互的商品数量
    # 在计算用户协同过滤矩阵的同时还需要记录每个用户所交互的商品数量，其表示形式为: num = {user_id1：num1, user_id2:num2, ...}
    sim = {}
    num = {}
    print('构建协同过滤矩阵...')
    for item, users in tqdm(item_users.items()): # 遍历所有的item去统计,用户两两之间共同交互的item数量
        for u in users:
            if u not in num: # 如果用户u不在字典num中，提前给其在字典中初始化为0,否则后面的运算会报key error
                num[u] = 0
            num[u] += 1 # 统计每一个用户,交互的总的item的数量
            if u not in sim: # 如果用户u不在字典sim中，提前给其在字典中初始化为一个新的字典,否则后面的运算会报key error
                sim[u] = {}
            for v in users:
                if u != v:  # 只有当u不等于v的时候才计算用户之间的相似度　
                    if v not in sim[u]:
                        sim[u][v] = 0
                    sim[u][v] += 1

    # 计算用户相似度矩阵 step2
    # 用户协同过滤矩阵其实相当于是余弦相似度的分子部分,还需要除以分母,即两个用户分别交互的item数量的乘积
    # 两个用户分别交互的item数量的乘积就是上面统计的num字典
    print('计算相似度...')
    for u, users in tqdm(sim.items()):
        for v, score in users.items():
            sim[u][v] =  score / math.sqrt(num[u] * num[v]) # 余弦相似度分母部分 
    

    # 对验证数据中的每个用户进行TopN推荐
    # 在对用户进行推荐之前需要先通过相似度矩阵得到与当前用户最相似的前K个用户，
    # 然后对这K个用户交互的商品中除当前测试用户训练集中交互过的商品以外的商品计算最终的相似度分数
    # 最终推荐的候选商品的相似度分数是由多个用户对该商品分数的一个累加和
    print('给测试用户进行推荐...')
    rec_dict = {}
    for u, _ in tqdm(val_users.items()): # 遍历测试集用户，给测试集中的每个用户进行推荐
        rec_dict[u] = {} # 初始化用户u的候选item的字典
        for v, score in sorted(sim[u].items(), key=lambda x: x[1], reverse=True)[:K]: # 选择与用户u最相似的k个用户
            for item in tra_users[v]: # 遍历相似用户之间交互过的商品
                if item not in tra_users[u]: # 如果相似用户交互过的商品，测试用户在训练集中出现过，就不用进行推荐，直接跳过
                    if item not in rec_dict[u]:
                        rec_dict[u][item] = 0   # 初始化用户u对item的相似度分数为０
                    rec_dict[u][item] += score  # 累加所有相似用户对同一个item的分数
    
    print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
    if not N:
        Top50_rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:50] for k, v in rec_dict.items()}
        Top50_rec_dict = {k: list([x[0] for x in v]) for k, v in Top50_rec_dict.items()}
        Top10_rec_dict = {k: v[:10] for k, v in Top50_rec_dict.items()}
        Top20_rec_dict = {k: v[:20] for k, v in Top50_rec_dict.items()}
        return Top10_rec_dict, Top20_rec_dict, Top50_rec_dict, rec_dict
    else:
        rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:N] for k, v in rec_dict.items()}
        rec_dict = {k: list([x[0] for x in v]) for k, v in rec_dict.items()}
        return rec_dict

def save_rec_dict(save_path,rec_dict):
    pickle.dump(rec_dict, open(os.path.join(save_path,'UserCF_rec_dict.txt'), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--TopN', type=int, default=0, help='number of top score items selected')
    parser.add_argument('--TopK', type=int, default=100, help='number of similar items/users')
    parser.add_argument('--rmse', action='store_true', help='number of similar items/users')
    args = parser.parse_args()

    train_users, valid_users = load_data(file_path='./data')

    if not args.TopN:
        Top10_rec_dict, Top20_rec_dict, Top50_rec_dict, rec_dict = Cosine_UserCF(train_users, valid_users, args.TopK ,args.TopN)
        print('Top 10:')
        rec_eval(Top10_rec_dict,valid_users,train_users)
        print('Top 20:')
        rec_eval(Top20_rec_dict,valid_users,train_users)
        print('Top 50:')
        rec_eval(Top50_rec_dict,valid_users,train_users)
        save_rec_dict('./data',rec_dict)
        print('Done.')
    else:
        rec_dict = Cosine_UserCF(train_users, valid_users, args.TopK ,args.TopN)
        print(f'Top {args.TopN}:')
        rec_eval(rec_dict,valid_users,train_users)
        print('Done.')
