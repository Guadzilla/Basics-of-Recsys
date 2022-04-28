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


def Cosine_Item_CF(trn_user_items, val_user_items, K, N):
    '''
    trn_user_items: 表示训练数据，格式为：{user_id1: [item_id1, item_id2,...,item_idn], user_id2...}
    val_user_items: 表示验证数据，格式为：{user_id1: [item_id1, item_id2,...,item_idn], user_id2...}
    K: Ｋ表示的是相似商品的数量，为每个用户交互的每个商品都选择其最相思的K个商品
    N: N表示的是给用户推荐的商品数量，给每个用户推荐相似度最大的N个商品
    '''

    # 建立user->item的倒排表
    # 倒排表的格式为: {user_id1: [item_id1, item_id2,...,item_idn], user_id2: ...} 也就是每个用户交互过的所有商品集合
    # 由于输入的训练数据trn_user_items,本身就是这中格式的，所以这里不需要进行额外的计算
    

    # 只要物品u,v共同被某个用户交互过，它们之间的相似度就 +1  --->  协同过滤矩阵
    # 最后再除以 sqrt(N(u)*N(v))  --->  相似度矩阵


    # 计算商品协同过滤矩阵  
    # 即利用user-items倒排表统计商品与商品之间被共同的用户交互的次数
    # 商品协同过滤矩阵的表示形式为：sim = {item_id1: {item_id２: num1}, item_id３: {item_id４: num２}, ...}
    # 商品协同过滤矩阵是一个双层的字典，用来表示商品之间共同交互的用户数量
    # 在计算商品协同过滤矩阵的同时还需要记录每个商品被多少不同用户交互的次数，其表示形式为: num = {item_id1：num1, item_id２:num2, ...}
    sim = {}
    num = {}
    print('构建相似性矩阵...')
    for uid, items in tqdm(trn_user_items.items()):
        for i in items:    
            if i not in num:
                num[i] = 0
            num[i] += 1
            if i not in sim:
                sim[i] = {}
            for j in items:
                if j not in sim[i]:
                    sim[i][j] = 0
                if i != j:
                    sim[i][j] += 1
    
    # 计算物品的相似度矩阵 step 2
    # 商品协同过滤矩阵其实相当于是余弦相似度的分子部分,还需要除以分母,即两个商品被交互的用户数量的乘积
    # 两个商品被交互的用户数量就是上面统计的num字典
    print('计算协同过滤矩阵．．．')
    for i, items in tqdm(sim.items()):
        for j, score in items.items():
            if i != j:
                sim[i][j] = score / math.sqrt(num[i] * num[j])
    

    # 对验证数据中的每个用户进行TopN推荐
    # 在对用户进行推荐之前需要先通过商品相似度矩阵得到当前用户交互过的商品最相思的前K个商品，
    # 然后对这K个用户交互的商品中除当前测试用户训练集中交互过的商品以外的商品计算最终的相似度分数
    # 最终推荐的候选商品的相似度分数是由多个相似商品对该商品分数的一个累加和
    rec_dict = {}
    print('给用户进行推荐．．．')
    for uid, _ in tqdm(val_user_items.items()):
        rec_dict[uid] = {} # 存储用户候选的推荐商品
        for hist_item in trn_user_items[uid]: # 遍历该用户历史喜欢的商品，用来下面寻找其相似的商品
            for item, score in sorted(sim[hist_item].items(), key=lambda x: x[1], reverse=True)[:K]:
                if item not in trn_user_items[uid]: # 进行推荐的商品一定不能在历史喜欢商品中出现
                    if item not in rec_dict[uid]:
                        rec_dict[uid][item] = 0
                    rec_dict[uid][item] += score
    
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
    pickle.dump(rec_dict, open(os.path.join(save_path,'ItemCF_rec_dict.txt'), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--TopN', type=int, default=0, help='number of top score items selected')
    parser.add_argument('--TopK', type=int, default=50, help='number of similar items/users')
    parser.add_argument('--rmse', action='store_true', help='number of similar items/users')
    args = parser.parse_args()

    train_users, valid_users = load_data(file_path='./data')

    if not args.TopN:
        Top10_rec_dict, Top20_rec_dict, Top50_rec_dict, rec_dict = Cosine_Item_CF(train_users, valid_users, args.TopK ,args.TopN)
        print('Top 10:')
        rec_eval(Top10_rec_dict,valid_users,train_users)
        print('Top 20:')
        rec_eval(Top20_rec_dict,valid_users,train_users)
        print('Top 50:')
        rec_eval(Top50_rec_dict,valid_users,train_users)
        save_rec_dict('./data',rec_dict)
        print('Done.')
    else:
        rec_dict = Cosine_Item_CF(train_users, valid_users, args.TopK ,args.TopN)
        print(f'Top {args.TopN}:')
        rec_eval(rec_dict,valid_users,train_users)
        print('Done.')
    