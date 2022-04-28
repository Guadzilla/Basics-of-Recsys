import pandas as pd
import numpy as np
import pickle
import math
import os 
import warnings

from torch import pairwise_distance
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from collections import defaultdict
from tqdm import tqdm
from evaluate import Recall,Precision,Coverage,Popularity,RMSE


def load_data(file_path):
    """
    加载数据
    """
    with open(os.path.join(file_path,'train_users.txt'), 'rb') as f1:
        train_users = pickle.load(f1)
    with open(os.path.join(file_path,'valid_users.txt'), 'rb') as f2:
        valid_users = pickle.load(f2) 

    return train_users, valid_users


def items_dict(data):
    """
    构建用户-物品/物品-用户双向索引
    data: array([[user1,item1],[user2,item2],...])
    """
    train_items = {}
    for user, item in data:
        if item not in train_items:
            train_items[item] = []
        train_items[item].append(user)
    return train_items



def Rating_diffs_matrix(train_items):
    """
    构建物品评分偏差矩阵
    train_items: {iid1:[uid1,uid2,...],iid2:[uid2,uid4,...] }
    """
    Ratings_diffs = {}
    N_set = {}
    print('开始构建物品评分偏差矩阵...')
    for itemx,itemx_history in tqdm(items_rating.items()):
        if itemx not in Ratings_diffs:
            Ratings_diffs[itemx] = defaultdict(float)
            N_set[itemx] = defaultdict(int)
        for itemy,itemy_history in items_rating.items():
            if itemx != itemy:
                for x_user in itemx_history:
                    if x_user in itemy_history:
                        Ratings_diffs[itemx][itemy] += items_rating[itemy][x_user] - items_rating[itemx][x_user]
                        N_set[itemx][itemy] += 1
    for itemx,ys in Ratings_diffs.items():
        for itemy,rating in ys.items():
            Ratings_diffs[itemx][itemy] /= N_set[itemx][itemy]
    return Ratings_diffs, N_set



def predict(Ratings_diffs, N_set, valid_data, users_rating):
    """
    在验证集上预测
    valid_data: array([[user1,item1],[user2,item2],...],[rating1,rating2,...])
    """
    # 预测评分
    # 首先找出Alice交互过的物品哪些与要预测的物品有过”共同被统一用户评分“的经历，即存在倒排索引Ratings_item[x][y]
    pre_list = []
    rel_list = []
    print('开始预测...')
    for idx,pairs in tqdm(enumerate(valid_data[0]),total=len(valid_data[1])):
        user,item = pairs
        rating = valid_data[1][idx]
        if user not in users_rating or item not in Ratings_diffs:
            continue
        user_history = users_rating[user]
        candidate_items = set()
        for iitem in Ratings_diffs[item]:
            if iitem in user_history:
                candidate_items.add(item)
        weighted_score = 0
        weight_sum = 0
        for iiitem in Ratings_diffs[item]:
            weight_sum += N_set[item][iiitem]
            weighted_score += Ratings_diffs[item][iiitem] * N_set[item][iiitem]
        pre_list.append(weighted_score/weight_sum)
        rel_list.append(rating)
    return pre_list, rel_list




if __name__ == "__main__":
    train_data, valid_data = load_data(file_path='./data')
    train_items  = items_dict(train_data)
    Ratings_diffs, N_set = Rating_diffs_matrix(items_rating)
    pre_list, rel_list = predict(Ratings_diffs, N_set, valid_data, users_rating)
    rmse = RMSE(rel_list,pre_list)
    print(f'均方根误差RMSE: {round(rmse,5)}')

