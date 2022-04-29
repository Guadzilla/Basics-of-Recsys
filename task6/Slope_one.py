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


# 加载数据
def get_data(root_path):
    # 读取数据时，定义的列名
    rnames = ['userID','itemID','Rating','timestamp']
    data = pd.read_csv(os.path.join(root_path, 'ratings.dat'), sep='::', engine='python', names=rnames)
    #data = pd.read_csv(os.path.join(root_path, 'sample.csv'))

    lbe = LabelEncoder()
    data['userID'] = lbe.fit_transform(data['userID'])
    n_users = len(lbe.classes_)
    data['itemID'] = lbe.fit_transform(data['itemID']) 
    n_items = len(lbe.classes_)


    X = np.array(data[['userID','itemID']])
    Y = np.array(data['Rating'])
    tra_X, val_X, tra_Y, val_Y = train_test_split(X, Y, test_size=0.2)
    train_data = (tra_X, tra_Y)       # array([[user1,item1],[user2,item2],...],[rating1,rating2,...])
    valid_data = (val_X, val_Y)       # array([[user1,item1],[user2,item2],...],[rating1,rating2,...])

    return n_users, n_items, train_data, valid_data


def user_item_dict(data):
    """
    构建用户-物品/物品-用户双向索引
    data: array([[user1,item1],[user2,item2],...],[rating1,rating2,...])
    """
    users_rating = {}
    items_rating = {}
    for idx,pairs in enumerate(data[0]):
        user, item = pairs
        rating = data[1][idx]
        if user not in users_rating:
            users_rating[user] = {}
        if item not in items_rating:
            items_rating[item] = {}
        users_rating[user][item] = rating
        items_rating[item][user] = rating
    return users_rating,items_rating



def Rating_diffs_matrix(items_rating):
    """
    构建物品评分偏差矩阵
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
    _, _, train_data, valid_data = get_data('../dataset/ml-1m')
    users_rating, items_rating = user_item_dict(train_data)
    Ratings_diffs, N_set = Rating_diffs_matrix(items_rating)
    pre_list, rel_list = predict(Ratings_diffs, N_set, valid_data, users_rating)
    rmse = RMSE(rel_list,pre_list)
    print(f'均方根误差RMSE: {round(rmse,5)}')

