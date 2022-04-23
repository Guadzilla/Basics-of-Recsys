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
from evaluate import Recall,Precision,Coverage,Popularity


def load_data():
    """
    加载数据，分割训练集、验证集
    """
    data = pd.read_table('../ml-1m/ratings.dat', sep='::', names = ['userID','itemID','Rating','Zip-code'])
    users = pd.read_table('../ml-1m/users.dat',sep='::',names=['UserID','Gender','Age','Occupation','Zip-code'])
    movies = pd.read_table('../ml-1m/movies.dat',sep='::',names=['MovieID','Title','Genres'],encoding='ISO-8859-1')
    user2index = defaultdict(int)
    item2index = defaultdict(int)
    count_user = 0
    count_item = 0

    for user in set(users['UserID']):
        user2index[user] = count_user
        count_user += 1

    for item in set(movies['MovieID']):
        item2index[item] = count_item
        count_item += 1

    data['userID'] = data['userID'].map(user2index)
    data['itemID'] = data['itemID'].map(item2index)

    tra_data, val_data = train_test_split(data,test_size=0.2)
    
    tra_users = tra_data.groupby('userID')['itemID'].apply(list).to_dict()
    val_users = val_data.groupby('userID')['itemID'].apply(list).to_dict()

    all_data = (tra_data, val_data, tra_users, val_users, count_user+1, count_item+1)
    return all_data


def pearsonrSim(x,y):
    """
    返回皮尔逊相关系数
    """
    if len(x)==0:
        return 0
    elif len(x)==1:
        return 1
    else:
        return pearsonr(x,y)[0]

def cosineSim(x,y):
    """
    返回余弦相似度
    """
    if len(x)<1:
        return 0
    return cosine_similarity([x,y])

def eval(rec_dict,val_users,tra_users):
    """
    rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
    val_users: 用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    tra_users: 训练集实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    """
    print('recall:',Recall(rec_dict, val_users))
    print('precision',Precision(rec_dict, val_users))
    print('coverage',Coverage(rec_dict, tra_users))
    print('Popularity',Popularity(rec_dict, tra_users))



def Pearsonr_ItemCF(n_user, n_item, tra_data, val_users, K ,TopN):
    '''
    K: K表示的是相似用户的数量，每个用户都选择与其最相似的K个用户
    N: N表示的是给用户推荐的商品数量，给每个用户推荐相似度最大的N个商品
    '''


    if os.path.exists('ratings_item.txt') and os.path.exists('ratings_user.txt'):
        print('读取用户-物品矩阵...')
        with open('ratings_user.txt', 'rb') as f1:
            ratings_user = pickle.load(f1)
        with open('ratings_item.txt', 'rb') as f2:
            ratings_item = pickle.load(f2)
    else:  
        ratings_user = dict()
        ratings_item = dict()
        print('开始创建用户-物品矩阵...')
        for _, row in tqdm(tra_data.iterrows(),total=len(tra_data)):
            user,item,rating = row['userID'],row['itemID'],row['Rating']
            if item not in ratings_item:
                ratings_item[item] = dict()
            ratings_item[item][user] = rating
            if user not in ratings_user:
                ratings_user[user] = dict()
            ratings_user[user][item] = rating

        print('用户-物品矩阵创建完毕!!!')
        with open('ratings_user.txt', 'wb') as f1:
            pickle.dump(ratings_user,f1)
        with open('ratings_item.txt', 'wb') as f2:
            pickle.dump(ratings_item,f2)

    if os.path.exists('item_similarity_matrix.txt'):
        print('读取物品相似度矩阵...')
        with open('item_similarity_matrix.txt','rb') as f3:
            similarity_matrix = pickle.load(f3)
    else:
        print('开始创建物品相似度矩阵...')
        # 相似度矩阵用二维数组储存，如果用字典保存，测试集评估会出现 key_error，比较麻烦
        similarity_matrix = -1 * np.ones(shape=(n_item,n_item))
        for itemx in tqdm(ratings_item, total=len(ratings_item)):
            for itemy in ratings_item:
                if itemy == itemx:
                    continue
                itemxVec = []
                itemyVec = []
                itemx_history = set(ratings_item[itemx].keys())
                itemy_history = set(ratings_item[itemy].keys())
                intersection = itemx_history.intersection(itemy_history)
                for item in intersection:
                    itemxVec.append(ratings_item[itemx][item])
                    itemyVec.append(ratings_item[itemy][item])
                similarity_matrix[itemx][itemy] = similarity_matrix[itemy][itemx] = pearsonrSim(itemxVec,itemyVec)
        print('相似度矩阵构建完毕')
        with open('item_similarity_matrix.txt','wb') as f3:
            similarity_matrix = pickle.dump(similarity_matrix,f3)

    # 处理pearsonr相关系数为nan的值
    df = pd.DataFrame(similarity_matrix)
    df = df.fillna(0)
    similarity_matrix = df.to_numpy()
    del df

    # 计算每个物品的平均评分，用于消除用户评分偏置
    avg_item_ratings = np.zeros(n_item)
    for item,rate_list in ratings_item.items():
        avg_rating = np.mean([rate for rate in rate_list.values()])
        avg_item_ratings[item] = avg_rating

    # 生成TopN推荐列表
    # 要预测的物品就是用户没有评分过的物品
    # 先筛选出用户交互过的商品
    # 再选出与这些物品最相似的K个物品，并且过滤掉用户已经交互过的物品
    # 再根据用户交互过的商品的得分，计算目标物品的得分
    # 对这些items得分降序排列
    val_users_set = set(val_users.keys())
    rec_dict = dict()
    factor = dict()
    print('给用户进行推荐...')
    for user in tqdm(val_users_set, total=len(val_users_set)):
        rec_dict[user] = dict() # 候选集得分
        factor[user] = dict() # 分母
        user_history = ratings_user[user].keys()
        for item in user_history:   # 选出与用户交互过的物品最相似的K个物品
            similar_items_idx = np.argsort(-similarity_matrix[item])[:K]
            similar_items = similarity_matrix[item][similar_items_idx]
            for iitem, score in zip(similar_items_idx, similar_items):
                if iitem not in user_history:   # 过滤掉用户已经交互过的物品
                    if iitem not in rec_dict[user]:
                        rec_dict[user][iitem] = 0
                    if iitem not in factor[user]:
                        factor[user][iitem] = 0
                    rec_dict[user][iitem] += score * (ratings_user[user][item] - avg_item_ratings[item])
                    factor[user][iitem] += score
        for item_idx,rank_score in rec_dict[user].items():
            rank_score /= factor[user][item_idx]
            rank_score += avg_item_ratings[item_idx]
    print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
    items_rank = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:TopN] for k, v in rec_dict.items()}
    items_rank = {k: set([x[0] for x in v]) for k, v in items_rank.items()}
    return rec_dict

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
    items_rank = {}
    print('给用户进行推荐．．．')
    for uid, _ in tqdm(val_user_items.items()):
        items_rank[uid] = {} # 存储用户候选的推荐商品
        for hist_item in trn_user_items[uid]: # 遍历该用户历史喜欢的商品，用来下面寻找其相似的商品
            for item, score in sorted(sim[hist_item].items(), key=lambda x: x[1], reverse=True)[:K]:
                if item not in trn_user_items[uid]: # 进行推荐的商品一定不能在历史喜欢商品中出现
                    if item not in items_rank[uid]:
                        items_rank[uid][item] = 0
                    items_rank[uid][item] += score
    
    print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
    items_rank = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:N] for k, v in items_rank.items()}
    items_rank = {k: set([x[0] for x in v]) for k, v in items_rank.items()}
    return items_rank


if __name__ == "__main__":
    all_data = load_data()
    tra_data, val_data, tra_users, val_users, n_user, n_item = all_data
    rec_dict = Pearsonr_ItemCF(n_user, n_item, tra_data, val_users, 10 ,10)
    #rec_dict = Cosine_Item_CF(tra_users, val_users, 10 ,10)
    eval(rec_dict,val_users,tra_users)


"""
Pearsonr_ItemCF()
recall: 0.18
precision 0.61
coverage 35.17
Popularity 5.539
"""

"""
Cosine_Item_CF()
recall: 9.2
precision 30.48
coverage 19.18
Popularity 7.171
"""

