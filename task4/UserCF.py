import pandas as pd
import numpy as np
import pickle
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
    if len(x)<2:
        return -1
    return pearsonr(x,y)[0]

def cosineSim(x,y):
    """
    返回余弦相似度
    """
    if len(x)<1:
        return 0
    return cosine_similarity([x,y])[0][1]


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



def UserCF(n_user, n_item, tra_data, val_users, K ,TopN):
    '''
    K: K表示的是相似用户的数量，每个用户都选择与其最相似的K个用户
    N: N表示的是给用户推荐的商品数量，给每个用户推荐相似度最大的N个商品
    '''


    if os.path.exists('ratings_user.txt'):
        print('读取用户-物品矩阵...')
        with open('ratings_user.txt', 'rb') as f1:
            ratings_user = pickle.load(f1)
        print('读取完毕!')
    else:
        ratings_user = dict()    # 双层字典的形式保存用户评分
        print('开始创建用户-物品矩阵...')
        for _, row in tqdm(tra_data.iterrows(),total=len(tra_data)):
            user,item,rating = row['userID'],row['itemID'],row['Rating']
            if user not in ratings_user:
                ratings_user[user] = dict()
            ratings_user[user][item] = rating
        print('用户-物品矩阵创建完毕!!!')
        with open('ratings_user.txt', 'wb') as f1:
            pickle.dump(ratings_user,f1)


    if os.path.exists('user_similarity_matrix.txt'):
        print('读取用户相似度矩阵...')
        with open('user_similarity_matrix.txt','rb') as f3:
            similarity_matrix = pickle.load(f3)
        print('读取完毕!')
    else:
        print('开始创建用户相似度矩阵...')
        # 相似度矩阵用二维数组储存，如果用字典保存，测试集评估会出现 key_error，比较麻烦
        similarity_matrix = -1 * np.ones(shape=(n_user,n_user)) 
        for userx in tqdm(ratings_user, total=len(ratings_user)):
            for usery in ratings_user:
                if usery == userx:
                    continue
                userxVec = []
                useryVec = []
                userx_history = set(ratings_user[userx].keys())
                usery_history = set(ratings_user[usery].keys())
                intersection = userx_history.intersection(usery_history)
                for item in intersection:
                    userxVec.append(ratings_user[userx][item])
                    useryVec.append(ratings_user[usery][item])
                similarity_matrix[userx][usery] = similarity_matrix[usery][userx] = pearsonrSim(userxVec,useryVec)
        print('相似度矩阵构建完毕')
        with open('user_similarity_matrix.txt','wb') as f3:
            similarity_matrix = pickle.dump(similarity_matrix,f3)


    # 计算每个用户的平均评分，用于消除用户评分偏置
    avg_user_ratings = np.zeros(n_user)
    for user,rate_list in ratings_user.items():
        avg_rating = np.mean([rate for rate in rate_list.values()])
        avg_user_ratings[user] = avg_rating

    # 生成TopN推荐列表
    # 先筛选出与目标用户最相似的K个用户
    # 再筛选出K个用户交互过的、且目标用户看过的items，预测这些items的得分
    # 对这些items得分降序排列
    val_users_set = set(val_users.keys())
    rec_dict = dict()
    print('开始预测...')
    for user in tqdm(val_users_set,total = len(val_users_set)):
        user_history = ratings_user[user].keys()
        similar_users = np.argsort(-similarity_matrix[user])[:K]
        candidate_items = set()             # 推荐候选集
        miss_items = list(range(n_item))    # 用户没看过的电影
        for i in user_history:
            miss_items.remove(i)
        for uuser in similar_users:
            for item in ratings_user[uuser]:
                if item in miss_items:
                    candidate_items.add(item)  
        record_list = dict() # 记录预测得分
        weighted_scores = 0 # 分母
        corr_value_sum = 0 # 分子
        for item in candidate_items:
            for uuser in similar_users:
                weighted_scores += similarity_matrix[user][uuser]
                if item not in ratings_user[uuser]:
                    ratings_user[uuser][item] = 0
                corr_value_sum += similarity_matrix[user][uuser] * (ratings_user[uuser][item] - avg_user_ratings[uuser])
            item_score = avg_user_ratings[user] + corr_value_sum/weighted_scores
            record_list[item] = item_score
        user_rec_list = sorted(record_list.items(),key=lambda x:x[1],reverse=True)[:TopN]
        rec_dict[user] = [x[0] for x in user_rec_list]
    print('预测完毕!')
    return rec_dict

    
if __name__ == "__main__":
    all_data = load_data()
    tra_data, val_data, tra_users, val_users, n_user, n_item = all_data
    rec_dict = UserCF(n_user, n_item, tra_data, val_users, 10 ,10)
    eval(rec_dict,val_users,tra_users)






