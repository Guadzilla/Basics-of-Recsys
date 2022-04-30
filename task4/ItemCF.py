import argparse
import pandas as pd
import numpy as np
import pickle
import math
import os 
import warnings

from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm   
from evaluate import Recall,Precision,Coverage,Popularity,RMSE

class ItemCF():
    def __init__(self, file_path, mode):
        assert mode in ['cosine','pearsonr'], "invalid mode"
        self.mode = mode
        self.load_data(file_path)


    def load_data(self, file_path):
        """
        加载数据，分割训练集、验证集
        """
        #data = pd.read_table(os.path.join(file_path,'ratings.dat'), sep='::', names = ['userID','itemID','Rating','Zip-code'])
        data = pd.read_csv(os.path.join(file_path, 'sample.csv'))
        uid_lbe = LabelEncoder()
        data['userID'] = uid_lbe.fit_transform(data['userID'])
        self.n_users = max(uid_lbe.classes_)
        iid_lbe = LabelEncoder()
        data['itemID'] = iid_lbe.fit_transform(data['itemID'])
        self.n_items = max(iid_lbe.classes_)

        self.train_data, self.valid_data = train_test_split(data,test_size=0.1)
        
        self.train_users = self.train_data.groupby('userID')['itemID'].apply(list).to_dict()
        self.valid_users = self.valid_data.groupby('userID')['itemID'].apply(list).to_dict()

    def pearsonrSim(self,x,y):
        """
        返回皮尔逊相关系数
        """
        if len(x)==0:
            return 0
        elif len(x)<3:
            return 1
        else:
            return pearsonr(x,y)[0]

    def Pearsonr_ItemCF(self, K ,N):
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
            for _, row in tqdm(self.train_data.iterrows(),total=len(self.train_data)):
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
            similarity_matrix = -1 * np.ones(shape=(self.n_items,self.n_items))
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
                    similarity_matrix[itemx][itemy] = similarity_matrix[itemy][itemx] = self.pearsonrSim(itemxVec,itemyVec)
            print('相似度矩阵构建完毕')
            with open('item_similarity_matrix.txt','wb') as f3:
                similarity_matrix = pickle.dump(similarity_matrix,f3)

        # 处理pearsonr相关系数为nan的值
        df = pd.DataFrame(similarity_matrix)
        df = df.fillna(0)
        similarity_matrix = df.to_numpy()
        del df

        # 计算每个物品的平均评分，用于消除用户评分偏置
        avg_item_ratings = np.zeros(self.n_items)
        for item,rate_list in ratings_item.items():
            avg_rating = np.mean([rate for rate in rate_list.values()])
            avg_item_ratings[item] = avg_rating

        # 生成TopN推荐列表
        # 要预测的物品就是用户没有评分过的物品
        # 先筛选出用户交互过的商品
        # 再选出与这些物品最相似的K个物品，并且过滤掉用户已经交互过的物品
        # 再根据用户交互过的商品的得分，计算目标物品的得分
        # 对这些items得分降序排列
        val_users_set = set(self.valid_users.keys())
        rec_dict = dict()
        factor = dict()
        print('给用户进行推荐...')
        for user in tqdm(val_users_set, total=len(val_users_set)):
            rec_dict[user] = dict() # 该用户的推荐物品得分字典
            factor[user] = dict() # 分母
            user_history = ratings_user[user].keys()
            for item in user_history:   # 选出与用户交互过的物品最相似的K个物品
                similar_items_idx = np.argsort(-similarity_matrix[item])[:K]
                similarity_of_items = similarity_matrix[item][similar_items_idx]
                for iitem, score in zip(similar_items_idx, similarity_of_items):
                    if iitem not in user_history:   # 过滤掉用户已经交互过的物品
                        if iitem not in rec_dict[user]:
                            rec_dict[user][iitem] = 0
                        if iitem not in factor[user]:
                            factor[user][iitem] = 0
                        #rec_dict[user][iitem] += score * (ratings_user[user][item] - avg_item_ratings[item])   # 含偏置
                        rec_dict[user][iitem] += score * ratings_user[user][item] # 不含偏置
                        factor[user][iitem] += score
            for item_idx,rank_score in rec_dict[user].items():
                #rank_score += avg_item_ratings[item_idx]   # 含偏置
                rank_score /= factor[user][item_idx]
                rec_dict[user][item_idx] = rank_score
        print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
        self.TopN_rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:N] for k, v in rec_dict.items()}
        self.TopN_rec_dict = {k: set([x[0] for x in v]) for k, v in self.TopN_rec_dict.items()}
        self.rec_dict = rec_dict


    def Cosine_Item_CF(self, K, N):
        '''
        train_users: 表示训练数据，格式为：{user_id1: [item_id1, item_id2,...,item_idn], user_id2...}
        valid_users: 表示验证数据，格式为：{user_id1: [item_id1, item_id2,...,item_idn], user_id2...}
        K: Ｋ表示的是相似商品的数量，为每个用户交互的每个商品都选择其最相思的K个商品
        N: N表示的是给用户推荐的商品数量，给每个用户推荐相似度最大的N个商品
        '''

        # 建立user->item的倒排表
        # 倒排表的格式为: {user_id1: [item_id1, item_id2,...,item_idn], user_id2: ...} 也就是每个用户交互过的所有商品集合
        # 由于输入的训练数据train_users,本身就是这中格式的，所以这里不需要进行额外的计算
        

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
        for uid, items in tqdm(self.train_users.items()):
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
        # 在对用户进行推荐之前需要先通过商品相似度矩阵得到当前用户交互过的商品最相似的前K个商品，
        # 然后选出其中目标用户没有交互过的商品，计算最终的相似度分数
        # 最终推荐的候选商品的相似度分数是由多个相似商品对该商品分数的一个累加和
        pre_dict = {}
        print('给用户进行推荐．．．')
        for uid, _ in tqdm(self.valid_users.items()):
            pre_dict[uid] = {} # 存储用户候选的推荐商品
            for hist_item in self.train_users[uid]: # 遍历该用户历史喜欢的商品，用来下面寻找其相似的商品
                for item, score in sorted(sim[hist_item].items(), key=lambda x: x[1], reverse=True)[:K]:
                    if item not in self.train_users[uid]: # 进行推荐的商品一定不能在历史喜欢商品中出现
                        if item not in pre_dict[uid]:
                            pre_dict[uid][item] = 0
                        pre_dict[uid][item] += score
        
        print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
        TopN_rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:N] for k, v in pre_dict.items()}
        self.TopN_rec_dict = {k: set([x[0] for x in v]) for k, v in TopN_rec_dict.items()}


    def predict(self, TopK, TopN):
        if self.mode == 'cosine':
            self.Cosine_Item_CF(TopK, TopN)
        elif self.mode =='pearsonr':
            self.Pearsonr_ItemCF(TopK, TopN)

    def eval(self,rmse=True):
        """
        rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
        val_users: 用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
        tra_users: 训练集实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
        """
        if rmse:
            pre_list = []
            rel_list = []
            for idx,row in self.valid_data.iterrows():
                userID,itemID,Rating,_ = row
                rel_list.append(Rating)
                if userID in self.rec_dict:
                    if itemID in self.rec_dict[userID]:
                        pre_list.append(self.rec_dict[userID][itemID])
                        continue
                pre_list.append(0)
            _rmse = RMSE(rel_list, pre_list)
            print(f'均方根误差RMSE:{round(_rmse,5)}')

        print('recall:',Recall(self.TopN_rec_dict, self.valid_users))
        print('precision',Precision(self.TopN_rec_dict, self.valid_users))
        print('coverage',Coverage(self.TopN_rec_dict, self.train_users))
        print('Popularity',Popularity(self.TopN_rec_dict, self.train_users))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--TopN', type=int, default=10, help='number of top score items selected')
    parser.add_argument('--TopK', type=int, default=10, help='number of similar items/users')
    parser.add_argument('--rmse', action='store_false', help='rmse')
    parser.add_argument('--mode', type=str, default='cosine', help='choose mode:cosine,pearsonr')
    args = parser.parse_args()

    model = ItemCF('../dataset/ml-1m', args.mode)
    model.predict(args.TopK, args.TopN)
    model.eval(rmse=args.rmse)


    