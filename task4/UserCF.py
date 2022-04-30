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

class UserCF():
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

    def Pearsonr_UserCF(self, TopK ,TopN):
        '''
        预测评分，数据集含rating
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
                user,item,rating = row['userID'], row['itemID'], row['Rating']
                if item not in ratings_item:
                    ratings_item[item] = dict()
                ratings_item[item][user] = rating
                if user not in ratings_user:
                    ratings_user[user] = dict()
                ratings_user[user][item] = rating

            print('用户-物品矩阵创建完毕!!!')
            with open('ratings_user.txt', 'wb') as f1:
                pickle.dump(ratings_user, f1)
            with open('ratings_item.txt', 'wb') as f2:
                pickle.dump(ratings_item, f2)


        if os.path.exists('user_similarity_matrix.txt'):
            print('读取用户相似度矩阵...')
            with open('user_similarity_matrix.txt','rb') as f3:
                similarity_matrix = pickle.load(f3)
        else:
            print('开始创建用户相似度矩阵...')
            # 相似度矩阵用二维数组储存，如果用字典保存，测试集评估会出现 key_error，比较麻烦
            similarity_matrix = -1 * np.ones(shape=(self.n_users,self.n_users)) 
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

        # 处理pearsonr相关系数为nan的值
        df = pd.DataFrame(similarity_matrix)
        df = df.fillna(0)
        similarity_matrix = df.to_numpy()
        del df
        # 计算每个用户的平均评分，用于消除用户评分偏置
        avg_user_ratings = np.zeros(self.n_users)
        for user,rate_list in ratings_user.items():
            avg_rating = np.mean([rate for rate in rate_list.values()])
            avg_user_ratings[user] = avg_rating

        # 生成TopN推荐列表
        # 找到与目标用户最相似的K个用户
        # 利用这些用户对物品的评分预测目标用户对物品的评分
        val_users_set = set(self.valid_users.keys())
        rec_dict = dict()
        factor = dict()
        print('开始预测...')
        for user in tqdm(val_users_set,total = len(val_users_set)):
            user_history = ratings_user[user].keys()
            similar_users = np.argsort(-similarity_matrix[user])[:TopK]
            similarity_of_users = similarity_matrix[user][similar_users]
            rec_dict[user] = {} # 该用户的推荐物品得分字典
            factor[user] = {} # 分母
            for uuser,score in zip(similar_users, similarity_of_users):
                for item in ratings_user[uuser]:
                    if item not in user_history:
                        if item not in rec_dict[user]:
                            rec_dict[user][item] = 0
                        if item not in factor[user]:
                            factor[user][item] = 0
                        rec_dict[user][item] += score * (ratings_user[uuser][item] - avg_user_ratings[uuser]) # 含偏置
                        #rec_dict[user][item] += score * ratings_user[uuser][item]  # 不含偏置
                        factor[user][item] += score
            for item_idx,rank_score in rec_dict[user].items():
                rank_score += avg_user_ratings[user] # 含偏置
                rank_score /= factor[user][item_idx]
        print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
        items_rank = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:TopN] for k, v in rec_dict.items()}
        items_rank = {k: set([x[0] for x in v]) for k, v in items_rank.items()}
        print('预测完毕!')
        return rec_dict, items_rank, ratings_user #计算RMSE用

    def Cosine_UserCF(self, TopK ,TopN):
        """
        仅预测是否会评分，不包含具体rating
        """
        # 建立item->users倒排表
        # 倒排表的格式为: {item_id1: {user_id1, user_id2, ... , user_idn}, item_id2: ...} 也就是每个item对应有那些用户有过点击
        # 建立倒排表的目的就是为了更方便的统计用户之间共同交互的商品数量
        print('建立倒排表...')
        item_users = {}
        for uid, items in tqdm(self.train_users.items()): # 遍历每一个用户的数据,其中包含了该用户所有交互的item
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
        pre_dict = {}
        for u, _ in tqdm(self.valid_users.items()): # 遍历测试集用户，给测试集中的每个用户进行推荐
            pre_dict[u] = {} # 初始化用户u的候选item的字典
            for v, score in sorted(sim[u].items(), key=lambda x: x[1], reverse=True)[:TopK]: # 选择与用户u最相似的k个用户
                for item in self.train_users[v]: # 遍历相似用户之间交互过的商品
                    if item not in self.train_users[u]: # 如果相似用户交互过的商品，测试用户在训练集中出现过，就不用进行推荐，直接跳过
                        if item not in pre_dict[u]:
                            pre_dict[u][item] = 0   # 初始化用户u对item的相似度分数为０
                        pre_dict[u][item] += score  # 累加所有相似用户对同一个item的分数
        
        print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
        rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:TopN] for k, v in pre_dict.items()}
        self.rec_dict = {k: set([x[0] for x in v]) for k, v in rec_dict.items()} # 将输出整合成合适的格式输出
            

    def predict(self, TopK, TopN):
        if self.mode == 'cosine':
            self.Cosine_UserCF(TopK, TopN)
        elif self.mode =='pearsonr':
            self.Pearsonr_UserCF(TopK, TopN)

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
            _rmse = RMSE(rel_list,pre_list)
            print(f'均方根误差RMSE:{round(_rmse,5)}')

        print('recall:',Recall(self.rec_dict, self.valid_users))
        print('precision',Precision(self.rec_dict, self.valid_users))
        print('coverage',Coverage(self.rec_dict, self.train_users))
        print('Popularity',Popularity(self.rec_dict, self.train_users))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--TopN', type=int, default=10, help='number of top score items selected')
    parser.add_argument('--TopK', type=int, default=10, help='number of similar items/users')
    parser.add_argument('--rmse', action='store_true', help='rmse')
    parser.add_argument('--mode', type=str, default='cosine', help='choose mode:cosine,pearsonr')
    args = parser.parse_args()

    model = UserCF('../dataset/ml-1m', args.mode)
    model.predict(args.TopK, args.TopN)
    model.eval(rmse=args.rmse)


    