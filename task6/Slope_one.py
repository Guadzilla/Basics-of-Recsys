import pandas as pd
import numpy as np
import os 
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from evaluate import RMSE, rec_eval
import random

seed = 42
np.random.seed(seed)
random.seed(seed)



class SlopeOne():
    def __init__(self, file_path):
        self.get_data(file_path)
        self.Ratings_diffs, self.Count = self.get_diffs_matrix()

    def get_data(self,root_path):
        rnames = ['userID','itemID','Rating','timestamp']
        #data = pd.read_csv(os.path.join(root_path, 'ratings.dat'), sep='::', engine='python', names=rnames)
        data = pd.read_csv(os.path.join(root_path, 'sample.csv'))

        train_data, valid_data = train_test_split(data[['userID','itemID','Rating']], test_size=0.1)
        self.train_users = train_data.groupby('userID')['itemID'].apply(list).to_dict()
        self.valid_users = valid_data.groupby('userID')['itemID'].apply(list).to_dict()
        self.train_ratings = self.get_ratings_dict(train_data)
        self.valid_ratings = self.get_ratings_dict(valid_data)


    def get_ratings_dict(self,data):
        """
        data: pd.DataFrame(['userID','itemID','Rating'])
        """
        users_rating = {}
        for _,row in data.iterrows():
            uid,iid,rating = row
            if uid not in users_rating:
                users_rating[uid] = {}
            if iid not in users_rating[uid]:
                users_rating[uid][iid] = rating
        return users_rating


    def get_diffs_matrix(self):
        """
        构建物品评分偏差矩阵
        """
        Diff = {}
        Count = {}
        print('开始构建物品评分偏差矩阵...')
        for user,u_rating in tqdm(self.train_ratings.items()):
            for itemx in u_rating:
                if itemx not in Diff:
                    Diff[itemx] = {}
                    Count[itemx] = {}
                for itemy in u_rating:
                    if itemx == itemy:
                        continue
                    if itemy not in Diff[itemx]:
                        Diff[itemx][itemy] = 0
                        Count[itemx][itemy] = 0
                    Diff[itemx][itemy] += self.train_ratings[user][itemx] - self.train_ratings[user][itemy]     #(公式 1)
                    Count[itemx][itemy] += 1
        for x in Diff:
            for y in Diff[x]:
                if x == y:
                    continue
                Diff[x][y] /= Count[x][y]
        return Diff, Count


    def predict(self, TopN):
        """
        在验证集上预测
        valid_data: pd.DataFrame(['userID','itemID','Rating'])
        valid_ratings: {uid1:{iid1:rating,iid2:rating,...},uid2:{...},...}
        """
        all_train_items = []
        for ratings in self.train_ratings.values():
            all_train_items += list(ratings.keys())
        all_train_items = set(all_train_items)
        # 预测评分
        # 首先找出Alice交互过的物品哪些与要预测的物品有过”共同被统一用户评分“的经历，即存在倒排索引Ratings_item[x][y]
        pre_list = []   #1 pre_list: [rating1,rating2,...] 预测评分列表
        rel_list = []   #1 rel_list: [rating1,rating2,...] 实际评分列表
        pre_dict = {}   #1 rec_dict: {uid1:{iid1:score,iid2:score,...},uid2:{...},...}    所有预测电影评分
        rel_dict = {}   #1 rel_dict: {uid1:[iid1,iid2,...]},uid2:[...],...}    每个用户实际观看的电影
        rec_dict = {}   #2 pre_dict: {uid1:[iid1,iid3,...]},uid2:[...],...}    每个用户预测TopN
        weight_sum = {}
        print('开始预测...')
        for uid, u_ratings in tqdm(self.valid_ratings.items()):
            if uid not in pre_dict:
                pre_dict[uid] = {}
                rel_dict[uid] = []
            uid_hist = list(self.train_ratings[uid].keys())
            for iid in all_train_items:
                if iid in uid_hist:
                    continue
                if iid not in self.Ratings_diffs:
                    pre_dict[uid][iid] = 0  # 预测评分
                    if iid in u_ratings:
                        pre_list.append(0)  # 预测评分列表
                        rel_list.append(u_ratings[iid]) # 实际评分列表
                        rel_dict[uid] += [iid]  # 实际观看列表
                    continue
                if iid not in pre_dict[uid]:
                    pre_dict[uid][iid] = 0
                related_items = set()           # 用来推断当前物品得分的相关物品的集合,从交互历史和偏差矩阵中选
                for item in uid_hist:
                    if item in self.Ratings_diffs[iid]:
                        related_items.add(item)
                if len(related_items) == 0:
                    continue
                weighted_score = 0
                weight_sum = 0
                for other_iid in related_items:   # (公式 2)
                    weight_sum += self.Count[other_iid][iid]
                    weighted_score += (self.train_ratings[uid][other_iid] - self.Ratings_diffs[other_iid][iid]) * self.Count[other_iid][iid]            
                pre_dict[uid][iid] = weighted_score/weight_sum # 预测评分

                if iid in u_ratings:
                    pre_list.append(weighted_score/weight_sum)  # 预测评分列表
                    rel_list.append(u_ratings[iid]) # 实际评分列表
                    rel_dict[uid] += [iid]  # 实际观看列表
        
        rec_dict = {uid:sorted(list(ratings.items()),key=lambda x:x[1],reverse=True)[:TopN] for uid,ratings in pre_dict.items()}
        rec_dict = {uid:[x[0] for x in v] for uid,v in rec_dict.items()}
        self.pre_list, self.rel_list, self.rec_dict, self.rel_dict = pre_list, rel_list, rec_dict, rel_dict

    def evaluate(self):
        rec_eval(self.rec_dict,self.rel_dict,self.train_users)
        rmse = RMSE(self.rel_list,self.pre_list)
        print(f'均方根误差RMSE: {round(rmse,5)}')

if __name__ == "__main__":
    model = SlopeOne(file_path='../dataset/ml-1m')
    model.predict(TopN=10)
    model.evaluate()


