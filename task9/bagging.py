from collections import defaultdict
import os 
import pickle
import argparse
from scipy.special import softmax
from evaluate import *
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('model', type=int, nargs='+',help='models selected')
args = parser.parse_args()

def load_rec_dict(file_path, model_name):
    with open(os.path.join(file_path,f'{model_name}_rec_dict.txt'),'rb') as f:
        Rec_dict = pickle.load(f)
    # Rec_dict:{uid1:[(iid1,score),(iid2,score),...],uid2:[(...),(...),...],...}
    new_rec_dict = {}
    for uid, iid_score_list in Rec_dict.items():
        new_rec_dict[uid] = dict()
        score_list = []
        iid_list = []
        for iid, score in iid_score_list.items():
            score_list.append(score)
            iid_list.append(iid)
        prob = softmax(score_list)      # 把物品评分转化成概率
        for iid, p in zip(iid_list, prob):
            new_rec_dict[uid][iid] = p
    return new_rec_dict

def load_data(file_path):
    """
    加载数据
    """
    with open(os.path.join(file_path,'train_users.txt'), 'rb') as f1:
        train_users = pickle.load(f1)
    with open(os.path.join(file_path,'valid_users.txt'), 'rb') as f2:
        valid_users = pickle.load(f2) 

    return train_users, valid_users


def mixup(*rec_dicts):
    new_rec_dict = {}
    num = {}    # 记录目标用户被推荐同一商品几次，后续做除数
    for rec_dict in tqdm(rec_dicts):
        for uid in rec_dict:
            if uid not in new_rec_dict:
                new_rec_dict[uid] = {}
                num[uid] = {}
            for iid, prob in rec_dict[uid].items():
                if iid not in new_rec_dict[uid]:
                    new_rec_dict[uid][iid] = 0
                    num[uid][iid] = 0
                new_rec_dict[uid][iid] += prob
                num[uid][iid] += 1
    for uid,iid_score in new_rec_dict.items():
        for iid in iid_score:
            new_rec_dict[uid][iid] /= num[uid][iid]
    return new_rec_dict

def bag_eval(val_rec_items, val_user_items, trn_user_items):
    recall = Recall(val_rec_items, val_user_items)
    precision = Precision(val_rec_items, val_user_items)
    coverage = Coverage(val_rec_items, trn_user_items)
    popularity = Popularity(val_rec_items, trn_user_items)
    print(f'{recall}\t{precision}\t{coverage}\t{popularity}',end='\t')

# 数据所在路径
file_path = './data'

# load 各模型推荐得分
n_items = 3883

# 加载数据
print('开始加载数据...')
model_list = ['ItemCF','UserCF','MF','Word2Vec']
models = []
for idx in args.model:
    models.append(model_list[idx-1])
models_rec_dict = []
for model in models:
    models_rec_dict.append(load_rec_dict(file_path,model))

# load 训练集、验证集数据
train_users, valid_users = load_data(file_path)

# 模型融合
print('开始模型融合...')
new_rec_dict = mixup(*models_rec_dict)

# TopN推荐
Top50_rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:50] for k, v in new_rec_dict.items()}
Top50_rec_dict = {k: list([x[0] for x in v]) for k, v in Top50_rec_dict.items()}
Top10_rec_dict = {k: v[:10] for k, v in Top50_rec_dict.items()}
Top20_rec_dict = {k: v[:20] for k, v in Top50_rec_dict.items()}
print('Top 10,20,50:')
bag_eval(Top10_rec_dict,valid_users,train_users)
bag_eval(Top20_rec_dict,valid_users,train_users)
bag_eval(Top50_rec_dict,valid_users,train_users)
print('\nDone.')