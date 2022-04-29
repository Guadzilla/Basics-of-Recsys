# 01导入包以及设置随机种子
import os

from psutil import NIC_DUPLEX_UNKNOWN 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import numpy as np
import torch
import torch.nn as nn
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorboardX import SummaryWriter
from tqdm import tqdm
from evaluate import *
from EarlyStopping import EarlyStopping

import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 02 定义超参数

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=8, help='the number of dataloader workers')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=100, help='the dimension of item embedding')
parser.add_argument('--epochs', type=int, default=50, help='the number of epochs to train for') # 50
parser.add_argument('--patience', type=int, default=5, help='EarlyStopping patience') 
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.001, help='learning rate decay rate')
parser.add_argument('--TopN', type=int, default=10, help='number of top score items selected')
parser.add_argument('--TopK', type=int, default=10, help='number of similar items/users')
parser.add_argument('--sample_ratio', type=int, default=1, help='nb of postive sample /nb of negtive sample')  
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args)

# 03 定义log
log_dir = f'./log/{datetime.datetime.now()}'
writer = SummaryWriter(log_dir)


# 04 定义模型
class MF(nn.Module):
    def __init__(self, n_users, n_items, K):
        super(MF,self).__init__()
        self.emb_dim = K
        self.user_b = nn.Parameter(torch.zeros(size=(n_users,1), requires_grad=True, device=device))  # 用户打分偏置
        self.item_b = nn.Parameter(torch.zeros(size=(n_items,1), requires_grad=True, device=device))  # 商品打分偏置
        #self.user_b = nn.Embedding(n_users, 1)
        #self.item_b = nn.Embedding(n_items, 1)
        self.u_emb = nn.Embedding(n_users, self.emb_dim)
        self.i_emb = nn.Embedding(n_items, self.emb_dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data_u, data_i):
        user_emb = self.u_emb(data_u).unsqueeze(1)
        item_emb = self.i_emb(data_i).unsqueeze(2)
        outputs = user_emb @ item_emb
        outputs = outputs.squeeze() + self.user_b[data_u].squeeze() + self.item_b[data_i].squeeze()
        return outputs.squeeze()


# 05 定义早停类(略)
early_stopping = EarlyStopping(patience=args.patience,verbose=True)

# 06 定义数据集 load_data,Dataset,DataLoader
def get_data(root_path):
    # 读取数据时，定义的列名
    data = pd.read_table(os.path.join(root_path,'ratings.dat'), sep='::', names = ['userID','itemID','Rating','Zip-code'])
    data = data[['userID','itemID','Rating']]
    uid_lbe = LabelEncoder()
    data['userID'] = uid_lbe.fit_transform(data['userID'])
    n_users = max(uid_lbe.classes_)
    iid_lbe = LabelEncoder()
    data['itemID'] = iid_lbe.fit_transform(data['itemID'])
    n_items = max(iid_lbe.classes_)
    train_data, valid_data = train_test_split(data,test_size=0.1)
    
    valid_users = valid_data.groupby('userID')['itemID'].apply(list).to_dict()
    train_users = train_data.groupby('userID')['itemID'].apply(list).to_dict()
    items_pool = np.array(train_data['itemID'])
    tra_X = np.array(train_data[['userID','itemID']])
    tra_Y = np.array(train_data['Rating'])
    val_X = np.array(valid_data[['userID','itemID']])
    val_Y = np.array(valid_data['Rating'])
    train_data = (tra_X, tra_Y)
    valid_data = (val_X, val_Y)

    return n_users, n_items, train_data, valid_data, items_pool, train_users, valid_users


print('Loading data...')
file_paths = '../dataset/ml-1m/'
n_users, n_items, train_data, valid_data, items_pool, train_users, valid_users = get_data(file_paths)
def RandomSelectNegativeSample(train_data, items_pool):
    """
    负采样
    train_data: [[[uid1,iid1],[uid1,iid2],...],[rating1,rating2,...]]
    """
    new_pairs = []
    new_ratings = []
    last_uid = None
    n_samples = 0
    u_hist = []

    for ui_pairs, rating in zip(train_data[0],train_data[1]):
        ui_pairs = list(ui_pairs)
        cur_uid = ui_pairs[0]
        if cur_uid==last_uid:
            new_pairs.append(ui_pairs)
            new_ratings.append(rating)
            u_hist.append(ui_pairs[1])
            last_uid = cur_uid
            n_samples += 1
        else:
            count = 0
            while count < args.sample_ratio * n_samples:
                item = items_pool[random.randint(0, len(items_pool) - 1)]
                if item in u_hist:
                    continue
                new_pairs.append([last_uid,item])
                new_ratings.append(0)
                count += 1
            new_pairs.append(ui_pairs)
            new_ratings.append(rating)
            last_uid = cur_uid
            n_samples = 1
            u_hist = [ui_pairs[1]]

    new_train_data = [new_pairs,new_ratings]
    return new_train_data
    
print('负采样...')
train_data = RandomSelectNegativeSample(train_data, items_pool)
class MovieDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[0]
        y = self.data[1]
        return x[index][0],x[index][1],y[index]

    def __len__(self):
        return len(self.data[1])



train_dataset = MovieDataset(train_data)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True ,num_workers=args.num_workers)
valid_dataset = MovieDataset(valid_data)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True ,num_workers=args.num_workers)


# 07 实例化模型，设置loss，优化器等

model = MF(n_users, n_items, args.embed_dim).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.lr_dc)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

# 08 开始训练
print('开始训练...')
for epoch in tqdm(range(args.epochs)):
    #=====================train============================
    model.train()
    sum_epoch_loss = 0
    for idx,(data_u,data_i,data_y) in enumerate(train_dataloader,0):
        data_u = data_u.to(device)
        data_i = data_i.to(device)
        data_y = data_y.to(device).to(torch.float32)
        outputs = model(data_u,data_i)
        optimizer.zero_grad()
        loss = criterion(data_y,outputs)
        sum_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        iter_num = idx + epoch * len(train_dataloader) + 1
        writer.add_scalar('loss/batch_loss', loss.item(), iter_num)

    sum_epoch_loss /= len(train_dataloader)
    writer.add_scalar('loss/train_loss', sum_epoch_loss, epoch)

    #=====================valid============================
    model.eval()
    valid_epoch_loss = 0
    for idx,(data_u,data_i,data_y) in enumerate(valid_dataloader,0):
        data_u = data_u.to(device)
        data_i = data_i.to(device)
        data_y = data_y.to(device).to(torch.float32)
        outputs = model(data_u,data_i)
        loss = criterion(outputs,data_y)
        valid_epoch_loss += loss.item()
    valid_epoch_loss /= len(valid_dataloader)
    writer.add_scalar('loss/valid_loss', valid_epoch_loss, epoch)

    #==================early stopping======================
    early_stopping(valid_epoch_loss,model=model,path='saved')
    if early_stopping.early_stop:
        print("Early stopping")
        break

# 09预测
import faiss
# 使用向量搜索库进行最近邻搜索

# uid = u_emb_idx ,且 u_emb_idx = faiss_idx , 所以 uid = faiss_idx 直接根据 uid 取向量
user_embedding = model.u_emb.weight.detach().cpu().numpy()
item_embedding = model.i_emb.weight.detach().cpu().numpy()
faiss.normalize_L2(user_embedding)   # 这是inplace操作 (计算余弦相似度,先正则化,再计算内积)
index = faiss.IndexFlatIP(args.embed_dim)
index.add(user_embedding)
# ascontiguousarray()函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
# 这里因为idx都是全的,方便索引,所以没有用字典
D, I = index.search(np.ascontiguousarray(user_embedding), args.TopK)


# 推荐TopN相似商品
print('开始生成推荐列表...')
rec_dict = dict()
val_rec = {}
rel_dict = valid_users
for val_user in tqdm(rel_dict):   # rel_dict: {uid1:[item1,item2,...],uid2:[item3,item4,...],...}
    rec_dict[val_user] = dict()
    for array_idx,user in enumerate(I[val_user]):    # I[val_user] : 和val_user TopK相似的用户
        for item in train_users[user]:               #
            if item not in train_users[val_user]:    # 这两行：筛选相似用户看过的、目标用户没看过的物品
                if item not in rec_dict[val_user]:
                    rec_dict[val_user][item]=0
                rec_dict[val_user][item] += D[val_user][array_idx]


Top50_rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:50] for k, v in rec_dict.items()}
Top50_rec_dict = {k: list([x[0] for x in v]) for k, v in Top50_rec_dict.items()}
Top10_rec_dict = {k: v[:10] for k, v in Top50_rec_dict.items()}
Top20_rec_dict = {k: v[:20] for k, v in Top50_rec_dict.items()}
# evaluate
print('Top 10:')
rec_eval(Top10_rec_dict,rel_dict,train_users)
print('Top 20:')
rec_eval(Top20_rec_dict,rel_dict,train_users)
print('Top 50:')
rec_eval(Top50_rec_dict,rel_dict,train_users)

"""
# 计算RMSE
predict_rating_all = np.matmul(user_embedding,item_embedding.T)
valid_matrix = []
for idx,pair in enumerate(valid_data[0]):
    user,item = pair
    rating = valid_data[1][idx]
    valid_matrix.append([user,item,rating])
valid_matrix = pd.DataFrame(valid_matrix,columns=['userID','itemID','Rating'])
valid_list = valid_matrix['Rating'].to_list()
predict_list = []

for idx, row in valid_matrix.iterrows():
    userID,itemID = row['userID'],row['itemID']
    predict_list.append(predict_rating_all[userID][itemID])

rmse = RMSE(valid_list,predict_list)
print(f'均方根误差RMSE: {round(rmse,5)}')
"""
    