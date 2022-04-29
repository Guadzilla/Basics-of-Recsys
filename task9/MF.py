# 01导入包以及设置随机种子
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import torch
import torch.nn as nn
import datetime
import pickle
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
parser.add_argument('--TopN', type=int, default=50, help='number of top score items selected')
parser.add_argument('--TopK', type=int, default=100, help='number of similar items/users')
parser.add_argument('--sample_ratio', type=int, default=1, help='nb of postive sample /nb of negtive sample')  
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args)

# 03 定义log
log_dir = os.curdir + '/log/' + str(str(datetime.datetime.now()))
writer = SummaryWriter(log_dir)


# 04 定义模型
class MF(nn.Module):
    def __init__(self, n_users, n_items, K):
        super(MF,self).__init__()
        self.emb_dim = K
        self.user_b = nn.Parameter(torch.zeros(size=(n_users,1), requires_grad=True, device=device))  # 用户打分偏置
        self.item_b = nn.Parameter(torch.zeros(size=(n_items,1), requires_grad=True, device=device))  # 商品打分偏置
        self.u_emb = nn.Embedding(n_users, self.emb_dim)
        self.i_emb = nn.Embedding(n_items, self.emb_dim)
        #self.sigmoid = nn.Sigmoid()
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


# 05 定义早停类
early_stopping = EarlyStopping(patience=args.patience,verbose=True)



# 06 定义数据集 load_data,Dataset,DataLoader
def load_data(file_path):
    """
    加载数据
    """
    with open(os.path.join(file_path,'train_users.txt'), 'rb') as f1:
        train_users = pickle.load(f1)
    with open(os.path.join(file_path,'valid_users.txt'), 'rb') as f2:
        valid_users = pickle.load(f2) 

    items_pool = []
    for _,item_list in train_users.items():
        items_pool += item_list

    return train_users, valid_users, items_pool


print('Loading data...')
train_users, valid_users, items_pool = load_data(file_path='./data')
origin_train_users = train_users
def RandomSelectNegativeSample(train_users, items_pool):
    """
    负采样
    """
    ret = dict()
    for uid, pos_items in train_users.items():
        ret[uid] = dict()
        for item in pos_items:
            ret[uid][item] = 1
        n = 0
        for i in range(0, len(train_users[uid]) * 5):
            item = items_pool[random.randint(0, len(items_pool) - 1)]
            if item in ret:
                continue
            ret[uid][item] = 0
            n += 1
            if n > args.sample_ratio * len(train_users):
                break
    return ret

print('负采样...')
train_data = RandomSelectNegativeSample(train_users, items_pool)

tra_X = []
tra_Y = []
for uid,ratings in train_data.items():
    for iid,rating in ratings.items():
        tra_X.append([uid,iid])
        tra_Y.append(rating)

val_X = []
val_Y = []
for uid,items in valid_users.items():
    for iid in items:
        val_X.append([uid,iid])
        val_Y.append(1)

train_data = (tra_X, tra_Y)
valid_data = (val_X, val_Y)

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

model = MF(n_users=6040, n_items=3883, K=args.embed_dim).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.lr_dc)


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
rel_dict = dict()
val_rec = {}
rel_dict = valid_users
for val_user in tqdm(valid_users):   # valid_users: {uid1:[item1,item2,...],uid2:[item3,item4,...],...}
    rec_dict[val_user] = dict()
    for array_idx,user in enumerate(I[val_user]):    # I[val_user] : 和val_user TopK相似的用户
        for item in origin_train_users[user]:               #
            if item not in origin_train_users[val_user]:    # 这两行：筛选相似用户看过的、目标用户没看过的物品
                if item not in rec_dict[val_user]:
                    rec_dict[val_user][item]=0
                rec_dict[val_user][item] += D[val_user][array_idx]


Top50_rec_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:50] for k, v in rec_dict.items()}
Top50_rec_dict = {k: list([x[0] for x in v]) for k, v in Top50_rec_dict.items()}
Top10_rec_dict = {k: v[:10] for k, v in Top50_rec_dict.items()}
Top20_rec_dict = {k: v[:20] for k, v in Top50_rec_dict.items()}
# evaluate
print('Top 10:')
rec_eval(Top10_rec_dict,rel_dict,origin_train_users)
print('Top 20:')
rec_eval(Top20_rec_dict,rel_dict,origin_train_users)
print('Top 50:')
rec_eval(Top50_rec_dict,rel_dict,origin_train_users)


def save_rec_dict(save_path,rec_dict):
    pickle.dump(rec_dict, open(os.path.join(save_path,'MF_rec_dict.txt'), 'wb'))

save_rec_dict('./data', rec_dict)
print('Done.')

