# 01导入包以及设置随机种子
import os 
import numpy as np
import torch
import torch.nn as nn
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorboardX import SummaryWriter
from tqdm import tqdm

from evaluate import *


import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 02 以类的方式定义超参数
class argparse():
    pass

args = argparse()
args.epochs = 10
args.batchsize = 128
args.K = 128
args.lr = 0.0005
args.device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
args.K_nearest = 100
args.TopN = 10

# 03 定义log
log_dir = os.curdir + '/log/' + str(str(datetime.datetime.now()))
writer = SummaryWriter(log_dir)


# 04 定义自己的模型
class MF(nn.Module):
    def __init__(self, n_users, n_items, K):
        super(MF,self).__init__()
        self.emb_dim = K
        self.u_emb = nn.Embedding(n_users, self.emb_dim)
        self.i_emb = nn.Embedding(n_items, self.emb_dim)

    def forward(self, data_u, data_i):
        user_emb = self.u_emb(data_u).unsqueeze(1)
        item_emb = self.i_emb(data_i).unsqueeze(2)
        outputs = user_emb @ item_emb
        return outputs.squeeze()


# 05 定义早停类(此步骤可以省略)

# 06 定义自己的数据集 load_data,Dataset,DataLoader
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
    train_data = (tra_X.T, tra_Y)
    valid_data = (val_X.T, val_Y)

    return n_users, n_items, train_data, valid_data


print('Loading data...')
file_paths = '../ml-1m/'
n_users, n_items, train_data, valid_data = get_data(file_paths)

class MovieDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[0]
        y = self.data[1]
        return x[0][index],x[1][index],y[index]

    def __len__(self):
        return len(self.data[1])



train_dataset = MovieDataset(train_data)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=True ,num_workers=8)
valid_dataset = MovieDataset(valid_data)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batchsize, shuffle=True ,num_workers=8)


# 07 实例化模型，设置loss，优化器等

model = MF(n_users, n_items, args.K).to(args.device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.1)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

# 08 开始训练
for epoch in tqdm(range(args.epochs)):
    model.train()
    sum_epoch_loss = 0
    for idx,(data_u,data_i,data_y) in enumerate(train_dataloader,0):
        data_u = data_u.to(args.device)
        data_i = data_i.to(args.device)
        data_y = data_y.to(args.device).to(torch.float32)
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
        data_u = data_u.to(args.device)
        data_i = data_i.to(args.device)
        data_y = data_y.to(args.device).to(torch.float32)
        outputs = model(data_u,data_i)
        loss = criterion(outputs,data_y)
        valid_epoch_loss += loss.item()
    valid_epoch_loss /= len(valid_dataloader)
    writer.add_scalar('loss/valid_loss', valid_epoch_loss, epoch)

# 09预测

# 此处可定义一个预测集的Dataloader,也可以直接将你的预测数据reshape,添加batch_size=1
import faiss


# 将验证集中的userID进行排序,方便与faiss搜索的结果进行对应
val_uids = sorted(set(train_data[0][0]))
trn_items = sorted(set(train_data[0][1]))

# 获取训练数据的实际索引与相对索引，
# 实际索引指的是数据中原始的userID
# 相对索引指的是，排序后的位置索引，这个对应的是faiss库搜索得到的结果索引
trn_items_dict = {}
for i, item in enumerate(trn_items):
    trn_items_dict[i] = item

trn_user_items = pd.DataFrame(train_data[0].T,columns=['userID','itemID']).groupby('userID')['itemID'].apply(list).to_dict()
val_user_items = pd.DataFrame(valid_data[0].T,columns=['userID','itemID']).groupby('userID')['itemID'].apply(list).to_dict()

# 使用向量搜索库进行最近邻搜索
user_embedding = model.u_emb.weight.detach().cpu().numpy()
item_embedding = model.i_emb.weight.detach().cpu().numpy()
index = faiss.IndexFlatIP(args.K)
index.add(item_embedding)
D, I = index.search(np.ascontiguousarray(user_embedding), args.K_nearest)

# 将推荐结果转换成可以计算评价指标的格式
# 选择最相似的TopN个item
val_rec = {}
for i, u in enumerate(val_uids):
    items = []
    for ii in I[i]:
        if ii in trn_items_dict:
            items.append(ii)
    #items = list(map(lambda x: trn_items_dict[x], list(I[i]))) # 先将相对索引转换成原数据中的userID
    items = list(filter(lambda x: x not in trn_user_items[u], items))[:10] # 过滤掉用户在训练集中交互过的商品id，并选择相似度最高的前N个
    val_rec[u] = set(items) # 将结果转换成统一的形式，便于计算模型的性能指标

# 计算评价指标
rec_eval(val_rec, val_user_items, trn_user_items)