# 01导入包以及设置随机种子
import os 
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
parser.add_argument('--embed_dim', type=int, default=256, help='the dimension of item embedding')
parser.add_argument('--epochs', type=int, default=50, help='the number of epochs to train for') # 50
parser.add_argument('--patience', type=int, default=5, help='EarlyStopping patience') 
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--TopN', type=int, default=20, help='number of top score items selected')
parser.add_argument('--K_nearest', type=int, default=20, help='number of similar items/users')
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
    train_data = (tra_X, tra_Y)
    valid_data = (val_X, val_Y)

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


user_embedding = model.u_emb.weight.detach().cpu().numpy()
item_embedding = model.i_emb.weight.detach().cpu().numpy()
index = faiss.IndexFlatIP(args.embed_dim)
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

# 计算RMSE
predict_rating_all = np.matmul(user_embedding,item_embedding.T)
valid_matrix = []
for idx,pair in enumerate(valid_data[0].T):
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

    