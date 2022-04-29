import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_save(load_path, save_path, test_rate = 0.1):
    """
    加载数据，分割训练集、验证集
    """
    data = pd.read_table(os.path.join(load_path,'ratings.dat'), sep='::', names = ['userID','itemID','Rating','Zip-code'])
    data = data[['userID','itemID']]
    uid_lbe = LabelEncoder()
    data['userID'] = uid_lbe.fit_transform(data['userID'])
    iid_lbe = LabelEncoder()
    data['itemID'] = iid_lbe.fit_transform(data['itemID'])

    train_data, valid_data = train_test_split(data,test_size=test_rate)
    
    train_users = train_data.groupby('userID')['itemID'].apply(list).to_dict()
    valid_users = valid_data.groupby('userID')['itemID'].apply(list).to_dict()

    train_data.to_csv('train_data.csv', index=False)
    valid_data.to_csv('valid_data.csv', index=False)
    pickle.dump(train_users, open(os.path.join(save_path,'train_users.txt'), 'wb'))
    pickle.dump(valid_users, open(os.path.join(save_path,'valid_users.txt'), 'wb'))

load_and_save(load_path='../dataset/ml-1m', save_path='./data', text_rate=0.1)