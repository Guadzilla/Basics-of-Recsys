import pandas as pd

data = pd.read_table('ratings.dat', sep='::', names = ['userID','itemID','Rating','Zip-code'])
data = data.iloc[:10000,:]
data.to_csv('sample.csv')