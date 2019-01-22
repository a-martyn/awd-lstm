#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'awd-lstm'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

#%%
path = '/Users/alanmartyn/git/awd-lstm/results/awd_lstm_fixed_seq_len.csv'
df = pd.read_csv(path)
df.head()

#%%
