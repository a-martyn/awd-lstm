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
from datetime import datetime

#%%
results_dir = '/Users/alanmartyn/git/awd-lstm/results/'
results_file = 'awd_lstm_fixed_seq_len.csv'
output_file = 'memory_plot.png'
df = pd.read_csv(results_dir+results_file)
df.head()


#%%
x = df['epoch']
y = [df['memalloc_Gb'], df['memcache_Gb']]
plt.stackplot(x, y, labels=['memalloc_Gb', 'memcache_Gb'])
plt.legend(loc='upper left')
plt.xlabel('epoch')
plt.ylabel('Gb')
plt.title('Updated at: ' + str(datetime.now()))
plt.savefig(results_dir+output_file)


