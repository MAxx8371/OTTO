import pandas as pd
import numpy as np
from fastparquet import ParquetFile
import pickle
from tqdm import tqdm
import random
from joblib import Parallel, delayed
import itertools
import yaml
from matplotlib import pyplot as plt
import fcntl

#  period = 0 : 07-31 ~ 08-07
#  period = 1 : 08-07 ~ 08-14
#  period = 2 : 08-14 ~ 08-21
#  period = 3 : 08-21 ~ 08-28

# hours = {0: 10342, 1: 6957, 2: 6678, 3: 9862, 4: 19830, 5: 37305, 6: 55632, 7: 73070, 8: 80453, 9: 84646, 10: 87527, 11: 91916, 12: 93753, 13: 96050, 14: 101612, 15: 102131, 16: 104070, 17: 115483, 18: 126113, 19: 136348, 20: 112748, 21: 66551, 22: 34753, 23: 17973}
# days = {1: 219845, 2: 199814, 3: 223385, 4: 267161, 28: 6902, 29: 244627, 30: 260206, 31: 249863}

'''convert splited_set to dataframe'''
df = pd.DataFrame()
f = open("./splited_train_set.pkl","rb")
f.seek(0)
cnt = 0
while True: #这里while True是因为pickle.load函数一次只取出了文件中的一个对象
    try:
        item = pickle.load(f,encoding='bytes')
        cnt += 1
        print(cnt)
    except EOFError: #EOFError错误接收文件读取完毕后报错，此时关闭文件并跳出循环
        f.close()
        break

# with open('./dataset/splited_train_set_df.pkl',mode='ab') as f:
#     pickle.dump(df, file=f)

# if __name__ == '__main__':

