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

# train_set : 7.31 ~ 8.28
# test_set  : 9.28 ~ 9.04
# Start Train: 2022-08-01 06:00:00 (北京时间)
# Finish Train: 2022-08-29 05:59:59

# Start Test: 2022-08-28 22:00:00 
# Finish Test: 2022-09-04 21:59:51

def train_set_split(_, df):
    df.sort_values('ts', inplace=True, ascending=True)
    splits = [1659909600,1660514400,1661119200,1661724000]
    session = []
    idx, pre_ts, cur_ts = 0,-1,-1
    for _, row in df.iterrows():
        if pre_ts==-1:
            pre_ts = row['ts']
            session.append(row)
        else:
            cur_ts = row['ts']
            if cur_ts-pre_ts > 15*60:
                # with open('./splited_train_set.pkl',mode='ab') as f:
                with open('./splited_train_set.txt',mode='a') as f:
                    # fcntl.flock(f.fileno(),fcntl.LOCK_EX)
                    fcntl.flock(f,fcntl.LOCK_EX)
                    for sess in session:
                        sess['period'] = idx
                        # pickle.dump(sess.to_dict(), file=f) 
                        f.write(str(sess.to_dict())+'\n')
                    # fcntl.flock(f,fcntl.LOCK_UN)
                session.clear()
                if cur_ts > splits[idx]:
                    idx += 1
            else :
                if cur_ts > splits[idx]:
                    # with open('./splited_train_set.pkl',mode='ab') as f:
                    with open('./splited_train_set.txt',mode='a') as f:
                        fcntl.flock(f,fcntl.LOCK_EX)
                        # fcntl.flock(f.fileno(),fcntl.LOCK_EX)
                        for sess in session:
                            sess['period'] = idx
                            # pickle.dump(sess.to_dict(), file=f)
                            f.write(str(sess.to_dict())+'\n')
                    idx += 1
            session.append(row)
            pre_ts = cur_ts
    # with open('./splited_train_set.pkl',mode='ab') as f:   
    with open('./splited_train_set.txt',mode='a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)   
        # fcntl.flock(f.fileno(),fcntl.LOCK_EX)   
        for sess in session:
            sess['period'] = idx
            # pickle.dump(sess.to_dict(), file=f)
            f.write(str(sess.to_dict())+'\n')

if __name__=='__main__':
    tqdm.pandas()
    path = r'C:\Users\MAx\Desktop\OTTO\archive\train.parquet'
    pf = ParquetFile(path)
    df = pf.to_pandas()
    df = df[1:10000]
    df_grouped = df.groupby('session')

    # for _, group in tqdm(df_grouped):
    #     train_set_split(group)

    # 多进程写入文件时会出错,给文件加锁
    Parallel(n_jobs=10, verbose=0, backend='loky')(
        delayed(train_set_split)(name, group) for name,group in tqdm(df_grouped))

    # f = open("splited_train_set.pkl","rb")
    # while True: #这里while True是因为pickle.load函数一次只取出了文件中的一个对象
    #     try:
    #         item = pickle.load(f)
    #         print(item)
    #     except EOFError: #EOFError错误接收文件读取完毕后报错，此时关闭文件并跳出循环
    #         f.close()
    #         break