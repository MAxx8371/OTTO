import pandas as pd
import numpy as np
from fastparquet import ParquetFile
import pickle
from tqdm import tqdm
from pandarallel import pandarallel
import random
from joblib import Parallel, delayed
import itertools
import yaml
from gensim.models import Word2Vec
import time

'''dict generation'''
# tqdm.pandas()

# path = r'C:\Users\MAx\Desktop\OTTO\archive\train.parquet'
# pf = ParquetFile(path)
# df = pf.to_pandas()

# aid2idx = {}
# aid_set = df['aid'].unique()
# for idx, aid in enumerate(aid_set):
#     aid2idx[aid]  = idx
# pickle.dump(aid2idx, file=open('./archive/aid2idx.pkl', 'wb+'))

'''graph generation'''
# aid2idx = pickle.load(file=open('./archive/aid2idx.pkl', 'rb'))

# graph_dict = {}
# def get_graph(x):
#     df = x.sort_values(by='ts',ascending=True)
#     pre_idx = -1
#     pre_ts = -1
#     for index,row in df.iterrows():
#         if pre_idx == -1:
#             pre_idx = aid2idx[row['aid']]
#             pre_ts =  datetime.utcfromtimestamp(row['ts'])
#         else:
#             cur_idx = aid2idx[row['aid']]
#             cur_ts = datetime.utcfromtimestamp(row['ts'])
#             if (cur_ts-pre_ts).total_seconds()/60 <= 15 :  #前后时间差小于15min
#                 if pre_idx in graph_dict:
#                     if cur_idx in graph_dict[pre_idx]:
#                         graph_dict[pre_idx][cur_idx] += 1
#                     else:
#                         graph_dict[pre_idx][cur_idx] = 1
#                 else:
#                     graph_dict[pre_idx] = {}
#                     graph_dict[pre_idx][cur_idx] = 1
            
#             pre_idx = cur_idx
#             pre_ts = cur_ts

# df.groupby('session').progress_apply(get_graph)
# pickle.dump(graph_dict, file=open('./archive/graph_15.pkl', 'wb+'))

'''corpus generation'''
# def deepwalk(walk_length, start_node):
#     walk = [start_node]
#     while len(walk) < walk_length:
#         cur = walk[-1]
#         if cur not in graph_dict:
#             break
#         cur_nbrs = list(graph_dict[cur].keys())
#         cur_weights = list(graph_dict[cur].values())  # 加权
#         if len(cur_nbrs) == 0 or (len(cur_nbrs)==1 and cur_nbrs[0]==cur):
#             break
#         else:
#             nxt = random.choices(cur_nbrs,weights=cur_weights)[0]
#             while(cur==nxt):
#                 nxt = random.choices(cur_nbrs,weights=cur_weights)[0]
#             walk.append(nxt)
#     return walk

# def _simulate_walks(num_walks, walk_length):
#     walks = []
#     nodes = [i for i in range(dict_len)]
#     for _ in range(num_walks):
#         random.shuffle(nodes)
#         for v in tqdm(nodes):           
#             walks.append(deepwalk(walk_length=walk_length, start_node=v))
#     return walks

# def partition_num(num, workers):
#     if num % workers == 0:
#         return [num // workers] * workers
#     else:
#         return [num // workers] * workers + [num % workers]


if __name__ == '__main__':
    # dict_len = 1855603
    # graph_dict = pickle.load(file=open('./archive/interval_15/graph_15.pkl', 'rb'))

    # hpy_path = r'cfg\graph_hpy.yaml'
    # with open(hpy_path) as f:
    #     hyp = yaml.load(f, Loader=yaml.FullLoader)

    # 并行处理
    # results = Parallel(n_jobs=hyp['workers'], verbose=True)(
    #     delayed(_simulate_walks)(num, hyp['walk_length']) for num in partition_num(hyp['num_walks'],hyp['workers']))

    # corpus = list(itertools.chain(*results))
    # pickle.dump(corpus, file=open('./archive/interval_15/corpus_15.pkl', 'wb+'))

    # 串行处理
    # corpus = []
    # nodes = [i for i in range(dict_len)]
    # for i in range(hyp['num_walks']):
    #     print('epoch:{}'.format(i))
    #     random.shuffle(nodes)
    #     for v in tqdm(nodes):       
    #         walk = deepwalk(walk_length=hyp['walk_length'], start_node=v)    
    #         if len(walk)>=2:
    #             corpus.append(walk)
    # pickle.dump(corpus, file=open('./archive/interval_15/corpus_15.pkl', 'wb+'))

    begin_time = time.time()

    corpus = pickle.load(file=open('./interval_15/corpus_15.pkl', 'rb'))
    model = Word2Vec(corpus, vector_size=128, window=5,min_count=1,sg=1,workers=10,epochs=12)
    model.save("./interval_15/word2vec.model")
    
    end_time = time.time()

    run_time = round(end_time-begin_time)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    print (f'训练时长：{hour}小时{minute}分钟{second}秒')