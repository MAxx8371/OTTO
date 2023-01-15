import pandas as pd
import numpy as np
from fastparquet import ParquetFile
import pickle
from tqdm import tqdm
import random
from joblib import Parallel, delayed

def get_action2order(df,type):
    df.sort_values('ts', inplace=True, ascending=True)
    last_time = -1
    for _,row in df.iterrows():
        if row['type']==type:
            last_time = row['ts']
        elif row['type']==2:
            if(row['ts']-last_time<=15*60):
                return 1
    return 0
    
def get_inherent_features(x):
    click_sum = x[x['type']==0]['ts'].count()
    cart_sum = x[x['type']==1]['ts'].count()
    order_sum = x[x['type']==2]['ts'].count()

    user_num = x['session'].nunique()

    click_user_sum = x[x['type']==0]['session'].nunique()
    cart_user_sum = x[x['type']==1]['session'].nunique()
    order_user_sum = x[x['type']==2]['session'].nunique()

    df_grouped = x.groupby('session')
    click2order_num = sum( Parallel(n_jobs=6, verbose=4, backend='multiprocessing')(
        delayed(get_action2order)(group,0) for _, group in df_grouped) )
    cart2ordered_num = sum( Parallel(n_jobs=6, verbose=4, backend='multiprocessing')(
        delayed(get_action2order)(group,1) for _, group in df_grouped) )

    return pd.Series({"user_num":user_num, "click_sum":click_sum, "cart_sum":cart_sum, "order_sum":order_sum,
                     "click_user_sum":click_user_sum, "cart_user_sum":cart_user_sum, "order_user_sum":order_user_sum,
                     "click2order_num":click2order_num, "cart2ordered_num":cart2ordered_num})

if __name__ == '__main__':
    tqdm.pandas()
    path = r'C:\Users\MAx\Desktop\OTTO\archive\train.parquet'
    pf = ParquetFile(path)
    df = pf.to_pandas()[0:1000]
    tmp = df.groupby('aid',as_index=False).progress_apply(get_inherent_features)

    tmp.to_pickle('data.pkl')

    print(tmp)

# print(tmp['aid'].count())
# print(tmp[tmp['click2order']>0.0]['aid'].count())
# print(tmp[tmp['cart2order']>0.0]['aid'].count())

# company=["A","B","C"]
# data=pd.DataFrame({
#     "company":[company[x] for x in np.random.randint(0,len(company),10)],
#     "salary":np.random.randint(5,50,10),
#     "age":np.random.randint(15,50,10)
# }
# )
# def get_oldest_staff(x):
#     df = x.groupby('salary',as_index=False)['age'].count()
#     tmp = x[0:1]['salary'].values[0]
#     return df[df['salary']==tmp]['count'].values[0]
#     return tmp
#     # print(x.groupby('salary').count().max())
#     # ctr = df['count'].max() / df['count'].min()

#     # return pd.Series({"ctr":ctr})

#     # return df.loc[x[0:1]['salary']]

# df = data.groupby('company',as_index=False).apply(get_oldest_staff)

# print(data)
# print()
# print(df)

