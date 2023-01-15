import pandas as pd
from joblib import Parallel, delayed
import gc

def gen_session_list(uid, df):
  df.sort_values('ts', inplace=True, ascending=True)
  last_time = -1
  cur_time = -1
  session_list = []
  session = []
  for row in df.iterrows():
    if last_time==-1:
      last_time = row['ts']
    else:
      if cur_time-last_time > 15*60:   #时间差大于15min则划入下一个session
        if len(session) > 2:
          session_list.append(session[:])
        session = []

      session.append(['aid'])
      last_time = cur_time
  if len(session) > 2:
    session_list.append(session[:])
  return uid, session_list


def gen_user_hist_session():
  user = pd.read_pickle('user_profile.pkl')
  total_user = user['user_id'].nunique()
  batch_size = 15000
  iters = (total_user-1) // batch_size + 1

  print(print("total", total_user, "iters", iters,"batch_size", batch_size))
  for i in range(iters):
    target_user = user['user_id'].values[i*batch_size: (i+1)*batch_size]
    sub_data = user.loc[user.user_id.isin(target_user)]
    df_grouped = sub_data.groupby('user_id')

    user_hist_session = Parallel(n_jobs=6, verbose=4, backend='mulpiprocessing')(
        delayed(gen_session_list)(name, group) for name, group in df_grouped)
    
    pd.to_pickle(user_hist_session,'user_hist_session_{}.pkl'.format(i))
    print(i, 'pickled')
    del user_hist_session
    gc.collect()
    print(i, 'del')
