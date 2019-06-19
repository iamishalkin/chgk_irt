import pickle
import numpy as np 
import pandas as pd

READ=True
if READ:
    with open('results.pickle', 'rb') as f:
        results = pickle.load(f)
else:
    with open('results.pickle', 'wb') as f:
        pickle.dump(results, f)

def explode(df, lst_cols, fill_value='', preserve_index=False):
    # func to make long table
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    orders = [num for range_len in lens for num in range(range_len)]
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    
    res.loc[:, 'q_numb'] = orders
    return res

exp = explode(results.loc[:, ['idteam', 'idtournament', 'mask', 'players']], ['mask'], fill_value='', preserve_index=False)\
            .rename(columns={'mask': 'q_taken'})

exp.loc[:, 'q_id'] = exp.idtournament * 1000 + exp.q_numb
exp['q_taken'] = exp.q_taken.astype(int)

LOAD = True
if LOAD:
    with open('skills.pickle', 'rb') as f:
        skills = pickle.load(f)
    with open('difficulties.pickle', 'rb') as f:
        difficulties = pickle.load(f)
else:
    skill_ids = roster.sort_values('idplayer').idplayer.unique()
    skills = np.array(list(zip(skill_ids, np.random.uniform(0,1, len(skill_ids))))   )

    dif_ids = np.sort(exp.q_id.unique())
    difficulties = np.array(list(zip(dif_ids, np.random.uniform(0,1, len(dif_ids))))   )

print('Vars loaded!')