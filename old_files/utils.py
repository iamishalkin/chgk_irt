import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
import shared_vars


def team_performance(players):
    # calculate average team performance
    return np.mean(
        shared_vars.skills[np.searchsorted(shared_vars.skills[:,0], players), 1] 
    )
def get_difficulties_by_q_id(q_ids):
    return shared_vars.difficulties[np.searchsorted(shared_vars.difficulties[:,0], q_ids), 1]
def sigmoid(x):
    return 1 / (1 + pd.np.exp(-x))

def sgd(lr, df, epochs=10, batch_size=16):
    exp = df.copy(deep=True)
    for _ in tqdm(range(epochs)):
        # get batch
        batch = exp.sample(n=batch_size)
        # calculate team performance
        t_perf = batch['players'].apply(lambda members: team_performance(members))
        # get question difficulty
        q_dif = batch['q_id'].apply(lambda question: get_difficulties_by_q_id(question))
        # calculate  result 
        team_sigmoid = sigmoid(t_perf - q_dif)
        # update question difficulty
        q_dif_update = lr * (batch['q_taken'] * (1-team_sigmoid) - (1-batch['q_taken'])*team_sigmoid)
        shared_vars.difficulties[np.searchsorted(shared_vars.difficulties[:,0], batch['q_id']), 1] -= q_dif_update
        # update player shared_vars.skills
        skill_update = -(lr * (batch['q_taken'] * (1-team_sigmoid) - (1-batch['q_taken'])*team_sigmoid) /
                     batch['players'].str.len())
        for idx in range(batch_size):
            shared_vars.skills[np.searchsorted(shared_vars.skills[:,0], batch['players'].iloc[idx]), 1] -= skill_update.iloc[idx]