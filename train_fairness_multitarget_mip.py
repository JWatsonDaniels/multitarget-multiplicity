"""
This script contains the multi-target optimization MIP including fairness consideration. We want to maximize the number of
people from protected group A who are in the top-K.

We begin by loading a dataset that includes multiple reasonable options for target variable. Then, train
a ranked output model for each target separately. We compare the conflicting ranks across this set of models
as a baseline. Then we set up an optimization problem for the index model parametrized by alpha and solve that
for each point.


"""
import os
import sys
import psutil
import dill
import json
import time
from datetime import timedelta
import argparse


############ normal script starts here #################
from prm.paths import get_results_file_rank, get_json_file_rank
from prm.utils import print_log
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import scipy as sp
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import rankdata
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# from matplotlib.offsetbox import AnchoredText
import seaborn as sns


def manual_rank(I, idx, samples):
    '''

    Parameters
    ----------
    I: Indicator matrix whether b > a
    pt_idx: point index

    Returns
    -------
    ordinal rank
    '''
    b_array = [x for x in range(samples)]
    b_array.pop(idx)  # remove a=b
    return int(sum(I[idx, b] for b in b_array ) + 1)

def indicator_b_greater_a(X_array, coefs_array, n_samples):
    '''
    Function to produce the indicator matrix determining whether
    example b should be ranked higher than example a.

    Parameters
    ----------
    X_array
    coefs_array
    n_samples

    Returns
    -------
    I matrix
    '''
    I = np.zeros([n_samples, n_samples])
    for k in range(n_samples):
        for l in range(n_samples):
            if (k != l):
                I[k, l] = (X_array[l, :] @ coefs_array) > (X_array[k, :] @ coefs_array)
    return I

def compute_RSS(X_array, y_array, beta_array):
    '''
    Computes 1/2 RSS objective.
    :param X_array:
    :param y_array:
    :param beta_array:
    :return: 1/2 RSS
    '''
    n, d_dim = X_array.shape
    Quad = np.dot(X_array.T, X_array)
    lin = np.dot(y_array.T, X_array)

    obj = sum(0.5 * Quad[i, j] * beta_array[i] * beta_array[j]
              for i, j in product(range(d_dim), repeat=2))
    obj -= sum(lin[i] * beta_array[i] for i in range(d_dim))
    obj += 0.5 * np.dot(y_array, y_array)
    return obj

def set_RSS_obj(model, y_array, w_lb = -GRB.INFINITY, w_ub = GRB.INFINITY):
    '''
    Set model objective for Linear Regression.
    :param model: gurobi Model object
    :param w_lb: lower bound on model coefficients
    :param w_ub: lower bound on model coefficients
    :return: gurobi Var - beta
    '''
    beta = model.addVars(dim + 1,
                             lb = w_lb,
                             ub =  w_ub,
                             name="beta") # Weights

    intercept = beta[intercept_idx] # first decision variable captures intercept
    intercept.varname = 'intercept'

    # Objective Function (OF): minimize 1/2 * RSS using the fact that
    # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
    Quad = np.dot(Xf.T, Xf)
    lin = np.dot(y_array.T, Xf)

    obj = sum(0.5 * Quad[i,j] * beta[i] * beta[j]
              for i, j in product(range(dim + 1), repeat=2))
    obj -= sum(lin[i] * beta[i] for i in range(dim + 1))
    obj += 0.5 * np.dot(y_array, y_array)
    model.setObjective(obj, GRB.MINIMIZE)

    return beta

# parse command line arguments to get settings file
parser = argparse.ArgumentParser()
parser.add_argument('settings_file', help='Path to the settings.json file')
args = parser.parse_args()

# Read the settings file
with open(args.settings_file, 'r') as file:
    settings = json.load(file)

data_name = settings['data_name']
from prm.paths import get_rank_data_csv
datafile = get_rank_data_csv(**settings)
obj = settings['obj']




c = pd.read_csv(datafile)
# # remove these columns: "index" "risk_score_t", "program_enrolled_t", "cost_t", "cost_avoidable_t"
# cols_to_remove = ["index", "risk_score_t", "program_enrolled_t", "cost_t", "cost_avoidable_t"]
# all_dropped = c[["index", 'dem_race_black', "risk_score_t", 'gagne_sum_t', "cost_t", "cost_avoidable_t", "program_enrolled_t", "split" ]]
#

if (c.shape[1] >30):
    cols_to_remove = ["index", "risk_score_t", "program_enrolled_t", "cost_t", "cost_avoidable_t"]
    all_dropped = c[["index", 'dem_race_black', "risk_score_t", 'gagne_sum_t', "cost_t", "cost_avoidable_t", "program_enrolled_t", "split" ]]
else:
    cols_to_remove = []
    all_dropped = c[['dem_race_black', "split"]]

c = c.drop(columns=cols_to_remove)

# check if there are duplicates in X,y for any of the targets
def check_duplicates(xy_df):
    any_duplicates = 0
    testing_df = xy_df.copy()
    xy_1 = testing_df.copy().drop(['log_cost_avoidable_t', 'gagne_sum_t'], axis = 1)
    if (np.sum(xy_1.duplicated()) > 0):
        print_log('****DUPLICATES ERROR**** for target = log_cost_t')
        any_duplicates+=1

    xy_2 = testing_df.copy().drop(['log_cost_t', 'gagne_sum_t'], axis = 1)
    if (np.sum(xy_2.duplicated()) > 0):
        print_log('****DUPLICATES ERROR**** for target = log_cost_avoidable_t')
        any_duplicates += 1

    xy_3 = testing_df.copy().drop(['log_cost_t', 'log_cost_avoidable_t'], axis = 1)
    if (np.sum(xy_3.duplicated()) > 0):
        print_log('****DUPLICATES ERROR**** for target = gagne_sum_t')
        any_duplicates += 1

    return any_duplicates

any_before_noise = check_duplicates(xy_df = c)

import random
random.seed(settings['random_seed'])
if (any_before_noise > 0):
    ## Add random noise to last column to prevent duplicates
    mu, sigma = 0.0, 0.001
    noise = np.random.normal(loc=mu, scale=sigma, size= c.shape[0])
    c['cost_radiology_tm1'] = c['cost_radiology_tm1'] + noise


any_after_noise = check_duplicates(xy_df = c)
if (any_after_noise == 0):
    print_log('DUPLICATES RESOLVED. random noise added to last column of features')

health_data = c.loc[c['split'] == 'train']
hold_out = c.loc[c['split'] == 'holdout']

'''STEP 1: Split train0 into Train and Tune. And set up dataframe for training. '''
train0_df, tune_df = train_test_split(health_data, test_size = 0.1,  random_state=settings['random_seed'])

# Drop columns not needed for training (split and demographics sex and race)
race_black_train0 = train0_df['dem_race_black']
sex_group_train0 = train0_df['dem_female']



training_df = train0_df.drop(['dem_race_black', 'dem_female', 'split'], axis=1)






'''STEP 2: Train y_hat on Train and save betas '''
y_multi = training_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
X = training_df.drop(['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t'], axis = 1)
X = X.to_numpy()
samples, dim = X.shape
intercept_idx = 0
Xf = np.insert(X, intercept_idx, 1.0, axis= 1)


'''Train the prediction model for each target individually'''
top_K = int(samples * settings['top_K_proportion'])

y_pred_multi = []
y_rank_multi = []
multi_modelCoefs = []
multi_MSE = []

model_per_target = []

rank_method = 'ordinal'

# Decision variables

for target_idx in range(y_multi.shape[1]):
    y = np.array(y_multi[y_multi.columns[target_idx]])
    #Training the linear regression and printing results below
    regressor = LinearRegression()
    regressor.fit(X, y)

    y_pred = regressor.predict(X)

    train_MSE = mean_squared_error(y, y_pred)
    train_coefs = np.insert(regressor.coef_, intercept_idx, regressor.intercept_)

    print_log('***** Training Linear Regression Results (no constraints) *****')
    print_log('n_samples: {}'.format(samples))
    print_log('coefs: {}'.format(train_coefs))
    print_log("Mean squared error: %.2f" % train_MSE)

    y_pred_multi.append(Xf @ train_coefs)
    y_rank_multi.append(rankdata(-1.0* y_pred, method = rank_method))
    multi_modelCoefs.append(train_coefs)
    multi_MSE.append(train_MSE)
    model_per_target.append(regressor)

    # y_df = y_multi[y_multi.columns[target_idx]]
    # y_df_with_train_index = y_df.reset_index()
    # y_df_with_train_index['y_pred'] = y_pred
    # y_df_with_train_index = (y_df_with_train_index.sort_values(by='y_pred', ignore_index = True, ascending=False)).reset_index()
    # y_df_with_train_index['rank'] = y_df_with_train_index['level_0'] + 1
    #
    # # now we need to recover the original y indices in the order that they were in before
    # df_with_rank = y_df_with_train_index.reset_index()

    # SAVE BASELINE RESULTS

    # convert predictions to a dataframe and simply sort to find ranking with a column
    # holdout_pred_df.sort_values(by=y_hat_col, ascending=False)

    # train_I_ab = indicator_b_greater_a(X_array = Xf,
    #                                    coefs_array = train_coefs,
    #                                    n_samples = samples)
    #
    # training_rank = np.array([manual_rank(I = train_I_ab, idx = pt, samples=samples) for pt in range(samples)])


'''Compare the rankings between them'''
training_ranks_multi = np.array(y_rank_multi)

# How many of the rankings match or are changed
# # how many points have ranking different from baseline
training_ranks_df =pd.DataFrame(np.array(y_rank_multi).T, columns= ['y1', 'y2', 'y3'])



num_changed = 0
num_flipped = 0
flipped_idx_train =[]
for index, row in training_ranks_df.iterrows():
    last_n_flipped = num_flipped

    these_ranks = np.array([row['y1'], row['y2'], row['y3']])
    if (len(np.unique(these_ranks > 1))):
        num_changed+=1
    elif (row['y1'] <= top_K ):
        if (row['y2'] > top_K) or (row['y3'] > top_K):
            num_flipped+=1
    elif (row['y2'] <= top_K ):
        if (row['y1'] > top_K) or (row['y3'] > top_K):
            num_flipped+=1
    elif (row['y3'] <= top_K ):
        if (row['y1'] > top_K) or (row['y2'] > top_K):
            num_flipped+=1

    if (row['y1'] > top_K ):
        if (row['y2'] <= top_K) or (row['y3'] <= top_K):
            num_flipped+=1
    elif (row['y2'] > top_K ):
        if (row['y1'] <= top_K) or (row['y3'] <= top_K):
            num_flipped+=1
    elif (row['y3'] > top_K ):
        if (row['y1'] <= top_K) or (row['y2'] <= top_K):
            num_flipped+=1

    if (last_n_flipped < num_flipped):
        flipped_idx_train.append(index)


# agreement between pairs of models
# (np.array(training_ranks_df <= top_K).T.astype(int) @ np.array(training_ranks_df <= top_K).astype(int) )
# this doesnt give us where they all agree

# (y1 dot y2) dot y3 would give us

print_log('***** Training Multi-target problem (no alpha) *****')
print_log('n: {}'.format(samples))
print_log('num pts changed rank: {}'.format(num_changed))
print_log('num pts flipped top_K: {}'.format(num_flipped))


'''Standardize y prediction'''
training_y_pred_df = pd.DataFrame(np.array(y_pred_multi).T, columns= ['y1', 'y2', 'y3'])
#axis=0 for pandas column
def standardize_y(y_df):
    '''
    𝑦−𝑚𝑒𝑎𝑛 (ˆ𝑦) / 𝑠𝑑 (ˆ𝑦)
    :param y_array:
    :return:
    '''
    return np.array((y_df - y_df.mean()) / y_df.std())

stand_y_pred_df = pd.DataFrame({'y1': standardize_y(training_y_pred_df['y1']),
                                 'y2': standardize_y(training_y_pred_df['y2']),
                                 'y3': standardize_y(training_y_pred_df['y3'])})

n_targets = stand_y_pred_df.shape[1]



















'''STEP 4: Use tune data for alpha formulation.'''

# Drop columns not needed for training (split and demographics sex and race)

if (settings['DEBUG'] == True):
    sampled_df = tune_df.sample(n = settings['n_samples'], random_state=settings['random_seed'])
    race_black_tune = sampled_df['dem_race_black']
    sex_group_tune = sampled_df['dem_female']
    tune_alphas_df = sampled_df.drop(['dem_race_black', 'dem_female', 'split'], axis = 1)
else:
    race_black_tune = tune_df['dem_race_black']
    sex_group_tune = tune_df['dem_female']
    tune_alphas_df = tune_df.drop(['dem_race_black', 'dem_female', 'split'], axis=1)



y_multi_tune = tune_alphas_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
X_tune = tune_alphas_df.drop(['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t'], axis = 1)
X_tune = X_tune.to_numpy()
samples_tune, dim_tune = X_tune.shape
Xf_tune = np.insert(X_tune, intercept_idx, 1.0, axis= 1)

top_K_tune = int(samples_tune * settings['top_K_proportion'])

y_hat_tune = []
# tune rankings
tune_I = []
tune_r = []
for each_t in range(n_targets):
    this_regressor = model_per_target[each_t]
    y_pred = this_regressor.predict(X_tune)
    y_hat_tune.append(y_pred)
    tune_r.append(rankdata(-1.0 * y_pred, method=rank_method))

y_pred_multi_tune = pd.DataFrame(np.array(y_hat_tune).T, columns= ['y1', 'y2', 'y3'])

# then standardize those y_hats
stand_y_pred_tune_df = pd.DataFrame({'y1': standardize_y(y_pred_multi_tune['y1']),
                                 'y2': standardize_y(y_pred_multi_tune['y2']),
                                 'y3': standardize_y(y_pred_multi_tune['y3'])})



tune_I_ab = np.array(tune_I)
tune_ranks_df = pd.DataFrame(np.array(tune_r).T, columns= ['y1', 'y2', 'y3'])

tune_pt_in_top = (np.array(tune_ranks_df['y3']) <= top_K_tune).astype(int)



# then use them to find alphas
'''Set up alpha optimization problem'''
stand_yk_pred_tune = np.array(stand_y_pred_tune_df)


M = np.zeros(samples_tune)

# Initialize Big-M values
for pt in range(samples_tune):
    # y_pred on point b
    this_yk_pred = stand_yk_pred_tune[pt]
    # y_pred over any point being compared (points a)
    a_array = [x for x in range(samples_tune)]
    a_array.pop(pt)  # remove a=b
    other_yk_preds = np.array([stand_yk_pred_tune[a] for a in a_array])

    # now what is the max difference between these?
    # M[pt] = this_yk_pred.max() - (other_yk_preds.min())
    #
    M[pt] = stand_yk_pred_tune.max() - stand_yk_pred_tune.min()


big_M = stand_yk_pred_tune.max() - stand_yk_pred_tune.min()

new_ranks_max = []
alphas_max = []

race_black_indicator = np.array(race_black_tune)
M_T0 = top_K_tune
M_T1 = samples_tune - top_K_tune

n_groupA = int(np.sum(race_black_indicator))
idx_groupA = np.where(race_black_indicator == 1)[0]

'''Maximize formulation'''
model_max = gp.Model("alpha_max")

alpha = model_max.addVars(n_targets, lb = 0.0, ub =  1.0, name='alpha')

T = model_max.addMVar(shape=(n_groupA), vtype=GRB.BINARY, name="T")
# we have data on which indiviudals are Black in subset_race_df
# we need to know proportion of these individuals in the top_K

# I_ab compares a>b
# I_ba compares b>a
I_ab = model_max.addMVar(shape = (n_groupA, samples_tune), vtype = GRB.BINARY, name = "I_ab")
I_ba = model_max.addMVar(shape = (samples_tune, n_groupA), vtype = GRB.BINARY, name = "I_ba")



# sum over group a indices
# obj = sum(T[ex] * race_black_indicator[ex] for ex in range(samples_tune))
# alpha_term = (0.5 * sum(alpha[target] for target in range(n_targets)))
if (obj == 'max'):
    model_max.setObjective(sum(T[ex] * race_black_indicator[ex] for ex in range(n_groupA)) +
                       (0.5 * sum(alpha[target] for target in range(n_targets))), GRB.MAXIMIZE)
else:
    model_max.setObjective(sum(T[ex] for ex in range(n_groupA)) -
                           (0.5 * sum(alpha[target] for target in range(n_targets))), GRB.MINIMIZE)

## Constraints
"""Constraint #1 """

# alphas sum to 1
model_max.addConstr(sum(alpha[target] for target in range(n_targets)) <= 1.0, name='alpha_not_1')

model_max.addConstr(sum(alpha[target] for target in range(n_targets)) >= 0.001, name='alpha_not_0')

# model_max.addConstr(sum(T[n] for n in range(samples_tune)) <= top_K_tune, name='alpha_not_0')


"""Constraint #2 and #3 """
for aa in range(n_groupA):
    ex_array = [x for x in range(samples_tune)]
    ex_array.pop(aa)
    # # Big-M constraint

    # this_rank = sum(I[aa, s] for s in ex_array ) + 1
    model_max.addConstr((top_K_tune - sum(I_ba[s, aa] for s in ex_array ) ) - M_T0 * T[aa] <= 0.0 ,
                        name="MT0_{}".format(aa))

    model_max.addConstr((1 + sum(I_ba[s, aa] for s in ex_array) - top_K_tune) - M_T1 * (1 - T[aa]) <= 0.0,
                        name="MT1_{}".format(aa))


    for bb in range(samples_tune):
        if (bb != aa):
            model_max.addConstr(I_ba[bb, aa] + I_ab[aa, bb] == 1.0, name="I_{},{}".format(bb, aa))

            this_index_model_bb = sum(alpha[target] * stand_yk_pred_tune[bb, target] for target in range(n_targets))
            this_index_model_aa = sum(alpha[target] * stand_yk_pred_tune[idx_groupA[aa], target] for target in range(n_targets))

            # # Big-M constraint
            model_max.addConstr((this_index_model_bb - this_index_model_aa) - M[bb] * I_ba[bb, aa] <= 0.0,
                                       name="M1_{},{}".format(bb, aa))

            model_max.addConstr((this_index_model_aa - this_index_model_bb) - M[idx_groupA[aa]] * I_ab[aa, bb] <= 0.0,
                                name="M2_{},{}".format(aa, bb))


alpha.start = 1 / n_targets


# Solve optimization problem
model_max.update()
model_max.params.timelimit = 600
model_max.setParam('OutputFlag', 0)  ## comment out to DEBUG
model_max.setParam('LogToConsole', 0)  ## comment out to DEBUG
# model_max.params.mipgap = 0.001
model_max.optimize()

## Unpack results
a_lst = []
for v in model_max.getVars()[0: n_targets]:
    item = {v.VarName: v.X}
    a_lst.append(v.X)

post_alphas = np.array(a_lst)
max_alphas_df = pd.DataFrame(post_alphas)



T_lst = []
for v in model_max.getVars()[n_targets: n_targets + n_groupA]:
    item = {v.VarName: v.X}
    T_lst.append(v.X)

T_var_result = np.array(T_lst)

obj_output = np.sum([T_var_result[ex]* race_black_indicator[ex]  for ex in range(n_groupA)])  + (0.5 * sum(post_alphas))

print_log('***** Max top-K selection for group A *****')
print_log('total sum group A: {}, top_K_tune: {}, Obj output: {}'.format(np.sum(race_black_indicator), top_K_tune, obj_output))



# print_log('num unique values in max rank: {}'.format(max_ranks_df.nunique())) ## debugging
# print_log('num unique values in max alphas: {}'.format(max_alphas_df.nunique())) ## debugging



index_tune_check = stand_y_pred_tune_df.dot(post_alphas)
index_rank_tune = pd.DataFrame(rankdata(-1.0 * index_tune_check, method=rank_method))

# we want to know how many Black patients are ranked in top_K originally
org_n_top_k = (tune_ranks_df <= top_K_tune).iloc[race_black_indicator == 1].sum(axis = 0)
print_log('n of group A in top k for tune orig y:\n{}'.format(org_n_top_k))
print_log('alphas:\n{}'.format(post_alphas))

# we want to know how many Black patients are ranked in top_K with index model
idx_n_top_k = (index_rank_tune <= top_K_tune).iloc[race_black_indicator == 1].sum(axis = 0)
print_log('n of group A in top k for tune INDEX:\n{}'.format(idx_n_top_k))

# test_alphas = np.array([0.5, 0.0, 0.5])
# index_tune_TEST = stand_y_pred_tune_df.dot(test_alphas)
# index_rank_TEST = pd.DataFrame(rankdata(-1.0 * index_tune_TEST, method=rank_method))
# # we want to know how many Black patients are ranked in top_K with index model
# idx_n_TEST = (index_rank_TEST <= top_K_tune).iloc[race_black_indicator == 1].sum(axis = 0)
# print_log('n of group A in top k for alpha [0.5, 0, 0.5]:\n{}'.format(idx_n_TEST))



## debugging test
# index_model_test1 = pd.DataFrame(stand_y_pred_tune_df.dot(previous_alphas), columns=['y_hat_prev'] )
# index_model_test1['new_yhat'] = stand_y_pred_tune_df.dot(post_alphas)
# index_model_test1['race_indicator'] = np.array(race_black_tune)
#
# debug_df = index_model_test1.sort_values(by='y_hat_prev')
# debug_df2 = index_model_test1.sort_values(by='new_yhat')
#
# index_rank_test = pd.DataFrame(rankdata(index_model_test, method=rank_method))
#
# index_model_test2 = stand_y_pred_tune_df.dot(post_alphas)
#
# index_rank_test2 = pd.DataFrame(rankdata(index_model_test2, method=rank_method))






















'''Step 5: find y_hat using training coefs to form index model.
 Check how many group A points are in the top for index vs y_hat models.'''

# holdout y_hats
# Drop columns not needed for training (split and demographics sex and race)
race_black_holdout = hold_out['dem_race_black']
sex_group_holdout = hold_out['dem_female']
holdout_df = hold_out.drop(['dem_race_black', 'dem_female', 'split'], axis=1)


y_multi_holdout = holdout_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
X_holdout = holdout_df.drop(['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t'], axis = 1)
X_holdout = X_holdout.to_numpy()
samples_holdout, dim_holdout = X_holdout.shape
Xf_holdout = np.insert(X_holdout, intercept_idx, 1.0, axis= 1)

top_K_holdout = int(samples_holdout * 0.03)

y_hat_holdout = []
rank_holdout = []
for each_t in range(n_targets):
    this_regressor = model_per_target[each_t]
    y_pred = this_regressor.predict(X_holdout)
    y_hat_holdout.append(y_pred)
    rank_holdout.append(rankdata(-1.0 * y_pred, method=rank_method))


y_pred_multi_holdout = pd.DataFrame(np.array(y_hat_holdout).T, columns= ['y1', 'y2', 'y3'])

rank_multi_holdout = pd.DataFrame(np.array(rank_holdout).T, columns= ['y1', 'y2', 'y3'])

# then standardize those y_hats
stand_y_pred_holdout_df = pd.DataFrame({'y1': standardize_y(y_pred_multi_holdout['y1']),
                                 'y2': standardize_y(y_pred_multi_holdout['y2']),
                                 'y3': standardize_y(y_pred_multi_holdout['y3'])})

## lets print the results first for the proportion of points using each of these individual y hats
y_hat_holdout_top_K = (rank_multi_holdout <= top_K_holdout).iloc[np.array(race_black_holdout) == 1].sum(axis = 0)
print_log('n of group A in top k for holdout orig y:\n{}'.format(y_hat_holdout_top_K))

## what is the index model result for proportion of points in top-K ?

# alphas_array = max_alphas_df.to_numpy().flatten()
### Previously found solutions
# alphas_array = np.array([0.20543498, 0.00312585, 0.79143916]) # 1000 N, top-K 30, ols_friendly [133]
# alphas_array = np.array([0.20653095, 0.0, 0.79346905]) # 500 N, top-K 15, limited_features [132]
# alphas_array = np.array([9.26934310e-05, 6.56947682e-01, 3.42959624e-01]) # 500 N, top-K 50, limited_features [130]
# alphas_array = np.array([0.0548116425096476, 0.0, 0.945188357490352]) # 500 N, top-K 15, ols_friendly [137]
# alphas_array = np.array([0.312284817367683, 0.0309785950475596, 0.656736587584758]) # 500 N, tuneK 50, limited_features [131]
# alphas_array = np.array([0.2313022 , 0.20195409, 0.56674371]) # 500 N, tuneK 50, ols_friendly [129]
# alphas_array = np.array([0.0 , 0.0, 1.0]) # 500 N, tuneK 15, ols_friendly [139]
alphas_array = np.array([0.04212436, 0.0, 0.95787564]) # 500 N, tuneK 15, ols_friendly, alpha lower bound 0.00001,  [137]

# top_K_tune = 15
# samples_tune = 500


index_model_holdout = stand_y_pred_holdout_df.dot(alphas_array)

index_rank = rankdata(-1.0 * index_model_holdout, method=rank_method)

index_rank_top_K = (index_rank <= top_K_holdout)
index_n_groupA = np.sum(index_rank_top_K[np.array(race_black_holdout) == 1])
print_log('n of group A in top k for INDEX MODEL:\n{}'.format(index_n_groupA))



## EXPERIMENTS #!
## save dataframe for Jake to plot results
# y_multi = training_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
dropped_hold_out = all_dropped.loc[all_dropped['split'] == 'holdout'].drop(columns="split")
dropped_hold_out['log_cost_t_hat'] = y_hat_holdout[0]
dropped_hold_out['gagne_sum_t_hat'] = y_hat_holdout[2]
dropped_hold_out['log_cost_avoidable_t_hat'] = y_hat_holdout[1]

# if risk_score_t is in the dataframe, then we can use it to calculate the percentile
if 'risk_score_t' in dropped_hold_out.columns:
    dropped_hold_out['risk_score_t_percentile'] = pd.qcut(dropped_hold_out['risk_score_t'].rank(method='first'), 100, labels=range(1, 101))

# ground truth index model
stand_y_true_holdout_df = pd.DataFrame({'y1': standardize_y(holdout_df['log_cost_t']),
                                 'y2': standardize_y(holdout_df['log_cost_avoidable_t']),
                                 'y3': standardize_y(holdout_df['gagne_sum_t'])})

index_true_holdout = stand_y_true_holdout_df.dot(alphas_array)
dropped_hold_out['index_model_t'] = np.array(index_true_holdout)

# y_pred index model
dropped_hold_out['index_model_t_hat'] = np.array(index_model_holdout)

dropped_hold_out['alpha_log_cost_t'] = alphas_array[0]
dropped_hold_out['alpha_log_cost_avoidable_t'] = alphas_array[1]
dropped_hold_out['alpha_gagne_sum_t'] = alphas_array[2]


# save results in a csv that is named to include the tune top_K and tune samples
if (obj == 'max'):
    dropped_hold_out.to_csv('results/maximize_index_results_tuneK{}_tuneN{}_{}.csv'.format(top_K_tune, samples_tune, data_name))
else:
    dropped_hold_out.to_csv('results/minimize_index_results_tuneK{}_tuneN{}_{}.csv'.format(top_K_tune, samples_tune, data_name))

quick_check = {'data': 'hold_out',
               'top_K': top_K_holdout,
               'frac_log_cost_t': y_hat_holdout_top_K[0] / race_black_holdout.sum(),
               'frac_log_cost_avoidable_t': y_hat_holdout_top_K[1] / race_black_holdout.sum(),
               'frac_gagne_t': y_hat_holdout_top_K[2] / race_black_holdout.sum(),
               'alpha_log_cost_t': alphas_array[0],
               'alpha_log_cost_avoidable_t': alphas_array[1],
               'alpha_gagne_t': alphas_array[2],
               'frac_index_model': index_n_groupA / race_black_holdout.sum()
               }

# save results in a csv that is named to include the tune top_K and tune samples
pd.DataFrame(quick_check, index=[0]).to_csv('results/check_fairmulti_{}_alphas_tuneK{}_tuneN{}.csv'.format(data_name, top_K_tune, samples_tune))

# '''EXPERIMENT 2 with Alex data'''
# ## save dataframe for Jake to plot results
# # y_multi = training_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
# dropped_hold_out = c.loc[c['split'] == 'holdout'].drop(columns="split")
# dropped_hold_out['log_cost_t_hat'] = y_hat_holdout[0]
# dropped_hold_out['gagne_sum_t_hat'] = y_hat_holdout[2]
# dropped_hold_out['log_cost_avoidable_t_hat'] = y_hat_holdout[1]
#
# dropped_hold_out['risk_score_t_percentile'] = pd.qcut(dropped_hold_out['risk_score_t'].rank(method='first'), 100, labels=range(1, 101))
#
# # ground truth index model
# stand_y_true_holdout_df = pd.DataFrame({'y1': standardize_y(holdout_df['log_cost_t']),
#                                  'y2': standardize_y(holdout_df['log_cost_avoidable_t']),
#                                  'y3': standardize_y(holdout_df['gagne_sum_t'])})
#
# index_true_holdout = stand_y_true_holdout_df.dot(alphas_array)
# dropped_hold_out['index_model_t'] = np.array(index_true_holdout)
#
# # y_pred index model
# dropped_hold_out['index_model_t_hat'] = np.array(index_model_holdout)
#
# dropped_hold_out['alpha_log_cost_t'] = alphas_array[0]
# dropped_hold_out['alpha_log_cost_avoidable_t'] = alphas_array[1]
# dropped_hold_out['alpha_gagne_sum_t'] = alphas_array[2]
#
#
# # save results in a csv that is named to include the tune top_K and tune samples
# dropped_hold_out.to_csv('results/table_2_index_results_tuneK{}_tuneN{}.csv'.format(top_K_tune, samples_tune))









