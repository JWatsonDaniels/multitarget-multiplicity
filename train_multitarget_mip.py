"""
This script contains the multi-target optimization problems.

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

# add the default settings for this script to the top
settings = {
    'n_samples': 1000,
    'data_name': 'dissecting_bias_dataset_three_outcomes_limited_features_train_and_holdout',
    'top_K': 10,
    'random_seed': 109,
    'DEBUG': True
    }


############ normal script starts here #################
from prm.paths import get_results_file_rank, get_json_file_rank
from prm.utils import compute_log_loss, print_log, predict_prob
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import scipy as sp
from itertools import product
from scipy import stats
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



data_name = settings['data_name']
from prm.paths import get_rank_data_csv
datafile = get_rank_data_csv(**settings)


c = pd.read_csv(datafile)

# # remove these columns: "index" "risk_score_t", "program_enrolled_t", "cost_t", "cost_avoidable_t"
# cols_to_remove = ["risk_score_t", "program_enrolled_t", "cost_t", "cost_avoidable_t"]
# all_dropped = c[[cols_to_remove]]
# c = c.drop(columns=cols_to_remove)

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



'''STEP 2: Train y_hat on Train for each target individually and save betas '''
# subset_data_df = data_df.sample(n=settings['n_samples'], random_state=settings['random_seed'])
y_multi = training_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
X = training_df.drop(['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t'], axis = 1)

X = X.to_numpy()
samples, dim = X.shape
intercept_idx = 0
Xf = np.insert(X, intercept_idx, 1.0, axis= 1)
# orthonormalize the features
# Xf = sp.linalg.orth(Xf)


top_K = settings['top_K']

y_pred_multi = []
y_rank_multi = []
multi_modelCoefs = []
multi_MSE = []

model_per_target = []

rank_method = 'ordinal'
n_targets = y_multi.shape[1]
# Decision variables

for target_idx in range(y_multi.shape[1]):
    y = np.array(y_multi[y_multi.columns[target_idx]])
    #Training the linear regression and printing results below
    regressor = LinearRegression()
    regressor.fit(X, y)

    y_pred = regressor.predict(X)

    train_MSE = mean_squared_error(y, y_pred)
    train_coefs = np.insert(arr = regressor.coef_, obj = intercept_idx, values = regressor.intercept_)

    print_log('***** Training Linear Regression Results (no constraints) *****')
    print_log('n_samples: {}'.format(samples))
    print_log('coefs: {}'.format(train_coefs))
    print_log("Mean squared error: %.2f" % train_MSE)

    y_pred_multi.append(Xf @ train_coefs)
    y_rank_multi.append(rankdata(-1.0* y_pred, method = rank_method))
    multi_modelCoefs.append(train_coefs)
    multi_MSE.append(train_MSE)
    model_per_target.append(regressor)

'''Compare the rankings between them'''
baseline_ranks_multi = np.array(y_rank_multi)

# How many of the rankings match or are changed
# # how many points have ranking different from baseline
baseline_ranks_df =pd.DataFrame(np.array(y_rank_multi).T, columns= ['y1', 'y2', 'y3'])

num_changed = 0
num_flipped = 0
flipped_idx_train =[]
for index, row in baseline_ranks_df.iterrows():
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
train_y_pred_df = pd.DataFrame(np.array(y_pred_multi).T, columns= ['y1', 'y2', 'y3'])
#axis=0 for pandas column
def standardize_y(y_df):
    '''
    𝑦−𝑚𝑒𝑎𝑛 (ˆ𝑦) / 𝑠𝑑 (ˆ𝑦)
    :param y_array:
    :return:
    '''
    return np.array((y_df - y_df.mean()) / y_df.std())

stand_y_pred_df = pd.DataFrame({'y1': standardize_y(train_y_pred_df['y1']),
                                 'y2': standardize_y(train_y_pred_df['y2']),
                                 'y3': standardize_y(train_y_pred_df['y3'])})













'''STEP 3: Use tune data for alpha formulation.'''

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



y_hat_tune = []
# tune rankings
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




tune_ranks_df = pd.DataFrame(np.array(tune_r).T, columns= ['y1', 'y2', 'y3'])



'''Set up alpha optimization problem'''
n_targets = stand_y_pred_tune_df.shape[1]

stand_yk_pred = np.array(stand_y_pred_tune_df)


# M = np.zeros(samples_tune)
#
# # Initialize Big-M values
# for pt in range(samples_tune):
#     # y_pred on point b
#     this_yk_pred = stand_yk_pred[pt]
#     # y_pred over any point being compared (points a)
#     a_array = [x for x in range(samples_tune)]
#     a_array.pop(pt)  # remove a=b
#     other_yk_preds = np.array([stand_yk_pred[a] for a in a_array])
#
#     # M[pt] = this_yk_pred.max() - (other_yk_preds.min()) # tighter bounds
#     M[pt] = stand_yk_pred.max() - stand_yk_pred.min() # this should work in theory
#
# M_i = np.zeros(samples_tune)
#
# # Initialize Big-M values
# for pt in range(samples_tune):
#     # y_pred on point b
#     this_yk_pred = stand_yk_pred[pt]
#     # y_pred over any point being compared (points a)
#     a_array = [x for x in range(samples_tune)]
#     a_array.pop(pt)  # remove a=b
#     other_yk_preds = np.array([stand_yk_pred[a] for a in a_array])
#
#     M_i[pt] = other_yk_preds.max() - (this_yk_pred.min()) # tighter bounds


big_M = stand_yk_pred.max() - stand_yk_pred.min() # this should work in theory



new_ranks_min = []
new_ranks_max = []

alphas_min = []
alphas_max = []

for pt_idx in range(samples_tune):

    '''Maximize formulation'''
    model_max = gp.Model("alpha_max")

    alpha = model_max.addVars(n_targets, lb = 0.0, ub =  1.0, name='alpha')
    I = model_max.addMVar(shape = (samples_tune, 2), vtype = GRB.BINARY, name = "I")
    # we need to compare pointwise between b and i
    # column = 0 keeps track of b > pt_idx
    # column = 1 keeps track of pt_idx > b

    b_array = [x for x in range(samples_tune)]
    b_array.pop(pt_idx)  # remove a=b
    obj = sum(I[b, 0] for b in b_array ) + 1
    alpha_term = (0.5 * sum(alpha[target] for target in range(n_targets)))
    model_max.setObjective(alpha_term + sum(I[b, 0] for b in b_array ) + 1 , GRB.MAXIMIZE)

    ## Constraints
    """Constraint #1 """
    # index model = sum alpha * y_pred

    # alphas sum to 1
    model_max.addConstr(sum(alpha[target] for target in range(n_targets)) <= 1.0, name='alpha_not_1')

    model_max.addConstr(sum(alpha[target] for target in range(n_targets)) >= 0.001, name='alpha_not_0')

    """Constraint #2 and #3 """
    for bb in range(samples_tune):
        # I_{ij} + I_{ji} <= 1 \forall j
        model_max.addConstr(I[bb, 0] + I[bb, 1] == 1.0, name="I_{}".format(bb))

        if (bb != pt_idx):
            this_index_model_b = sum(alpha[target] * stand_yk_pred[bb, target] for target in range(n_targets))
            this_index_model_pt = sum(alpha[target] * stand_yk_pred[pt_idx, target] for target in range(n_targets))

            # Big-M constraint
            # model_max.addConstr((this_index_model_b - this_index_model_pt) <= big_M * I[bb, 0],
            #                 name="M9b_{}".format(bb))

            model_max.addConstr((this_index_model_pt - this_index_model_b) <= big_M * I[bb, 1],
                                name="M9c_{}".format(bb))

    alpha.start = 1 / n_targets
    # Solve optimization problem
    model_max.update()
    model_max.params.timelimit = 1000
    model_max.setParam('OutputFlag', 0)  ## comment out to DEBUG
    model_max.setParam('LogToConsole', 0)  ## comment out to DEBUG
    model_max.params.mipgap = 0.001

    model_max.optimize()
    ## Unpack results
    a_lst = []
    for v in model_max.getVars()[0: n_targets]:
        item = {v.VarName: v.X}
        a_lst.append(v.X)

    post_alphas = np.array(a_lst)
    alphas_max.append(post_alphas)

    I_lst = []
    for v in model_max.getVars()[n_targets: n_targets + 2 * samples_tune]:
        item = {v.VarName: v.X}
        I_lst.append(v.X)

    I_var_result = np.array(I_lst).reshape(samples_tune, 2)

    optimal_rank = sum(I_var_result[b, 0] for b in b_array) + 1 + (0.5 * sum(post_alphas[target] for target in range(n_targets)))
    # print_log('Optimal rank: {}'.format(optimal_rank))
    new_ranks_max.append(optimal_rank)

    '''Minimize formulation'''
    model_min = gp.Model("alpha_min")

    alpha = model_min.addVars(n_targets, lb=0.0, ub=1.0, name='alpha')
    I = model_min.addMVar(shape=(samples_tune, 2), vtype=GRB.BINARY, name="I")
    # we need to compare pointwise between b and i
    # column = 0 keeps track of b > pt_idx
    # column = 1 keeps track of pt_idx > b

    obj = sum(I[b, 0] for b in b_array) + 1
    alpha_term = -1.0 * (0.5 * sum(alpha[target] for target in range(n_targets)))
    model_min.setObjective(alpha_term + sum(I[b, 0] for b in b_array) + 1, GRB.MINIMIZE)

    ## Constraints
    """Constraint #1 """

    # alphas sum to 1
    eta = 1e-5
    model_min.addConstr(sum(alpha[target] for target in range(n_targets)) <= 1.0, name='alpha_not_1')

    model_min.addConstr(sum(alpha[target] for target in range(n_targets)) >= 0.001, name='alpha_not_0')

    """Constraint #2 and #3 """
    for bb in range(samples_tune):
        # I_{ij} + I_{ji} <= 1 \forall j
        model_min.addConstr(I[bb, 0] + I[bb, 1] == 1.0, name="I_{}".format(bb))

        if (bb != pt_idx):
            this_index_model_b = sum(alpha[target] * stand_yk_pred[bb, target] for target in range(n_targets))
            this_index_model_pt = sum(alpha[target] * stand_yk_pred[pt_idx, target] for target in range(n_targets))

            # Big-M constraint
            model_min.addConstr((this_index_model_b - this_index_model_pt) <= big_M * I[bb, 0],
                                name="M9b_{}".format(bb))

            # model_min.addConstr((this_index_model_pt - this_index_model_b) <= big_M * I[bb, 1],
            #                     name="M9c_{}".format(bb))

    alpha.start = 1 / n_targets
    # Solve optimization problem
    model_min.update()
    model_min.params.timelimit = 1000
    model_min.setParam('OutputFlag', 0)  ## test out DEBUG
    model_min.setParam('LogToConsole', 0)  ## test out DEBUG
    model_min.params.mipgap = 0.001

    model_min.optimize()
    ## Unpack results
    a_lst = []
    for v in model_min.getVars()[0: n_targets]:
        item = {v.VarName: v.X}
        a_lst.append(v.X)

    post_alphas = np.array(a_lst)
    alphas_min.append(post_alphas)

    I_lst = []
    for v in model_min.getVars()[n_targets: n_targets + 2 * samples_tune]:
        item = {v.VarName: v.X}
        I_lst.append(v.X)

    I_var_result = np.array(I_lst).reshape(samples_tune, 2)

    optimal_rank = sum(I_var_result[b, 0] for b in b_array) + 1 - (0.5 * sum(post_alphas[target] for target in range(n_targets)))
    # print_log('Optimal rank: {}'.format(optimal_rank))
    new_ranks_min.append(optimal_rank)



# convert results into dataframes
min_ranks_df = pd.DataFrame(new_ranks_min)
max_ranks_df = pd.DataFrame(new_ranks_max)

min_alphas_df = pd.DataFrame(alphas_min)
max_alphas_df = pd.DataFrame(alphas_max)

print_log('num unique values in min rank: {}'.format(min_ranks_df.nunique()))
print_log('num unique values in max rank: {}'.format(max_ranks_df.nunique()))

print_log('num unique values in min alphas: {}'.format(min_alphas_df.nunique()))
print_log('num unique values in max alphas: {}'.format(max_alphas_df.nunique()))


# For each point, we have a min and max rank
# We want to determine whether that rank has been flipped
num_flipped = 0
num_stable = 0
for each_pt in range(samples_tune):
    if ((np.array(new_ranks_min))[each_pt] <= top_K) and ((np.array(new_ranks_max))[each_pt] > top_K):
        num_flipped += 1

    if ((np.array(new_ranks_max))[each_pt] <= top_K):
        num_stable += 1

print('n points flipped {} / {}'.format(num_flipped, samples_tune))
print('n points always top {} / {}'.format(num_stable, top_K))

## For additional results 3/8/23
#  Extract dataframe with all alpha values for N = 1000, K = 40
#  Include min rank and max rank
# ID, min rank, max rank, alphas min rank, alphas max rank
# make sure to label which alpha is which. And standardized y hats.
# Id, yhat1, yhat2, yhat3, min rank, maximum rank, alpha for min rank, alpha for max rank
output_df = stand_y_pred_tune_df.copy()
# output_df['index'] = subset_data_df['index'].tolist()

output_df['min_rank'] = new_ranks_min
output_df['max_rank'] = new_ranks_max

# subset_data_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
min_alphas_array = np.array(alphas_min)
output_df['alpha_min_log_cost_t'] = min_alphas_array[:, 0]
output_df['alpha_min_log_cost_avoidable_t'] = min_alphas_array[:, 1]
output_df['alpha_min_gagne_sum_t'] = min_alphas_array[:, 2]

max_alphas_array = np.array(alphas_max)
output_df['alpha_max_log_cost_t'] = max_alphas_array[:, 0]
output_df['alpha_max_log_cost_avoidable_t'] = max_alphas_array[:, 1]
output_df['alpha_max_gagne_sum_t'] = max_alphas_array[:, 2]

output_df['n_flips'] = num_flipped

## save data to .csv
# output_df.to_csv('results/{}_sampled_multitarget_results_min9b_max9b.csv'.format(data_name))
output_df.to_csv('results/{}_sampled_multitarget_results_min9b_max9c.csv'.format(data_name))
# output_df.to_csv('results/{}_sampled_multitarget_results_min9c_max9c.csv'.format(data_name))

## save data to .csv
tune_df.to_csv('results/{}_sampled_multitarget_dataset.csv'.format(data_name))


'''Sanity Check results'''

# Calculate the combined yhat for each point by taking the dot product

check_ranks_min = []
check_ranks_max = []
for each_i in range(samples_tune):
    # check minimum rank with alphas
    min_alphas = alphas_min[each_i]
    this_y_pred_min = stand_y_pred_tune_df.dot(min_alphas)
    this_min_rank = rankdata(-1.0 * this_y_pred_min, method=rank_method)
    check_ranks_min.append(this_min_rank[each_i])

    # check max rank with alphas
    max_alphas = alphas_max[each_i]
    this_y_pred_max = stand_y_pred_tune_df.dot(max_alphas)
    this_max_rank = rankdata(-1.0 * this_y_pred_max, method=rank_method)
    check_ranks_max.append(this_max_rank[each_i])

# how does this manual check compare with the ranks from above?
min_compared = np.isclose(np.array(check_ranks_min), np.round(new_ranks_min).astype(int), rtol=1 )
# min_compared = np.array(check_ranks_min) == np.round(new_ranks_min).astype(int)
print('n pts close in min rank: {} / {}'.format(np.sum(min_compared), samples_tune))

max_compared = np.isclose(np.array(check_ranks_max), np.array(new_ranks_max), rtol=1 )
# max_compared = np.array(check_ranks_max) == np.round(new_ranks_max).astype(int)
print('n pts close in max rank: {} / {}'.format(np.sum(max_compared), samples_tune))
#





