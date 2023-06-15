"""
This script takes a dataset, orthonormalizes X, trains a baseline model with ranked output,
computes unflippable points to be removed from consideration, trains a candidate model
for each flippable point, then computes predictive multiplcity metric (ambiguity) for the
ranked output setting, and plots results.

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
    'data_name': 'dissecting_bias_dataset_all_outcomes_ols_friendly_features',
    'n_samples': 50,
    'top_K': 20,
    'random_seed': 109
    }


from prm.paths import get_results_file_rank, get_json_file_rank_AltMIP, get_results_file_rank_AltMIP
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
import folktables
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage

def compute_ambiguity(baseline, competing, K):
    '''
    Computes top-K ambiguity
    :return:
    '''
    # # how many points have ranking different from baseline
    num_bot_flips = 0
    num_top_flips = 0
    n = len(baseline)
    for r in range(n):
        if (baseline[r] <= top_K):
            # count if this_rank flips
            if (competing[r] > K):
                num_top_flips += 1
        else:
            if (competing[r] <= K):
                num_bot_flips += 1

    prop_all = (num_top_flips + num_bot_flips) / n
    prop_top = num_top_flips / K
    return prop_all, prop_top


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


def manual_rank(I, idx):
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

def set_RSS_obj(model, w_lb = -GRB.INFINITY, w_ub = GRB.INFINITY, warm_start = False, baseline_coefs = None):
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
    lin = np.dot(y.T, Xf)

    obj = sum(0.5 * Quad[i,j] * beta[i] * beta[j]
              for i, j in product(range(dim+1), repeat=2))
    obj -= sum(lin[i] * beta[i] for i in range(dim+1))
    obj += 0.5 * np.dot(y, y)
    model.setObjective(obj, GRB.MINIMIZE)

    if warm_start:
        beta.start = baseline_coefs

    return beta

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

data_name = settings['data_name']
from prm.paths import get_rank_data_csv
datafile = get_rank_data_csv(**settings)

c = pd.read_csv(datafile)
# remove these columns: "index" "risk_score_t", "program_enrolled_t", "cost_t", "cost_avoidable_t"
if (data_name != "dissecting_bias_dataset_three_outcomes_limited_features_train_and_holdout"):
    cols_to_remove = ["index", "risk_score_t", "program_enrolled_t", "cost_t", "cost_avoidable_t"]
    all_dropped = c[["index", 'dem_race_black', "risk_score_t", 'gagne_sum_t', "cost_t", "cost_avoidable_t", "program_enrolled_t", "split" ]]
    c = c.drop(columns=cols_to_remove)


health_data = c.loc[c['split'] == 'train']

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


# Drop columns not needed for training (split and demographics sex and race)
race_group = health_data['dem_race_black']
sex_group = health_data['dem_female']
data_df = health_data.drop(['dem_race_black', 'dem_female', 'split'], axis = 1)

# run script for each target in multi-target dataset
# sample a small subset of data
subset_data_df = data_df.sample(n=settings['n_samples'], random_state=settings['random_seed']) ## debugging locally
y_multi = subset_data_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]

# y_multi = data_df[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
n_targets = y_multi.shape[1]

X = subset_data_df.drop(['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t'], axis = 1)

'''Train the prediction model for each target individually'''
X = X.to_numpy()
intercept_idx = 0
Xf = np.insert(X, intercept_idx, 1.0, axis=1)

# orthonormalize the features to calculate the unflippable points but use the normal Xf for the MIP
Xf = sp.linalg.orth(Xf)
samples, dim = Xf.shape
dim = dim - 1

# for target_idx in range(n_targets):
target_idx = 2
y = np.array(y_multi[y_multi.columns[target_idx]])

# Set 'data_name' to reflect with multi-target variable is being run
settings['data_name'] = 'HEALTH2_' + y_multi.columns[target_idx]
results_file = get_results_file_rank_AltMIP(**settings)
json_file = get_json_file_rank_AltMIP(**settings)
data_name = y_multi.columns[target_idx]


top_K = settings['top_K']

# Decision variables
'''Training the baseline model and printing results below'''
start_time = time.monotonic()

regressor = gp.Model("Baseline")
beta = set_RSS_obj(regressor)
### Solve baseline problem with no constraints
regressor.params.timelimit = 60
regressor.params.mipgap = 0.001
regressor.optimize()

print_log('***** Baseline Linear Regression Results (no constraints) *****')
print_log('n_samples: {}'.format(samples))
print_log('RSS no constraints: %g' % regressor.ObjVal)
print_log('coefs: {}'.format(beta))

# SAVE BASELINE RESULTS
baseline_obj = regressor.ObjVal
baseline_coefs_lst = []
for v in regressor.getVars()[0:dim+1]:
    item = {v.VarName: v.X}
    baseline_coefs_lst.append(v.X)

baseline_coefs = np.array(baseline_coefs_lst)

'''Defining our functions to compute rank'''
baseline_I_ab = indicator_b_greater_a(X_array = Xf,
                                      coefs_array = baseline_coefs,
                                      n_samples = samples)

baseline_rank = np.array([manual_rank(I = baseline_I_ab, idx = pt) for pt in range(samples)])


## Because we now include epsilon in our constraints, we run the model over Rashomon set for all the different
## values of epsilon

epsilon = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
epsilons_to_train = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
obj_thresholds = (1 + epsilon) * baseline_obj

y_pred = Xf @ baseline_coefs
# Let 𝑖0 be a point not in the top-𝐾 according to the
# loss-minimizing ˆ𝒘, and let 𝑖 be a point whose predicted value is greater than that of point 𝑖0
# error_rate = 0.01

all_models = []
all_modelVars = []
all_modelCoefs = []
all_RSS = []

for error_rate in epsilons_to_train:
    print_log('**** epsilon: {}'.format(error_rate))
    max_epsilon = baseline_obj * error_rate  # set this as error rate * optimal 0.5 RSS
    unflippable = []
    for i_0 in range(samples):
        gaps = y_pred[i_0] - y_pred[(baseline_rank < baseline_rank[i_0])]
        X_greater = Xf[baseline_rank < baseline_rank[i_0]]
        norm_array = []
        for each_pt in X_greater:
            norm_array.append(np.sqrt(max_epsilon) * sp.linalg.norm(Xf[i_0] - each_pt))

        # B(i_0, i) =  \Delta_{i_0,i}(\hat\wb) + \sqrt{\epsilon}\|x_{i_0} - x_i\|_2 < 0
        # B(i_0, i) =  y_pred(i_0) - y_pred(i) + \sqrt{\epsilon}\|x_{i_0} - x_i\|_2 < 0
        bounds = gaps + np.array(norm_array)
        if (np.sum(bounds < 0.0) >= top_K):
            unflippable.append(i_0)

    feasible_pts = []
    for option in range(samples):
        if np.isin(option, np.array(unflippable), invert=True):
            feasible_pts.append(option)

    # print_log('***CHECK baseline rank of feasible pts: {}'.format(np.sort(baseline_rank[feasible_pts]))) ## DEBUGGING

    '''Initialize constants needed for the constrained MIP '''

    eta = 1e-6  # small number close to zero to avoid numerical issues when M = 0
    M = np.full([samples, samples], eta)

    # Initialize Big-M values
    for i in range(samples):
        for j in range(samples):
            if (i != j):
                # constant term \sqrt{||w_0||^2 + \epsilon }
                constant_term = np.sqrt(np.linalg.norm(baseline_coefs) ** 2 + max_epsilon)
                x_term = np.linalg.norm(Xf[j, :] - Xf[i, :])
                M[i, j] = x_term * constant_term

    M_constant = np.max(M)

    # I_idx_start = dim + 1
    # I_idx_end = samples**2 + dim + 1
    # r_idx_start, r_idx_end = samples**2 + dim + 1 , samples**2 + dim + 1 + samples

    '''For each flippable point, run a constrained MIP to flip that point. save results'''
    for pt_idx in np.array(feasible_pts):

        model = gp.Model("Candidates")
        # We can set the bounds here to inf because we will be adding a constraint that bounds w with max_epsilon
        beta = model.addVars(dim + 1,
                             lb=-GRB.INFINITY,
                             ub=GRB.INFINITY,
                             name="beta")

        intercept = beta[intercept_idx]  # first decision variable captures intercept
        intercept.varname = 'intercept'

        # Objective Function (OF): define obj as rank
        I = model.addMVar(shape=(samples, 2), vtype=GRB.BINARY, name="I")
        # we need to compare pointwise between b and i
        # column = 0 keeps track of b > pt_idx
        # column = 1 keeps track of pt_idx > b

        b_array = [x for x in range(samples)]
        b_array.pop(pt_idx)  # remove a=b
        obj = sum(I[b, 0] for b in b_array) + 1

        ## We maximize rank for points starting in top-K,  minimize rank for points starting outside top-K
        if (baseline_rank[pt_idx] <= top_K):
            model.setObjective(obj, GRB.MAXIMIZE)
        else:
            model.setObjective(obj, GRB.MINIMIZE)

        ## Constraints
        """Constraint #1 and #2 """
        for bb in range(samples):
            if (bb != pt_idx):
                # I_{ij} + I_{ji} <= 1 \forall j
                model.addConstr(I[bb, 0] + I[bb, 1] == 1.0, name="I_{}".format(bb))

                # Big-M constraint
                model.addConstr(sum((Xf[bb, d] - Xf[pt_idx, d]) * beta[d]
                                    for d in range(dim + 1)) - M_constant * I[bb, 0] <= 0.0,
                                name="M_{}".format(bb))

                model.addConstr(sum((Xf[pt_idx, d] - Xf[bb, d]) * beta[d]
                                    for d in range(dim + 1)) - M_constant * I[bb, 1] <= 0.0,
                                name="M2_{}".format(bb))

        """Constraint #3 """
        # beta - baseline_coefs <= max_epsilon
        # beta_minus_beta0 = [beta[d] - baseline_coefs[d] for d in range(dim + 1)]
        beta_minus_beta0_sum = sum((beta[d] - baseline_coefs[d]) ** 2 for d in range(dim + 1))  # or multiply by itself
        # norm_beta_minus_beta0 = beta_minus_beta0_sum * beta_minus_beta0_sum
        model.addConstr( sum( (beta[d] - baseline_coefs[d]) ** 2 for d in range(dim + 1)) <= max_epsilon)

        # Solve optimization problem
        model.update()
        model.params.timelimit = 100
        model.setParam('OutputFlag', 0)  ## test out DEBUG
        model.setParam('LogToConsole', 0)  ## test out DEBUG
        model.params.mipgap = 0.001

        model.optimize()

        # print_log('Rank of point {}: {} (baseline), {} (constrained)'.format(pt_idx, baseline_rank[pt_idx], model.ObjVal))

        if (model.SolCount > 0):
            print_log('solution exists')
            all_models.append(model)
            # save the model
            # save the RSS of the model to be compared
            model_coefs = [v.X for v in model.getVars()[0:dim + 1]]
            this_rss = compute_RSS(X_array=Xf, y_array=y, beta_array=np.array(model_coefs))
            all_RSS.append(this_rss)
            all_modelCoefs.append(model_coefs)

            results_output = dict(settings)
            results_output.update({
                'objectives_arr': np.array(all_RSS),
                'models_coefs': all_modelCoefs,
                'obj_baseline': baseline_obj,
                'model_baseline': baseline_coefs,
                'output_filename': str(json_file),
                'data_name': data_name,
                'data': subset_data_df
            })

            # save to disk
            with open(results_file, 'wb') as outfile:
                dill.dump(results_output, outfile, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

            json_output = dict(settings)
            json_output.update({
                'objectives_arr': all_RSS,
                'models_coefs': all_modelCoefs,
                'obj_baseline': baseline_obj,
                'model_baseline': baseline_coefs.tolist(),
                'output_filename': str(json_file),
                'data_name': data_name,
                'data': subset_data_df.values.tolist()
            })

            json.dump(json_output, open(json_file, 'w'))

end_time = time.monotonic()
runtime = str(timedelta(seconds=end_time - start_time))
print_log('\n TOTAL RUNTIME: {}'.format(runtime))

'''Plot Ambiguity with results'''
models_RSS = np.array(all_RSS)

n_competing = []
ambiguity = []

for eps in range(epsilon.shape[0]):
    # for this value of epsilon, are there any competing models according to the array of model objectives?
    any_competing = np.less_equal(models_RSS, obj_thresholds[eps])
    # print('number of training points: {}'.format(samples))
    print('epsilon: {:.0%}, num models competing: {}'.format(epsilon[eps], np.sum(any_competing)))
    n_competing.append(any_competing)

    # now what is the model idex of those competing models
    competing_idx = np.where(np.less_equal(models_RSS, obj_thresholds[eps]))[0]

    # for model in np.array(all_models)[competing_idx]:
    for this_coefs in np.array(all_modelCoefs)[competing_idx]:
        # now lets check how many points are impacted over this set of competing models
        # compute indicator matrix then use I and coefs to compute rank
        # this_coefs = np.array([v.X for v in model.getVars()[0:dim + 1]])
        this_I = indicator_b_greater_a(X_array=Xf,
                                       coefs_array=this_coefs,
                                       n_samples=samples)

        this_rank = np.array([manual_rank(I=this_I, idx=p) for p in range(samples)])

        prop_all_ranks, prop_top_ranks = compute_ambiguity(baseline=baseline_rank, competing=this_rank, K=top_K)

        this_ambiguity = {'epsilon': epsilon[eps],
                          'proportion_all_ranks': prop_all_ranks,
                          'proportion_top_ranks': prop_top_ranks}
        ambiguity.append(this_ambiguity)

if (len(ambiguity) >0):
    ambiguity_df = pd.DataFrame(ambiguity).groupby('epsilon').max()

    results_output.update({
        'runtime': runtime,
        'ambiguity_df': ambiguity_df
    })

    # save to disk
    with open(results_file, 'wb') as outfile:
        dill.dump(results_output, outfile, protocol=dill.HIGHEST_PROTOCOL, recurse=True)