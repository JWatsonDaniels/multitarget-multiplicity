'''This script loads the results file from the cluster experiments and produces predictive multiplicity
analysis and plots to be saved for review.'''

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from itertools import product
from scipy import stats
import scipy as sp
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import dill
from os.path import exists
from prm.paths import get_results_file_rank
from prm.paths import get_results_file_rank, get_json_file_rank_AltMIP, get_results_file_rank_AltMIP
import json
import pickle5 as pickle

sns.set()

# Define the color maps for plots
# color_map = plt.cm.get_cmap('RdYlBu')
# color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])

# Plot settings
sns.set_context("paper")
# Set the font to be serif, rather than sans
sns.set(font='serif')
# Make the background white, and specify the specific font family
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
sns.set_style("ticks", {"xtick.major.size": 8})


def compute_ambiguity(baseline, competing, K):
    '''
    Computes top-K ambiguity
    :return:
    '''
    # # how many points have ranking different from baseline
    num_bot_flips = 0
    num_top_flips = 0
    baseline_of_flipped = []
    n = len(baseline)
    for r in range(n):
        if (baseline[r] <= top_K):
            # count if this_rank flips
            if (competing[r] > K):
                num_top_flips += 1
                baseline_of_flipped.append(baseline[r])
        else:
            if (competing[r] <= K):
                num_bot_flips += 1
                baseline_of_flipped.append(baseline[r])

    prop_all = (num_top_flips + num_bot_flips) / n
    prop_top = num_top_flips / K
    return prop_all, prop_top, baseline_of_flipped


'''load results files and calculate predictive multiplicity'''
settings = {
    'data_name': 'dissecting_bias_dataset_three_outcomes_limited_features_train_and_holdout',
    'n_samples': 1000,
    'big_m': 10.0,
    'top_K': 40,
    'w_max': 100,
    'target_idx': 0,
    'random_seed': 109
    }


# results_file = get_results_file_rank_MIP(**settings)
# json_file = get_json_file_rank_MIP(**settings)
data_name = settings['data_name']

# Set 'data_name' to reflect with multi-target variable is being run
target_idx = settings['target_idx']
target_names = np.array(['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t'])
settings['data_name'] = 'HEALTH2_' + target_names[target_idx]
results_file = get_results_file_rank_AltMIP(**settings)
json_file = get_json_file_rank_AltMIP(**settings)
data_name = target_names[target_idx]

if exists(results_file):
    # try:
    #     with open(results_file, 'rb') as infile:
    #         results = dill.load(infile)
    #     data = pd.DataFrame(results['data'])
    try:
        print('trying pickle5')
        with open(results_file, 'rb') as infile:
            results = pickle.load(infile)
    except:
        print('loading json')
        # Opening JSON file
        f = open(json_file)
        # returns JSON object as
        # a dictionary
        results = json.load(f)
        print('json has been loaded')


data = pd.DataFrame(results['data'])
# Extract results objects
samples = results['n_samples']
all_objs = np.array(results['objectives_arr'])
all_models = results['models_coefs']
# models_Vars = results['models_Vars']
baseline_obj = results['obj_baseline']
baseline_coefs = np.array(results['model_baseline'])
top_K = results['top_K']
data_name = results['data_name']

models_RSS = np.array(all_objs)
## are any of the models within 1% MSE of the baseline model?
# epsilon = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
epsilon = np.array([0.005, 0.01, 0.015, 0.02, 0.025])
obj_thresholds = (1 + epsilon) * baseline_obj

## use actual dataframe for Xf now
X = data.drop(['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t'], axis = 1)

'''Train the prediction model for each target individually'''
X = X.to_numpy()
intercept_idx = 0
Xf = np.insert(X, intercept_idx, 1.0, axis=1)

# orthonormalize the features to calculate the unflippable points but use the normal Xf for the MIP
Xf = sp.linalg.orth(Xf)
samples, dim = Xf.shape
dim = dim - 1

# for target_idx in range(n_targets):
y_multi = data[['log_cost_t', 'log_cost_avoidable_t', 'gagne_sum_t']]
y = np.array(y_multi[y_multi.columns[target_idx]])


rank_method = "ordinal"
baseline_coefs = results['model_baseline']
## Baseline rankings
Y_pred_baseline = Xf @ baseline_coefs
baseline_rank = stats.rankdata(-1.0 * Y_pred_baseline, method = rank_method)

n_competing = []
ambiguity = []


Y_pred_competing = []
all_ranks = []

eps_1_coefs = []
eps_1_baseline_ranks = []

for eps in range(epsilon.shape[0]):
    # for this value of epsilon, are there any competing models according to the array of model objectives?
    any_competing = np.less_equal(models_RSS, obj_thresholds[eps])
    # print('number of training points: {}'.format(samples))
    print('epsilon: {:.0%}, num models competing: {}'.format(epsilon[eps], np.sum(any_competing) ))
    n_competing.append(any_competing)

    # now what is the model idex of those competing models
    competing_idx = np.where(np.less_equal(models_RSS, obj_thresholds[eps]))[0]

    # for model in np.array(all_models)[competing_idx]:
    for this_coefs in np.array(all_models)[competing_idx]:
        # now lets check how many points are impacted over this set of competing models
        # compute indicator matrix then use I and coefs to compute rank
        # this_coefs = np.array([v.X for v in model.getVars()[0:dim + 1]])



        Y_pred = Xf @ this_coefs

        Y_pred_competing.append(Y_pred)
        this_rank = stats.rankdata(-1.0 * Y_pred, method = rank_method)



        prop_all_ranks, prop_top_ranks, baseline_of_flipped = compute_ambiguity(baseline = baseline_rank, competing = this_rank, K = top_K)

        # step through baseline rank of flipped points to see if any are new

        if (epsilon[eps] == 0.01):
            eps_1_coefs.append(this_coefs)
            eps_1_baseline_ranks.extend(baseline_of_flipped)

        this_ambiguity = {'epsilon': epsilon[eps],
                          'proportion_all_ranks': prop_all_ranks,
                          'proportion_top_ranks': prop_top_ranks}
        ambiguity.append(this_ambiguity)

y_hat_df = pd.DataFrame(Y_pred_competing)
ranks_df = pd.DataFrame(all_ranks)

# save coefs to csv to copy into our results file
epsilon_1percent_coefs_df = pd.DataFrame(np.array(eps_1_coefs))
epsilon_1percent_coefs_df.to_csv('1percent_coefs_{}_{}_top{}_nsamples{}.csv'.format(target_names[target_idx], data_name, top_K, samples))

baseline_coefs_df = pd.DataFrame(np.array(baseline_coefs))
baseline_coefs_df.to_csv('baseline_coefs_{}_{}_top{}_nsamples{}.csv'.format(target_names[target_idx], data_name, top_K, samples))

baseline_ranks_df = pd.DataFrame(np.unique(eps_1_baseline_ranks))
baseline_ranks_df.to_csv('baseline_ranks_{}_{}_top{}_nsamples{}.csv'.format(target_names[target_idx], data_name, top_K, samples))


import matplotlib.ticker as mtick
# Plot settings
sns.set_context("paper")
# Set the font to be serif, rather than sans
sns.set(font='serif')
# Make the background white, and specify the specific font family
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
sns.set_style("ticks", {"xtick.major.size": 8})

if len(ambiguity) > 0:
    # ambiguity_df = pd.DataFrame(ambiguity)
    ambiguity_df = pd.DataFrame(ambiguity).groupby('epsilon').max()

    fig1, ax1 = plt.subplots(1, 1, figsize = (8, 5))

    ax1.scatter(epsilon, ambiguity_df.proportion_all_ranks, label = "all points")
    ax1.plot(epsilon, ambiguity_df.proportion_all_ranks)

    ax1.scatter(epsilon, ambiguity_df.proportion_top_ranks,  label = "top-K points")
    ax1.plot(epsilon, ambiguity_df.proportion_top_ranks)

    plt.xticks(epsilon, epsilon.astype(str).tolist())
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax1.set_ylim([0.0, 1.05])
    plt.xlabel("epsilon", fontsize = 12)
    plt.ylabel("proportion of conflicting ranks", fontsize = 12)
    plt.title('Ambiguity, top-K = {}'.format(top_K), fontsize = 16)

    for i,j in zip(epsilon,ambiguity_df.proportion_all_ranks):
        ax1.annotate(str(np.round(j, decimals=2) ),xy=(i-0.0007,j-0.08))
    for i,j in zip(epsilon,ambiguity_df.proportion_top_ranks):
        ax1.annotate(str(j),xy=(i,j+0.02))
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('ambiguity_{}_{}_top{}_nsamples{}_wmax_{}.png'.format(target_names[target_idx], data_name, top_K, samples, int(settings['w_max']) ))




ambiguity_df.to_csv('results/ambiguity_{}_{}_top{}_nsamples{}.csv'.format(target_names[target_idx], data_name, top_K, samples))
