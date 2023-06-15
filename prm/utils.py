import sys
import time
from pathlib import Path
import warnings
import re
import datetime
import numpy as np

### general functions
def convert_Xy_to_Z(X, y, add_intercept = True):
    """
    converts classification dataset (X, y) into a "target matrix" Z = X * y
    :param X: feature matrix with n rows x d columns
    :param y: label vector of length n
    :param add_intercept: if True, will prepend feature matrix with a column of 1s to represent the itnercept
    :return: Z - target matrix with n rows x d columns where z[i,j] = x[i,j] * y[i]
    """

    # check X
    assert X.ndim == 2, 'X should be a matrix'
    n, d = X.shape
    assert n > 0, 'X should have at least 1 row'
    assert d > 0, 'X should have at least 1 column'
    assert np.isfinite(X).all(), 'components of X should be finite'

    # check y
    y = np.array(y).flatten()
    assert len(y) == n, 'y should have length {}'.format(n)
    y[y == 0] = -1
    assert np.isin(y, [-1, 1]).all()

    if add_intercept:
        X = np.insert(X, obj = 0, values = np.ones(n), axis = 1)

    # target matrix
    Z = y[:, np.newaxis] * X

    return Z

def log_loss_value(Z, w):
    """
    computes the logistic loss of a coefficient vector w
    see also: http://stackoverflow.com/questions/20085768/

    Parameters
    ----------
    Z = Xf * y  numpy.array containing data with shape = (n_rows, n_cols)
    w           numpy.array of coefficients with shape = (n_cols,)

    Returns
    -------
    loss_value  scalar = 1/n_rows * sum(log( 1 .+ exp(-Z*w))
    """
    scores = Z.dot(w)
    pos_idx = scores > 0
    loss_value = np.empty_like(scores)
    loss_value[pos_idx] = np.log1p(np.exp(-scores[pos_idx]))
    loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(np.exp(scores[~pos_idx]))
    return loss_value.mean()

def predict_prob(X, w, add_intercept = True):

    n, d = X.shape
    if add_intercept:
        assert d == len(w) - 1
        intercept_idx = 0
        coefficient_idx = np.arange(1, d + 1)
        scores = X.dot(w[coefficient_idx]) + w[intercept_idx]
    else:
        scores = X.dot(w)

    probs = np.zeros_like(scores)
    pos_idx = np.greater_equal(scores, 0)
    neg_idx = np.logical_not(pos_idx)
    probs[pos_idx] = 1.0 / (1.0 + np.exp(-scores[pos_idx]))
    probs[neg_idx] = np.exp(scores[neg_idx]) / (1.0 + np.exp(scores[neg_idx]))
    return probs


### printing
_LOG_TIME_FORMAT = "%m/%d/%y @ %I:%M %p"

def print_log(msg, print_flag = True):
    if print_flag:
        if isinstance(msg, str):
            print_str = '%s | %s' % (time.strftime(_LOG_TIME_FORMAT, time.localtime()), msg)
        else:
            print_str = '%s | %r' % (time.strftime(_LOG_TIME_FORMAT, time.localtime()), msg)
        print(print_str)
        sys.stdout.flush()


#### splitting dataset into parts
TRIVIAL_PART_ID = "P01N01"
PART_PATTERN = "^P[0-9]{2}N[0-9]{2}$"
PART_ID_HANDLE = lambda p, n: 'P{:02}N{:02}'.format(p, n)
PART_PARSER = re.compile(PART_PATTERN)

PART_ID_PATTERN = "P[0-9]{2}N[0-9]{2}"
PART_ID_PARSER = re.compile("P[0-9]{2}N[0-9]{2}")

def filter_indices_to_part(u_idxs, part_id = TRIVIAL_PART_ID):
    """
    Given a list of indices, slices to a selection of the indices based on the part_id.
    :param u_idxs: The indices of unique points in a dataset.
    :param part_id: The part_id used to determine the slice. Specifies number of parts and the part to select.
    :return: The unique indices sliced to the specified part.
    """

    # extract info from params
    n_unique = len(u_idxs)
    part, n_parts = parse_part_id(part_id)

    # warn if splitting into more parts than can be made nonempty
    if n_parts > len(u_idxs):
        message = 'Splitting {} indices into {} parts. Some parts will be empty.'.format(len(u_idxs), n_parts)
        warnings.warn(message)

    # get indices to split on
    part_idxs = np.linspace(0, n_unique, n_parts + 1).astype(int)
    start = part_idxs[part - 1]
    end = part_idxs[part]

    return u_idxs[start:end]

def parse_part_id(part_id):
    """
    Given a part_id, extracts the number of parts and which part is selected
    :param part_id: The part_id to extract. Should be of the format specified at the top of this file.
    :return: The part selected and the number of parts.
    """
    # check that part_id is of the correct format
    assert bool(PART_PARSER.match(part_id)), 'invalid part_id.'

    # extract from the part_id
    str_nums = re.findall('\d+', part_id)
    part, n_parts = tuple(map(int, str_nums))

    # check that the information extracted is consistent
    assert part <= n_parts
    return part, n_parts

def get_part_id_helper(file_names):
    """
    :param file_names: list of file_names or a single file_name
    :return: dictionary containing keys for each partition, each partition containing a list of files matching the partition
    """

    if isinstance(file_names, (str, Path)):
        file_names = [file_names]

    # filter file names to files that exist on disk
    file_names = [Path(f) for f in file_names if f.exists()]
    file_names = list(set(file_names))

    # extra all part ids
    part_ids = [PART_ID_PARSER.findall(f.name) for f in file_names]
    part_ids = [p[0] for p in part_ids if len(p) > 0]

    # extract distinct counts
    part_counts = [re.findall('N[0-9]{2}', p) for p in part_ids]
    part_counts = [p[0][1:] for p in part_counts]
    distinct_part_counts = set([int(p) for p in part_counts])

    out = {}
    for n in distinct_part_counts:

        part_pattern = 'P[0-9]{2}N%02d' % n
        expected_parts = set(range(1, n + 1))

        matched_names = [re.search(part_pattern, f.name) for f in file_names]
        matched_names = [f for f in matched_names if f is not None]
        matched_parts = [f.group(0) for f in matched_names]
        matched_names = [f.string for f in matched_names]
        matched_files = [f for f in file_names if f.name in matched_names]
        modification_times = [datetime.datetime.fromtimestamp(f.stat().st_mtime) for f in matched_files]

        matched_parts = [re.search('P[0-9]{2,}', p) for p in matched_parts]
        matched_parts = [int(p.group(0)[1:]) for p in matched_parts]
        matched_parts = set(matched_parts)

        missing_parts = expected_parts.difference(matched_parts)
        missing_part_ids = ['P%02dN%02d' % (n, p) for p in missing_parts]

        out[n] = {
            'complete': len(missing_parts) == 0,
            'last_modification_time': max(modification_times) if len(modification_times) else None,
            'matched_parts': matched_parts,
            'matched_files': matched_files,
            'modification_times': modification_times,
            'missing_parts': missing_part_ids,
            }

    return out


is_geq_or_close = lambda a, b: np.logical_or(np.greater_equal(a, b), np.isclose(a, b)).astype(int)
is_leq_or_close = lambda a, b: np.logical_or(np.less_equal(a, b), np.isclose(a, b)).astype(int)


def compute_log_loss(X, y, w, add_intercept = True):
    """
    :param X: feature matrix (n rows x d columns)
    :param y: label vector with n elements
    :param w: coefficient vector with (d + 1) elements
    :return: logistic loss of feature matrix over coefficient vector
    """
    Z = convert_Xy_to_Z(X, y, add_intercept = add_intercept)
    return log_loss_value(Z, w)


def compute_discrepancy_indicators(X, y, w, p0, delta, add_intercept = True):
    """
    :param p0: baseline probabilities (n rows)
    :param y: label vector with n elements (n rows)
    :param w: coefficient vector with (d + 1) elements
    :return: logistic loss of feature matrix over coefficient vector
    """
    assert X.shape[0] == len(y)
    assert X.shape[0] == len(p0)
    assert np.isin(y, (-1, 1)).all()
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.isfinite(w).all()
    assert np.isfinite(p0).all()
    assert np.isfinite(delta)
    p = predict_prob(X = X, w = w, add_intercept = add_intercept)
    dp = np.abs(p0 - p)
    return is_geq_or_close(dp, delta)