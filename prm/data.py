import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from copy import copy
import dill
from dataclasses import dataclass, field
from typing import List
from .cross_validation import validate_cvindices
RAW_DATA_FILE_SUFFIX = '.csv'

def load_data_from_csv(file_name, random_seed):
    """

    Parameters
    ----------
    file_name
    random_seed

    Returns
    -------

    """
    raw_data = pd.read_csv(file_name)
    XY = raw_data.values
    d = XY.shape[1] - 1
    X = XY[:, 0:d]
    Y = XY[:, d]

    # convert y into 0,1
    bin_y = np.array(Y == 1)
    Y = bin_y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = random_seed)
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return data


class BinaryClassificationDataset(object):
    """class to represent/manipulate datasets for a binary classification task"""

    def __init__(self, X, y, **kwargs):
        """

        :param X:
        :param y:
        :param kwargs:
        """

        # complete dataset
        y = np.array(y).flatten().astype(np.int_)
        y[y <= 0] = -1
        self._full = _BinaryClassificationSample(parent = self, X = X, y = y)

        # variable names
        self._names = _ClassificationVariableNames(parent = self, y = kwargs.get('y_name', 'y'), X = kwargs.get('X_names', ['x%02d' % j for j in range(1, self.d + 1)]))

        # cvindices
        self._cvindices = kwargs.get('cvindices')

        # indicator to check if we have split into train, test, splits
        self.reset()

    #### state ###
    def __check_rep__(self):

        # check data
        assert self._full.__check_rep__()

        # check names
        assert self.names.__check_rep__()

        # if there is a split, then double check
        assert self.n == self.training.n + self.validation.n + self.test.n
        if self._cvindices is not None:
            validate_cvindices(self._cvindices)

        if self._fold_id is not None:
            assert self._cvindices is not None

        if hasattr(self, 'training'):
            assert self.training.__check_rep__()

        if hasattr(self, 'validation'):
            assert self.validation.__check_rep__()

        if hasattr(self, 'test'):
            assert self.test.__check_rep__()

        return True

    def reset(self):
        """
        initialize data object to a state before CV
        :return:
        """
        self._fold_id = None
        self._fold_number_range = []
        self._fold_num_test = 0
        self._fold_num_range = 0
        self._is_split = False
        self.training = self._full
        self.validation = self._full.filter(indices = np.zeros(self.n, dtype = np.bool_))
        self.test = self._full.filter(indices = np.zeros(self.n, dtype = np.bool_))
        assert self.__check_rep__()

    #### built-ins ####
    def __eq__(self, other):
        return (self._full == other._full) and \
               all(np.array_equal(self.cvindices[k], other.cvindices[k]) for k in self.cvindices.keys())

    def __len__(self):
        return self.n

    def __repr__(self):
        return f'BinaryClassificationDataset<n={self.n}, d={self.d}>'

    def __copy__(self):
        ds_copy = BinaryClassificationDataset(X = self.X, y = self.y, X_names = self.names.X, y_name = self.names.y, cvindices = self.cvindices)
        return ds_copy

    ### read, save, load ####
    @staticmethod
    def read_csv(data_file, **kwargs):
        """
        loads raw data from CSV
        :param data_file: Path to the data_file
        :param helper_file: Path to the helper_file or None.
        :param weights_file: Path to the weights_file or None.
        If helper_file or weights_file are 'None', the function will search for these files on disk using the name of the dataset file
        :return:
        """

        # extract common file header from dataset file
        file_header = str(data_file).rsplit('_data%s' % RAW_DATA_FILE_SUFFIX)[0]

        # convert file names into path objects with the correct extension
        files = {
            'data': '{}_data'.format(file_header),
            }
        files = {k: Path(v).with_suffix(RAW_DATA_FILE_SUFFIX) for k, v in files.items()}

        # read data file
        df = pd.read_csv(files['data'], sep = ',')

        # read helper file
        # indices
        colnames = {
            'y': df.columns[0],
            'X': df.columns[1:].tolist(),
            }

        # initialize dataset
        data = BinaryClassificationDataset(
                X = df[colnames['X']].values.astype(np.float_),
                y = df[colnames['y']].values.astype(np.float_),
                X_names = colnames['X'],
                y_name = colnames['y']
                )

        return data

    def save(self, file, overwrite = False, check_save = True):
        """
        saves object to disk
        :param file:
        :param overwrite:
        :param check_save:
        :return:
        """

        f = Path(file)
        if f.is_file() and overwrite is False:
            raise IOError('file %s already exists on disk' % f)

        # check data integrity
        assert self.__check_rep__()

        # save a copy to disk
        data = copy(self)
        data.reset()
        with open(f, 'wb') as outfile:
            dill.dump({'data': data}, outfile, protocol = dill.HIGHEST_PROTOCOL)

        if check_save:
            loaded_data = self.load(file = f)
            assert data == loaded_data

        return f

    @staticmethod
    def load(file):
        """
        loads processed data file from disk
        :param file: path of the processed data file
        :return: data and cvindices
        """
        f = Path(file)
        if not f.is_file():
            raise IOError('file: %s not found' % f)

        with open(f, 'rb') as infile:
            file_contents = dill.load(infile)
            assert 'data' in file_contents, 'could not find `data` variable in pickle file contents'
            assert file_contents['data'].__check_rep__(), 'loaded `data` has been corrupted'

        data = file_contents['data']
        return data

    #### variable names ####
    @property
    def names(self):
        """ pointer to names of X, G, T, E"""
        return self._names

    #### parent level properties ####
    @property
    def n(self):
        """ number of examples in full dataset"""
        return self._full.n

    @property
    def d(self):
        """ number of features in full dataset"""
        return self._full.d

    @property
    def df(self):
        return self._full.df

    @property
    def X(self):
        """ feature matrix """
        return self._full.X

    @property
    def y(self):
        """ label vector"""
        return self._full.y

    @property
    def labels(self):
        return self._full.labels

    #### cross validation ####
    @property
    def cvindices(self):
        return self._cvindices

    @cvindices.setter
    def cvindices(self, cvindices):
        self._cvindices = validate_cvindices(cvindices)

    @property
    def fold_id(self):
        return self._fold_id

    @fold_id.setter
    def fold_id(self, fold_id):
        assert self._cvindices is not None, 'cannot set fold_id on a BinaryClassificationDataset without cvindices'
        assert isinstance(fold_id, str), 'invalid fold_id'
        assert fold_id in self.cvindices, 'did not find fold_id in cvindices'
        self._fold_id = str(fold_id)
        self._fold_number_range = np.unique(self.folds).tolist()

    @property
    def folds(self):
        return self._cvindices.get(self._fold_id)

    @property
    def fold_number_range(self):
        return self._fold_number_range

    @property
    def fold_num_validation(self):
        return self._fold_num_validation

    @property
    def fold_num_test(self):
        return self._fold_num_test

    def split(self, fold_id, fold_num_validation = None, fold_num_test = None):
        """
        :param fold_id:
        :param fold_num_validation: fold to use as a validation set
        :param fold_num_test: fold to use as a hold-out test set
        :return:
        """
        # set fold_id if it has been passed
        if fold_id is not None:
            self.fold_id = fold_id

        assert self.fold_id is not None

        # parse fold numbers
        if fold_num_validation is not None and fold_num_test is not None:
            assert int(fold_num_test) != int(fold_num_validation)

        training_folds = []
        if fold_num_validation is not None:
            fold_num_validation = int(fold_num_validation)
            assert fold_num_validation in self.fold_number_range
            self._fold_num_validation = fold_num_validation
            training_folds.append(fold_num_validation)

        if fold_num_test is not None:
            fold_num_test = int(fold_num_test)
            assert fold_num_test in self.fold_number_range
            self._fold_num_test = fold_num_test
            training_folds.append(fold_num_test)

        # determine indices for each subsample
        if len(training_folds) > 0:
            training_idx = np.isin(self.folds, training_folds, invert = True)
        else:
            training_idx = np.ones(self.n, dtype = np.bool_)

        if fold_num_validation is not None:
            validation_idx = np.isin(self.folds, self.fold_num_validation)
        else:
            validation_idx = np.zeros(self.n, dtype = np.bool_)

        if fold_num_test is not None:
            test_idx = np.isin(self.folds, self.fold_num_test)
        else:
            test_idx = np.zeros(self.n, dtype = np.bool_)

        # create subsamples
        self.training = self._full.filter(indices = training_idx)
        self.validation = self._full.filter(indices = validation_idx)
        self.test = self._full.filter(indices = test_idx)
        return


@dataclass
class _BinaryClassificationSample:
    """class to store and manipulate a subsample of points in a survival dataset"""

    parent: object
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray = None
    _labels: tuple = (-1, 1)

    def __post_init__(self):
        self.X = np.atleast_2d(np.array(self.X, np.float))
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]

        # encode labels
        y = np.array(self.y, dtype = np.float).flatten()
        if self.n > 0:
            negative_label = np.unique(y)[0]
            neg_idx = np.equal(y, negative_label)
            y[neg_idx] = self.labels[0]
            y[~neg_idx] = self.labels[1]

        if self.indices is None:
            self.indices = np.ones(self.n)
        else:
            self.indices = self.indices.flatten()
        assert self.__check_rep__()

    #### built-in #####
    def __len__(self):
        return self.n

    def __eq__(self, other):
        chk = isinstance(other, _BinaryClassificationSample) and np.array_equal(self.y, other.y) and np.array_equal(self.X, other.X)
        return chk

    def __check_rep__(self):
        """returns True is object satisfies representation invariants"""
        assert isinstance(self.X, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        assert self.n == len(self.y)
        assert np.sum(self.indices) == self.n
        assert np.isfinite(self.X).all()
        assert np.isin(self.y, self.labels).all(), 'y values must be stored as {}'.format(self.labels)
        return True

    #### properties #####
    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values):

        assert len(values) == 2
        assert values[0] < values[1]
        assert isinstance(values, (np.ndarray, list, tuple))
        self._labels = tuple(np.array(values, dtype = np.int))

        # change y encoding using new labels
        if self.n > 0:
            y = np.array(self.y, dtype = np.float).flatten()
            neg_idx = np.equal(y, self._labels[0])
            y[neg_idx] = self._labels[0]
            y[~neg_idx] = self._labels[1]
            self.y = y

    @property
    def df(self):
        """
        generates pandas data.frame with T, E, G, X,
        :param names:
        :return:
        """
        df = pd.DataFrame(self.X, columns = self.parent.names.X)
        df.insert(column = self.parent.names.y, value = self.y, loc = 0)
        return df

    #### method #####
    def filter(self, indices):
        """filters samples based on indices"""
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()
        return _BinaryClassificationSample(parent = self.parent, X = self.X[indices], y = self.y[indices], indices = indices)

@dataclass
class _ClassificationVariableNames:
    """class to represent the names of the features, censoring indicator, and targets in a classification dataset"""
    parent: object
    X: List[str] = field(repr = True)
    y: str = field(repr = True, default = 'y')

    def __post_init__(self):
        assert self.__check_rep__()

    @staticmethod
    def check_name_str(s):
        """check variable name"""
        return isinstance(s, str) and len(s.strip()) > 0

    def __check_rep__(self):
        """returns True is object satisfies representation invariants"""
        assert isinstance(self.X, list) and all([self.check_name_str(n) for n in self.X]), 'X must be a list of strings'
        assert len(self.X) == len(set(self.X)), 'X must be a list of unique strings'
        assert self.check_name_str(self.y), 'y must be at least 1 character long'
        return True
