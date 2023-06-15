from pathlib import Path

repo_dir = Path(__file__).absolute().parent.parent
results_dir = repo_dir / "results/"
reports_dir = repo_dir / "reports/"
templates_dir = repo_dir / "templates/"
data_dir = repo_dir / "data/"

# create directories to store results
results_dir.mkdir(exist_ok = True)

# create a directories to store reports
reports_dir.mkdir(exist_ok = True)

def get_repo_dir(**kwargs):
    return repo_dir

def get_processed_data_file(data_name,  **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / '{}_processed.pickle'.format(data_name)
    return f

def get_results_file_rank(data_name, n_samples, w_max, top_K, **kwargs):
    """
    returns file name for ranked output analysis
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0

    f = results_dir / '{}_{}samples_wmax{}_top{}_ranked_output.results'.format(data_name, n_samples, int(w_max), top_K )
    return f

def get_json_file_rank(data_name, n_samples, w_max, top_K, **kwargs):
    """
    returns file name for ranked output analysis
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0

    f = results_dir / '{}_{}samples_wmax{}_top{}_ranked_output.json'.format(data_name, n_samples, int(w_max), top_K )
    return f

def get_json_file_rank_MIP(data_name, n_samples, w_max, top_K, **kwargs):
    """
    returns file name for ranked output analysis
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0

    f = results_dir / '{}_{}samples_wmax{}_top{}_ranked_origMIP.json'.format(data_name, n_samples, int(w_max), top_K )
    return f

def get_results_file_rank_MIP(data_name, n_samples, w_max, top_K, **kwargs):
    """
    returns file name for ranked output analysis
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0

    f = results_dir / '{}_{}samples_wmax{}_top{}_ranked_origMIP.results'.format(data_name, n_samples, int(w_max), top_K )
    return f

def get_json_file_rank_AltMIP(data_name, n_samples, w_max, top_K, **kwargs):
    """
    returns file name for ranked output analysis
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0

    f = results_dir / '{}_{}samples_wmax{}_top{}_ranked_altMIP.json'.format(data_name, n_samples, int(w_max), top_K )
    return f

def get_results_file_rank_AltMIP(data_name, n_samples, w_max, top_K, **kwargs):
    """
    returns file name for ranked output analysis
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0

    f = results_dir / '{}_{}samples_wmax{}_top{}_ranked_altMIP.results'.format(data_name, n_samples, int(w_max), top_K )
    return f

def get_rank_data_csv(data_name,  **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / '{}.csv'.format(data_name)
    return f