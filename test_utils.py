import numpy as np
import pandas as pd


def generate_intervals(min_range, max_range, interval_size, upper_limit=-1):
    intervals = np.arange(0, max_range + interval_size, interval_size, dtype=int)  # interval_size spaced seq. of ints
    if upper_limit > 0:
        upper_limit_mask = ~(intervals > upper_limit)
        upper_limit_mask[-1] = True  # setting maximum length as final limit
        intervals = intervals[upper_limit_mask]
    return intervals


def get_session_count_slices(data, interval_size, upper_limit=-1, session_id='session_id', user_id='user_id',
                             item_id='item_id',
                             timestamp='created_at'):
    data_test = data[data['in_eval'] == True].copy()
    unique_user_sessions_id = data_test.groupby([user_id, session_id]).count().reset_index()
    sessions_per_user = data.groupby([user_id, session_id]).count().reset_index().groupby(user_id).count()
    max_range, min_range = sessions_per_user[session_id].max(), sessions_per_user[session_id].min()
    intervals = generate_intervals(min_range, max_range, interval_size, upper_limit)
    user_slices = []
    session_slices = []
    limits = []
    for idx, value in enumerate(intervals[:-1]):
        min_limit, max_limit = intervals[idx], intervals[idx + 1]
        user_slice = sessions_per_user[
            (sessions_per_user[session_id] > min_limit) & (sessions_per_user[session_id] <= max_limit)].index
        session_slice = unique_user_sessions_id[unique_user_sessions_id[user_id].isin(user_slice.values)][session_id]
        user_slices.append(user_slice)
        session_slices.append(session_slice)
        limits.append([min_limit, max_limit])
    return user_slices, session_slices, limits


def get_users_slices(data, interval_size, upper_limit=-1, session_id='session_id', user_id='user_id', item_id='item_id',
                     timestamp='created_at'):
    mean_session_numbers = data.groupby([user_id, session_id]).count().reset_index().groupby(user_id).mean()
    max_range, min_range = mean_session_numbers[item_id].max(), mean_session_numbers[item_id].min()
    intervals = generate_intervals(min_range, int(round(max_range)), interval_size, upper_limit)
    user_slices = []
    limits = []
    for idx, value in enumerate(intervals[:-1]):
        min_limit, max_limit = intervals[idx], intervals[idx + 1]
        user_slice = mean_session_numbers[
            (mean_session_numbers[item_id] > min_limit) & (mean_session_numbers[item_id] <= max_limit)].index
        user_slices.append(user_slice)
        limits.append([min_limit, max_limit])
    return user_slices, limits


def get_session_slices(data, eval_data, interval_size, upper_limit=-1, session_id='session_id', user_id='user_id',
                       item_id='item_id', timestamp='created_at'):
    # data = data[data['in_eval'] == True].copy()
    data = eval_data
    mean_session_length = data.groupby(session_id).count()
    max_range, min_range = mean_session_length[item_id].max(), mean_session_length[item_id].min()
    intervals = generate_intervals(min_range, max_range, interval_size, upper_limit)
    session_slices = []
    limits = []
    for idx, value in enumerate(intervals[:-1]):
        min_limit, max_limit = intervals[idx], intervals[idx + 1]
        session_slice = mean_session_length[(mean_session_length[item_id] > min_limit) &
                                            (mean_session_length[item_id] <= max_limit)].index
        limits.append([min_limit, max_limit])
        session_slices.append(session_slice)

    return session_slices, limits


def get_user_history_slices(data, interval_size, upper_limit=-1, multiplier=100, session_id='session_id',
                            user_id='user_id',
                            item_id='item_id', timestamp='created_at'):
    data_test = data[data['in_eval'] == True].copy()
    unique_user_sessions_id = data_test.groupby([user_id, session_id]).count().reset_index()
    user_history_length = data.groupby(user_id).count()
    max_range, min_range = user_history_length[item_id].max(), user_history_length[item_id].min()
    intervals = multiplier * generate_intervals(min_range, max_range, interval_size, upper_limit)
    user_slices = []
    session_slices = []
    limits = []

    for idx, value in enumerate(intervals[:-1]):
        min_limit, max_limit = intervals[idx], intervals[idx + 1]
        user_slice = user_history_length[
            (user_history_length[item_id] > min_limit) & (user_history_length[item_id] <= max_limit)].index
        session_slice = unique_user_sessions_id[unique_user_sessions_id[user_id].isin(user_slice.values)][session_id]
        user_slices.append(user_slice)
        session_slices.append(session_slice)
        limits.append([min_limit, max_limit])

    return user_slices, session_slices, limits


def generate_test_slices(eval_slice_mode, eval_data, test_data, interval_size, upper_limit):
    """

    @param eval_slice_mode: 'session' groups by session length,'user' groups by mean number o session length,
    'history' groups by user total number of interactions, 'session-count' groups by the total number of session of
    each user
    @param eval_data: DataFrame  corresponding to test_results
    @param test_data: Dataset corresponding to test set
    @param interval_size: size of segment for each partition of sessions
    @param upper_limit: groups partitions that are larger than this value as one slice
    @return: 'test_slice' array of arrays with values in each interval, 'test_limits' array of arrays containing value
    limits for each interval

    """
    if eval_slice_mode == 'user':
        test_slices, test_limits = get_users_slices(test_data.df, interval_size, upper_limit)
    elif eval_slice_mode == 'session':
        test_slices, test_limits = get_session_slices(test_data.df, eval_data, interval_size, upper_limit)
    elif eval_slice_mode == 'history':
        _, test_slices, test_limits = get_user_history_slices(test_data.df, interval_size, upper_limit)
    elif eval_slice_mode == 'session-count':
        _, test_slices, test_limits = get_session_count_slices(test_data.df, interval_size, upper_limit)
    else:
        raise RuntimeError('Unknown eval_slice_mode: {}'.format(eval_slice_mode))

    return test_slices, test_limits


def multiple_mode_test(slices_results, test_data, eval_data, interval_size, upper_limit, results_path, dataset_name,
                       session_key='session_id', slice_modes=None):
    if slice_modes is None:
        slice_modes = ['user', 'session', 'history', 'session-count']
    results = {}
    for slice_mode in slice_modes:
        print('Evaluation mode: {}'.format(slice_mode))
        test_slices, test_limits = generate_test_slices(slice_mode, eval_data, test_data, interval_size, upper_limit)
        #print([len(test_slices), len(test_limits)])
        create_results_file(slices_results, slice_mode, test_slices, test_limits,
                                                               eval_data, interval_size,
                                                               results_path, dataset_name, session_key)
    


def create_full_results_file_from_preds(predictions_df, eval_cutoffs=[5, 10, 20]):
    slices_results = pd.DataFrame([])
    full_results = {}
    full_results['Loss'] = predictions_df['loss'].mean()
    for idx, k in enumerate(eval_cutoffs):
        full_results['R@' + str(k)] = predictions_df['R@' + str(k)].mean()
        full_results['MRR@' + str(k)] = predictions_df['MRR@' + str(k)].mean()
    full_results['time'] = '00:00:00'
    full_results['slice'] = 'full'
    full_results['slice_ids_count'] = int(predictions_df.shape[0])
    slices_results = slices_results.append(full_results, ignore_index=True)
    return slices_results


def create_results_file_from_preds(predictions_df, test_data, results_path, dataset_name, eval_cutoffs=[5, 10, 20],
                                   interval_size=5, upper_limit=200, session_key='session_id',slice_modes=None):
    slices_results = pd.DataFrame([])
    full_results = {}
    full_results['Loss'] = predictions_df['loss'].mean()
    for idx, k in enumerate(eval_cutoffs):
        full_results['R@' + str(k)] = predictions_df['R@' + str(k)].mean()
        full_results['MRR@' + str(k)] = predictions_df['MRR@' + str(k)].mean()
    full_results['time'] = '00:00:00'
    full_results['slice'] = 'full'
    full_results['slice_ids_count'] = int(predictions_df.shape[0])
    slices_results = slices_results.append(full_results, ignore_index=True)
    multiple_mode_test(slices_results, test_data, predictions_df, interval_size, upper_limit, results_path,
                       dataset_name, session_key, slice_modes)
    


def create_results_file(slices_results, slice_mode, test_slices, test_limits, eval_data, interval_size, results_path,
                        dataset_name, session_key):
    slice_count = [x.nunique() for x in test_slices]
    for i, test_slice in enumerate(test_slices):
        slice_k = str(test_limits[i][0]) + '-' + str(test_limits[i][1])
        if slice_count[i] > 0:
            slice_eval_data = select_test_sessions(test_slice, eval_data, session_key)
            if slice_eval_data.shape[0] > 0:
                data = ["{:.4f}".format(x) for x in slice_eval_data[eval_data.columns[-6:]].mean().values]
                test_result = {}
                for idx, col in enumerate(eval_data.columns[-6:]):
                    test_result[col] = [data[idx]]

                slice_result = pd.DataFrame.from_dict(test_result)
                slice_result['time'] = ['0:00:00']
                slice_result['Loss'] = ['{:.4f}'.format(0.)]
                slice_result['slice'] = [slice_k]
                slice_result['slice_ids_count'] = [slice_count[i]]
                slices_results = slices_results.append(slice_result[slices_results.columns], ignore_index=True)

        else:
            dummy_slice_result = {col: '{:.4f}'.format(0.) for col in slices_results.columns}
            dummy_slice_result['slice_ids_count'] = slice_count[i]
            dummy_slice_result['time'] = '0:00:00'
            dummy_slice_result['slice'] = slice_k
            slices_results = slices_results.append(dummy_slice_result, ignore_index=True)
            # slices_results.loc[len(slices_results)] = ['{:.4f}'.format(0.)] * 7 + ['0:00:00', slice_k, slice_count[i]]
        results_file_url = results_path + '/' + dataset_name + '-' + slice_mode + '-' + str(
            interval_size) + '-slice-results.csv'
        # print('Saving file: {}'.format(results_file_url))
        slices_results.to_csv(results_file_url, index=False)
    return slices_results, results_file_url


def select_test_sessions(session_slice, data, session_key):
    return data[data[session_key].isin(session_slice)]


def save_eval_data(eval_data, results_path, dataset_name):
    file_name = results_path + '/' + dataset_name + '_predictions.pred'
    print("Saving raw results in: {}".format(file_name))
    eval_data.to_csv(file_name, index=False)
    return file_name


def load_eval_data(results_path, dataset_name):
    file_name = results_path + '/' + dataset_name + '_predictions.pred'
    print("Loading raw results from: {}".format(file_name))
    eval_data = pd.read_csv(file_name)
    return eval_data
