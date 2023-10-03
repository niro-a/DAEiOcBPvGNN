def gen_event_outliers(dataframe, obj1, obj2, obj3, start, end, n, k, random_state=None):
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    from faker import Faker
    from tqdm import tqdm

    if random_state:
        np.random.seed(random_state)

    dataframe['event_timestamp'] = pd.to_datetime(dataframe['event_timestamp'])
    dta = dataframe.iloc[:, start:end].values

    # Convert event_timestamp column to datetime
    dataframe['event_timestamp'] = pd.to_datetime(dataframe['event_timestamp'])

    # Make dataframe's datetime objects timezone naive
    dataframe['event_timestamp'] = dataframe['event_timestamp'].dt.tz_localize(None)

    # Select candidates
    n = int((n // 3) * 3)
    k = int(k)

    row_set = set(range(dataframe.shape[0]))

    outlier_idx = np.random.choice(list(row_set), size=n, replace=False)
    np.random.shuffle(outlier_idx)

    candidate_set = row_set.difference(set(outlier_idx))
    candidate_idx = np.random.choice(list(candidate_set), size=n * k)

    outlier_idx_1 = np.random.choice(list(outlier_idx), size=int(len(outlier_idx) * (1 / 3)), replace=False)
    outlier_idx_2 = np.random.choice(list(set(outlier_idx).difference(set(outlier_idx_1))),
                                     size=int(len(outlier_idx) * 1 / 3), replace=False)
    outlier_idx_3 = list(set(outlier_idx).difference(set(outlier_idx_1)).difference(set(outlier_idx_2)))

    # Attributes Outliers
    for i, idx in enumerate(tqdm(outlier_idx_1, desc="Attribute Outliers")):
        cur_candidates = candidate_idx[k * i: k * (i + 1)]

        euclidean_dist = [euclidean_distances(dta[idx].reshape(1, -1), dta[cnd].reshape(1, -1)) for cnd in
                          cur_candidates]
        max_dist_idx = np.argmax(euclidean_dist)
        max_dist_row = list(cur_candidates)[max_dist_idx]

        dta[idx] = dta[max_dist_row]

    df = pd.DataFrame(data=dta, index=dataframe.iloc[:, start:end].index, columns=dataframe.iloc[:, start:end].columns)
    dataframe = dataframe.iloc[:, :start].join(df)

    # Timeshift Outliers
    for idx in tqdm(outlier_idx_2, desc="Timestamp Outliers"):

        ts = dataframe[dataframe[obj1] == dataframe.at[idx, obj1]].event_timestamp.tolist()
        if obj2 is not None:
            ts += dataframe[dataframe[obj2] == dataframe.at[idx, obj2]].event_timestamp.tolist()
        if obj3 is not None:
            ts += dataframe[dataframe[obj3] == dataframe.at[idx, obj3]].event_timestamp.tolist()

        min_ts = min(ts, default=None)
        max_ts = max(ts, default=None)
        if min_ts == None:
            print('warn: no timestamp idx2')
            outlier_idx_2 = np.delete(outlier_idx_2, np.where(outlier_idx_2 == idx))
        elif max_ts == None:
            print('warn: no timestamp idx2')
            outlier_idx_2 = np.delete(outlier_idx_2, np.where(outlier_idx_2 == idx))
        else:
            rng_ts = max_ts - min_ts
            new_ts = Faker().date_time_between(start_date=min_ts - 0.05 * rng_ts, end_date=max_ts + 0.05 * rng_ts)

            dataframe.at[idx, 'event_timestamp'] = new_ts

    y_outlier = np.zeros((len(dataframe), 1))
    y_outlier[outlier_idx_1] = 1
    y_outlier[outlier_idx_2] = 2
    dataframe['y'] = y_outlier.astype('int')

    # Insert Random Event
    num_new_rows = len(outlier_idx_3)
    new_rows = dataframe.loc[outlier_idx_3].copy()  # Create new rows by copying the observations in outlier_idx_3

    new_rows['event_activity'] = [
        'random activity {}'.format(np.random.randint(1, len(dataframe['event_activity'].unique())))
        for _ in range(num_new_rows)]  # Replace 'event_activity' with a random value

    for idx, row in tqdm(new_rows.iterrows(), total=new_rows.shape[0],
                         desc="Random Event Outliers"):  # Change timestamp as in the "Timeshift Outliers" section
        ts = dataframe[dataframe[obj1] == row[obj1]].event_timestamp.tolist()
        if obj2 is not None:
            ts += dataframe[dataframe[obj2] == row[obj2]].event_timestamp.tolist()
        if obj3 is not None:
            ts += dataframe[dataframe[obj3] == row[obj3]].event_timestamp.tolist()

        min_ts = min(ts, default=None)
        max_ts = max(ts, default=None)
        if min_ts is not None and max_ts is not None:
            rng_ts = max_ts - min_ts
            new_ts = Faker().date_time_between(start_date=min_ts - 0.05 * rng_ts, end_date=max_ts + 0.05 * rng_ts)
            new_rows.at[idx, 'event_timestamp'] = new_ts

    new_rows['y'] = 3  # Set outlier label to 3
    dataframe = pd.concat([dataframe, new_rows])

    col = dataframe['event_id']
    dataframe.insert(1, 'old_event_id', col)
    dataframe = dataframe.sort_values(by=['event_timestamp'])
    dataframe = dataframe.reset_index(drop=True)
    dataframe['event_id'] = dataframe.index

    return dataframe


def dataset_generation(dataframe, data_dict, iterations=10, anom_per=0.05, initial_seed=0):
    import pandas as pd
    import numpy as np
    from joblib import Parallel, delayed
    from datetime import timedelta
    import time
    import os

    start_time = time.time()

    if not os.path.exists(data_dict['path']):
        os.makedirs(data_dict['path'])

    def process_iteration(i):
        itr = i + 1
        print('Iteration', itr)

        random_state = initial_seed + i
        np.random.seed(random_state)

        df_outliers = gen_event_outliers(dataframe=dataframe, obj1=data_dict['obj1'], obj2=data_dict['obj2'],
                                         obj3=data_dict['obj3'], start=data_dict['start'], end=dataframe.shape[1],
                                         n=dataframe.shape[0] * anom_per + 1 / 3 * anom_per ** 2 * dataframe.shape[0],
                                         k=data_dict['k'], random_state=random_state)

        # Create dummy variables
        df_dummies = pd.get_dummies(df_outliers['event_activity'], prefix='event_act')

        # Insert the dummy columns after the start column
        for i, (col_name, col_data) in enumerate(df_dummies.iteritems()):
            df_outliers.insert(data_dict['start'] + 1 + i, col_name, col_data)

        col_to_move = df_outliers.pop("y")
        df_outliers = pd.concat([df_outliers, col_to_move], axis=1)

        df_outliers.columns = df_outliers.columns.str.lower().str.replace(' ', '_')

        filename = data_dict['path'] + data_dict['name'] + '_outlier_' + str(anom_per) + 'prc_' + str(itr) + '.csv'
        df_outliers.to_csv(filename, index=False)

        if itr == 1:
            print("Range of activity columns: [{} - {}]".format(data_dict['start'] + 1,
                                                                data_dict['start'] + len(df_dummies.columns)))

    # use joblib to parallelize the loop
    Parallel(n_jobs=-1)(delayed(process_iteration)(i) for i in range(iterations))

    end_time = time.time()  # get the end time
    total_time = end_time - start_time  # calculate total time
    time_object = timedelta(seconds=total_time)  # convert total time to datetime.timedelta object

    # Extract minutes and seconds
    minutes = int(time_object.total_seconds() // 60)
    seconds = int(time_object.total_seconds() % 60)

    print(f"Total time: {minutes} minutes {seconds} seconds")  # print the total time taken