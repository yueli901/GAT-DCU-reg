import os
import h5py
import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from data import utils
from config import DATA_PATH

def get_geo_feature():
    """
    node_features: [n=498, d=4+4+2] lat/lon/dur/dist + origin lat/lon + dest lat/lon (all normalized) + num of n connected to origin + num of n connected to destination
    edge_feature_in/edge_feature_out: [e=2646, d=2] lat/lon of nodes in original graph
    e_in/e_out: list of lists, edge index same as list index, elements in lists are edge index that connect with the edge (original graph) origin or destination
    """
	# get info
    node_features, edge_loc, e_in, e_out = utils.distance_matrix()

    # normalize edge features
    edge_loc = (edge_loc - np.mean(edge_loc, axis=0)) / np.std(edge_loc, axis=0)
    
	# generate edge features
    edge_feature_in = []
    for i, l in enumerate(e_in):
        for j in l:
            edge_feature_in.append(edge_loc[int(node_features[i,4])-1])
    edge_feature_in = np.stack(edge_feature_in)
    
    edge_feature_out = []
    for i, l in enumerate(e_out):
        for j in l:
            edge_feature_out.append(edge_loc[int(node_features[i,5])-1])
    edge_feature_out = np.stack(edge_feature_out)
    
	# merge node features
    node_features_expand = []
    n = node_features.shape[0]
    for i in range(n):
        f = np.concatenate([node_features[i,0:4], 
                            edge_loc[int(node_features[i,4])-1],
                            edge_loc[int(node_features[i,5])-1],
                            np.array([len(e_in[i])]),np.array([len(e_out[i])])])
        node_features_expand.append(f)
    node_features = np.stack(node_features_expand)
    node_features = (node_features - np.mean(node_features, axis=0)) / np.std(node_features, axis=0)
    return node_features, (edge_feature_in, edge_feature_out, e_in, e_out)


def dataloader(dataset):
    """
    print(data.shape)
    print(sites.shape)
    print(timestamps.shape)
    (498, 35040, 2)
    (498,)
    (35040,)
    """
    with h5py.File(os.path.join(DATA_PATH, 'sensor498_2019-01-01_2019-12-31_imputed.h5'), 'r') as f:  
        data = tf.convert_to_tensor(f['data'][:]) 
        sites = tf.convert_to_tensor(f['sites'][:]) 
        timestamps = tf.convert_to_tensor(f['timestamps'][:]) 

    # Reorder sites to the correct sequence
    sensor_index = utils.sensor_index()
    reorder_indices = [sensor_index[sensor_id] for sensor_id in sites.numpy()]
    inverse_indices = np.argsort(reorder_indices)
    data = tf.gather(data, inverse_indices, axis=0)
    sites = tf.gather(sites, inverse_indices, axis=0)

    # aggregate data for different time interval, 1 is original, 2 is 30 minutes, 4 is 1hour, 12 is 3 hours
    ti = dataset['time_interval']
    num_time_steps = data.shape[1]
    reshaped_data = tf.reshape(data, (data.shape[0], -1, ti, data.shape[2]))
    volume_data = tf.reduce_sum(reshaped_data[:, :, :, 0], axis=2) # Volume aggregation by sum
    speed_data = tf.reduce_mean(reshaped_data[:, :, :, 1], axis=2) # Speed aggregation by mean
    data = tf.stack([volume_data, speed_data], axis=-1)
    timestamps = timestamps[::ti]
    
    n_timestamp = int(timestamps.shape[0]) # adjust length of total data
    num_train = int(n_timestamp * dataset['train_prop'])
    num_eval = int(n_timestamp * dataset['eval_prop'])
    num_test = n_timestamp - num_train - num_eval

    train = tf.identity(data[:, :num_train, :])
    eval = tf.identity(data[:, num_train: num_train + num_eval, :])
    test = tf.identity(data[:, -num_test:, :])

    train_timestamps = timestamps[:num_train]
    eval_timestamps = timestamps[num_train: num_train + num_eval]
    test_timestamps = timestamps[-num_test:]

    return (train, train_timestamps), (eval, eval_timestamps), (test, test_timestamps)


def dataiter_all_sensors_seq2seq(df_with_t, scaler, setting, shuffle=True):
    """
    df_with_t: output from dataloader (df, timestamps)
    scaler: scaler that has the params of train

    data: input tensor fo ST-MetaNet, history [b, n=498, t=12, d=13] d=13 (1 traffic, 1 volume, 2 masks, 1 time of the day, 7 day of the week, 1 holidays)
    label: output tensor of ST-MetaNet, ground truth [b, n=498, t=12, d=13]
    """
    dataset = setting['dataset']
    training = setting['training']
    
    df, timestamps = df_with_t
    n, t, _ = df.shape
    df = scaler.transform(df)
    
    nan_mask = tf.cast(tf.math.is_nan(df), tf.float32)
    df = tf.where(tf.math.is_nan(df), tf.constant(-1.0, dtype=tf.float32), df)
    df = tf.concat([df, nan_mask], axis=-1)  # Shape: (n, t, d=4)

    parsed_timestamps = [datetime.datetime.strptime(ts.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for ts in timestamps.numpy()]
    seconds_since_epoch = np.array([ts.timestamp() for ts in parsed_timestamps], dtype=np.float32)
    
    # Time of the day (normalized to [0,1])
    time_of_day = (seconds_since_epoch % 86400) / 86400
    time_of_day = tf.tile(tf.expand_dims(time_of_day, axis=0), [n,1])
    time_of_day = tf.expand_dims(time_of_day, axis=-1)  # Shape: (n, t, 1)
    
    # Day of the week
    day_of_week = np.array([ts.weekday() for ts in parsed_timestamps], dtype=np.int32)
    day_of_week = tf.one_hot(day_of_week, depth=7)
    day_of_week = tf.expand_dims(day_of_week, axis=0)  # Shape becomes (1, t, 7)
    day_of_week = tf.tile(day_of_week, [n, 1, 1])  # Tiled to shape (n, t, 7)

    # holiday
    import holidays
    uk_holidays = holidays.country_holidays('GB', subdiv='ENG')
    is_holiday = np.array([1 if ts.date() in uk_holidays else 0 for ts in parsed_timestamps], dtype=np.float32)
    is_holiday = tf.convert_to_tensor(is_holiday, dtype=tf.float32)
    is_holiday = tf.expand_dims(is_holiday, axis=0)  # Shape becomes (1, t)
    is_holiday = tf.tile(is_holiday, [n, 1])  # Shape becomes (n, t)
    is_holiday = tf.expand_dims(is_holiday, axis=-1)  # Final shape: (n, t, 1)

    df = tf.cast(df, tf.float32)
    time_of_day = tf.cast(time_of_day, tf.float32)
    day_of_week = tf.cast(day_of_week, tf.float32)
    df = tf.concat([df, time_of_day, day_of_week, is_holiday], axis=-1)
    
    input_len = dataset['input_len']
    output_len = dataset['output_len']
    
    total_samples = t - input_len - output_len + 1

    def gen():
        indices = list(range(total_samples))
        if shuffle:
            random.shuffle(indices)
        for i in indices:
            data_batch = df[:, i: i + input_len, :]
            label_batch = df[:, i + input_len: i + input_len + output_len, :]
            data_ts = timestamps[i: i + input_len]
            label_ts = timestamps[i + input_len: i + input_len + output_len]
            yield data_batch, label_batch, data_ts, label_ts
   
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(n, input_len, df.shape[-1]), dtype=tf.float32),
            tf.TensorSpec(shape=(n, output_len, df.shape[-1]), dtype=tf.float32),
            tf.TensorSpec(shape=(input_len), dtype=tf.string),
            tf.TensorSpec(shape=(output_len), dtype=tf.string)
        )
    )
    
    dataset = dataset.batch(training['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def dataloader_all_sensors_seq2seq(setting):
    train, valid, test = dataloader(setting['dataset'])
    train_data, _ = train
    scaler = utils.Scaler(train_data)
    return dataiter_all_sensors_seq2seq(train, scaler, setting, shuffle=True), \
           dataiter_all_sensors_seq2seq(valid, scaler, setting, shuffle=False), \
           dataiter_all_sensors_seq2seq(test, scaler, setting, shuffle=False), \
           scaler
