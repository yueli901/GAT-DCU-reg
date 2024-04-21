import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from config import DATA_PATH

class Scaler:
    def __init__(self, data):
        # Select only the first two features for calculating mean and std
        reshaped_data = tf.reshape(data[..., :2], [-1, 2])
        valid_data_mask = ~tf.math.is_nan(reshaped_data)
    
        means = []
        stds = []
        
        # Iterate over each feature
        for i in range(reshaped_data.shape[-1]):
            # Apply the mask to each feature
            feature_data = tf.boolean_mask(reshaped_data[:, i], valid_data_mask[:, i])
            
            # Calculate mean and standard deviation for the feature
            mean = tf.reduce_mean(feature_data)
            std = tf.math.reduce_std(feature_data)
        
            # Append the results to the lists
            means.append(mean)
            stds.append(std)
        
        # Convert lists to tensors
        self.mean = tf.stack(means)
        self.std = tf.stack(stds)
    
    def transform(self, data):
        # Normalize only the first two features
        normalized_data = (data[..., :2] - self.mean) / (self.std + 1e-8)
        return normalized_data
    
    def inverse_transform(self, data):
        # Reverse normalization for both features
        denormalized_data = data[..., :2] * self.std + self.mean
        return denormalized_data



def distance_matrix():
    """
    node_features: dual graph node features [n=498, d=4] (lat, lon, distance, duration)
    e_in_array / e_out_array: len=498, each element is the list of incoming/outgoing neighbour node index, total edges in each graph e=2646
    edge_loc: [e_unique=180, d=2]
    """
    sensor_data = pd.read_excel(os.path.join(DATA_PATH, 'tris_edge_features.xlsx'))
    sensor_data['Duration (s)'] = sensor_data['Duration (s)'].str.replace('s', '').astype(float)

    # edge features are nodes locations in the original graph
    edge_loc = pd.read_csv(os.path.join(DATA_PATH, 'tris_node_features.csv'))
    edge_loc = edge_loc[['Latitude', 'Longitude']].to_numpy()

    # node features are edge features and sensor locations in the original graph
    node_features = sensor_data[['Latitude', 'Longitude', 'Distance (m)', 'Duration (s)', 'Origin', 'Destination']].to_numpy()

    n = len(sensor_data)
    
    e_in = {i: [] for i in range(n)}
    e_out = {i: [] for i in range(n)}

    # Map each junction to the list of road segments (edges) connected to it, both directions
    junction_to_edges = {}
    for i, row in sensor_data.iterrows():
        origin, dest = row['Origin'], row['Destination']
        junction_to_edges.setdefault(origin, []).append(i)
        junction_to_edges.setdefault(dest, []).append(i)

    # Build the e_in and e_out lists
    for edge_id, row in sensor_data.iterrows():
        origin, dest = row['Origin'], row['Destination']
        # For e_out: Find all edges that are also connected to this edge's destination
        e_out[edge_id] = [other_edge for other_edge in junction_to_edges[dest] if other_edge != edge_id]
        # For e_in: Find all edges that are also connected to this edge's origin
        e_in[edge_id] = [other_edge for other_edge in junction_to_edges[origin] if other_edge != edge_id]
        # This is dual map structure is considered better than include ending at the origin or starting from the destination of an edge, attention takes care of everything

    # Convert e_in and e_out to arrays
    e_in_array = np.array([e_in[i] for i in range(n)], dtype=object)
    e_out_array = np.array([e_out[i] for i in range(n)], dtype=object)
    return node_features, edge_loc, e_in_array, e_out_array

def sensor_index():
    df = pd.read_excel(os.path.join(DATA_PATH, 'tris_edge_features.xlsx'))
    sensor_idx = {sensor_id: i for i, sensor_id in enumerate(df['Id'])}
    return sensor_idx

def fill_missing(data):
	# data = data.copy()
	# data[data < 1e-5] = float('nan')
	# data = data.fillna(method='pad')
	# data = data.fillna(method='bfill')
	return data

