import tensorflow as tf
# Disable all GPUs
tf.config.set_visible_devices([], 'GPU')

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
import os
import yaml
import logging
import argparse
import datetime
import random
import h5py

import config
from config import PARAM_PATH, EVALUATE_PATH

# Set logging
# current_time = datetime.datetime.now()
# formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
logging.basicConfig(filename=os.path.join(EVALUATE_PATH,'training_log.log'), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', filemode='w')

import model
import data.dataloader
from data.dataloader import get_geo_feature



# Define a function for the training loop
def train_model(net, train_dataset, eval_dataset, node_features, scaler, num_epochs, optimizer, clip_gradient):
    for epoch in range(num_epochs):
        for step, (data, label, _, _) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                data = tf.transpose(data, perm=[1, 0, 2, 3])  # [b, t, n, d] to [n, b, t, d]
                label = tf.transpose(label, perm=[1, 0, 2, 3])  # [b, t, n, d] to [n, b, t, d]
                pred = net(node_features, data, label, is_training=True)
                mse, mse_speed, mse_volume, mae_speed, mae_volume = compute_loss(label, pred, scaler)
            grads = tape.gradient(mse, net.trainable_variables)
            grads = [tf.clip_by_value(grad, -clip_gradient, clip_gradient) for grad in grads]
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            
            # Logging and additional metrics
            if step % 10 == 0:  # Adjust the logging frequency as needed
                logging.info(f'Epoch {epoch}, Step {step}, Training MSE: {mse.numpy()}, Training RMSE_speed: {np.sqrt(mse_speed.numpy())}, Training RMSE_volume:{np.sqrt(mse_volume.numpy())}, Training MAE_speed: {mae_speed.numpy()}, Training MAE_volume:{mae_volume.numpy()}')

            if step % 50 == 0:
                mse, mse_speed, mse_volume, mae_speed, mae_volume = evaluate_model(net, eval_dataset, node_features, scaler)
                logging.info(f'Epoch {epoch}, Step {step}, Validation MSE: {mse.numpy()}, Validation RMSE_speed: {np.sqrt(mse_speed.numpy())}, Validation RMSE_volume:{np.sqrt(mse_volume.numpy())}, Validation MAE_speed: {mae_speed.numpy()}, Validation MAE_volume:{mae_volume.numpy()}')


def compute_loss(label, pred, scaler):
    # Extracting the first feature from label, pred, and mask
    mask = label[:, :, :, 2:4]
    label = label[:, :, :, :2]
    pred = pred[:, :, :, :2]

    # Invert the mask: 1 for valid data, 0 for NA
    valid_data_mask = 1 - mask

    # Apply the mask to label and pred
    masked_label = label * valid_data_mask
    masked_pred = pred * valid_data_mask

    # Calculate MSE loss only for valid data points
    mse = tf.reduce_sum(tf.square(masked_label - masked_pred)) / tf.reduce_sum(valid_data_mask)

    pred = scaler.inverse_transform(pred)
    label = scaler.inverse_transform(label)
    masked_label = label * valid_data_mask
    masked_pred = pred * valid_data_mask
    mse_speed = tf.reduce_sum(tf.square(masked_label[..., 0] - masked_pred[..., 0])) / tf.reduce_sum(valid_data_mask[...,0])
    mse_volume = tf.reduce_sum(tf.square(masked_label[..., 1] - masked_pred[..., 1])) / tf.reduce_sum(valid_data_mask[...,1])
    mae_speed = tf.reduce_sum(tf.abs(masked_label[..., 0] - masked_pred[..., 0])) / tf.reduce_sum(valid_data_mask[...,0])
    mae_volume = tf.reduce_sum(tf.abs(masked_label[..., 1] - masked_pred[..., 1])) / tf.reduce_sum(valid_data_mask[...,1])
    
    return mse, mse_speed, mse_volume, mae_speed, mae_volume

# Define a function for evaluation
def evaluate_model(net, dataset, node_features, scaler):
    # Evaluate the model on the given dataset
    MSE = 0
    MSE_speed = 0
    MSE_volume = 0
    MAE_speed = 0
    MAE_volume = 0
    num_batches = 0
    prediction = []
    ground_truth = []
    for data, label, _, _ in dataset:
        data = tf.transpose(data, perm=[1, 0, 2, 3])  # [b, t, n, d] to [n, b, t, d]
        label = tf.transpose(label, perm=[1, 0, 2, 3])  # [b, t, n, d] to [n, b, t, d]
        pred = net(node_features, data, label, is_training=False)
        mse, mse_speed, mse_volume, mae_speed, mae_volume = compute_loss(label, pred, scaler)
        MSE += mse
        MSE_speed += mse_speed
        MSE_volume += mse_volume
        MAE_speed += mae_speed
        MAE_volume += mae_volume
        num_batches += 1
    mse = MSE / num_batches
    mse_speed = MSE_speed / num_batches
    mse_volume = MSE_volume / num_batches
    mae_speed = MAE_speed / num_batches
    mae_volume = MAE_volume / num_batches
    return mse, mse_speed, mse_volume, mae_speed, mae_volume


# Main function
def main(args):
    # Load settings
    with open(args.file, 'r') as f:
        settings = yaml.safe_load(f)

    np.random.seed(settings['seed'])
    tf.random.set_seed(settings['seed'])
    random.seed(settings['seed'])

    dataset_setting = settings['dataset']
    model_setting = settings['model']
    train_setting = settings['training']

	# set meta hiddens
    if 'meta_hiddens' in model_setting.keys():
        config.MODEL['meta_hiddens'] = model_setting['meta_hiddens']

    # add timestamp for model name
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    filename = f"{model_setting['name']}-{str(15*int(dataset_setting['time_interval']))}min-{str(args.epochs)}-{formatted_time}.h5"
    name = os.path.join(PARAM_PATH, filename)
    
    model_type = getattr(model, model_setting['type'])
    net = model_type.net(settings) # initialize model

    # Data setup
    train_dataset, eval_dataset, test_dataset, scaler = getattr(data.dataloader, dataset_setting['dataloader'])(settings)
    
    # Optimizer setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_setting['lr'])
    
    # Training
    node_features, _ = get_geo_feature()
    train_model(net, train_dataset, eval_dataset, node_features, scaler, args.epochs, optimizer, train_setting['clip_gradient'])

    # Evaluation
    mse, mse_speed, mse_volume, mae_speed, mae_volume = evaluate_model(net, eval_dataset, node_features, scaler)
    logging.info(f'FINAL Validation MSE: {mse.numpy()}, Validation RMSE_speed: {np.sqrt(mse_speed.numpy())}, Validation RMSE_volume:{np.sqrt(mse_volume.numpy())}, Validation MAE_speed: {mae_speed.numpy()}, Validation MAE_volume:{mae_volume.numpy()}')

    # Test
    mse, mse_speed, mse_volume, mae_speed, mae_volume = evaluate_model(net, test_dataset, node_features, scaler)
    logging.info(f'FINAL Test MSE: {mse.numpy()}, Test RMSE_speed: {np.sqrt(mse_speed.numpy())}, Test RMSE_volume:{np.sqrt(mse_volume.numpy())}, Test MAE_speed: {mae_speed.numpy()}, Test MAE_volume:{mae_volume.numpy()}')
    
    # Save model parameters
    net.save_weights(name)
    print("Model saved successfully.")

    # rename log file
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    os.rename(os.path.join(EVALUATE_PATH,'training_log.log'), os.path.join(EVALUATE_PATH,f"training_log-{str(15*int(dataset_setting['time_interval']))}min-{str(args.epochs)}-{formatted_time}.log"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--epochs', type=int)
    # parser.add_argument('--feature', type=str)
    args = parser.parse_args()

    main(args)