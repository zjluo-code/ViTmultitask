#!/usr/bin/env python
# coding=utf-8
import os
import time
import math
import h5py
import logging
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
    LayerNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D,
    GlobalMaxPooling2D, Softmax, Concatenate, Dropout, Layer, MultiHeadAttention
)
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- GPU Configuration ---
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if gpu_list:
    try:
        gpu_idx = 0 if len(gpu_list) > 2 else 0
        tf.config.experimental.set_visible_devices(gpu_list[gpu_idx], 'GPU')
        tf.config.experimental.set_memory_growth(gpu_list[gpu_idx], True)
        print(f"GPU {gpu_idx} configured successfully.")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")

# Define the 4 core task label names to predict
TARGET_NAMES = [
    'FIT_RES_D4000_n',
    'FIT_RES_M',
    'DESI_redshift',
    'FIT_RES_EW_hd_a'  # Target: H-delta equivalent width
]


# --- 1. Single-file reading and label extraction function ---
def read_h5_file(file_path):
    """
    Read a single HDF5 file, extract images, 4 physical labels and ZCAT_TARGETID.
    Since the original HDF5 dataset has been pre-filtered, we directly read and convert data types here.
    """
    with h5py.File(file_path, 'r') as F:
        # 1. Load images [N, H, W, C]
        x_data = np.array(F['image'][:], dtype='f4')
        if x_data.ndim == 4 and x_data.shape[1] == 5:
            x_data = np.transpose(x_data, (0, 2, 3, 1))
            
        # 2. Extract 4 multi-task labels
        labels_list = []
        for name in TARGET_NAMES:
            label_arr = np.array(F[name][:], dtype='f4').reshape(-1, 1)
            labels_list.append(label_arr)
            
        y_phys = np.hstack(labels_list)
        
        # 3. Extract ZCAT_TARGETID
        target_ids = np.array(F['ZCAT_TARGETID'][:], dtype='i8')

    # Apply log10 transformation to galaxy mass
    y_phys[:, 1] = np.log10(y_phys[:, 1] + 1e-8)
    
    return x_data, y_phys, target_ids


# --- 2. Data loading and pipeline construction ---
def load_datasets_split(base_path):
    """
    Load the pre-split train, valid, test HDF5 files,
    and apply Z-score normalisation to all splits using the training set mean and standard deviation.
    """
    print("Loading pre-split datasets (train / valid / test)...")
    
    train_file = os.path.join(base_path, 'DESI_HSC_BGS_MDnHdA_train.h5')
    valid_file = os.path.join(base_path, 'DESI_HSC_BGS_MDnHdA_valid.h5')
    test_file  = os.path.join(base_path, 'DESI_HSC_BGS_MDnHdA_test.h5')
    
    X_train, Y_train_phys, train_ids = read_h5_file(train_file)
    X_valid, Y_valid_phys, valid_ids = read_h5_file(valid_file)
    X_test,  Y_test_phys,  test_ids  = read_h5_file(test_file)
    
    print(f"Train samples: {X_train.shape[0]}, Valid samples: {X_valid.shape[0]}, Test samples: {X_test.shape[0]}")

    # Compute training set mean and standard deviation
    y_mean = np.mean(Y_train_phys, axis=0)
    y_std = np.std(Y_train_phys, axis=0)
    y_std[y_std == 0] = 1.0  

    np.save('phy_label_mean.npy', y_mean)
    np.save('phy_label_std.npy', y_std)
    
    print("\nNormalization parameters computed from Train Set:")
    for name, m, s in zip(TARGET_NAMES, y_mean, y_std):
        print(f"Target [{name:<18}]: Mean = {m:.4f}, Std = {s:.4f}")

    
    Y_train_norm = (Y_train_phys - y_mean) / y_std
    Y_valid_norm = (Y_valid_phys - y_mean) / y_std
    Y_test_norm  = (Y_test_phys  - y_mean) / y_std

    return (X_train, Y_train_norm), (X_valid, Y_valid_norm), (X_test, Y_test_norm, Y_test_phys, test_ids), y_mean, y_std


# --- Lossless astronomical image augmentation ---
@tf.function
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    return image, label


# --- 3. Loss function and custom evaluation metrics ---
class MMDMultiTaskHuberLoss(tf.keras.losses.Loss):
    def __init__(self, weights=[2.0, 1.0, 1.0, 3.0], delta=1.0, mmd_weight=1.0, 
                 sigmas=[0.1, 0.5, 1.0, 2.0, 5.0], reduction=tf.keras.losses.Reduction.AUTO, 
                 name="mmd_informed_huber_loss", **kwargs):
        super(MMDMultiTaskHuberLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        if isinstance(weights, tf.Tensor):
            self.task_weights = weights
        else:
            self.task_weights = tf.constant(weights, dtype=tf.float32)
            
        self.delta = delta
        self.mmd_weight = mmd_weight
        self.sigmas = sigmas

    def _rbf_kernel(self, x, y):
        x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
        dist_sq = x_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(x, y, transpose_b=True)
        dist_sq = tf.maximum(0.0, dist_sq)
        
        kernel_val = 0.0
        for sigma in self.sigmas:
            gamma = 1.0 / (2.0 * (sigma ** 2))
            kernel_val += tf.exp(-gamma * dist_sq)
        return kernel_val

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        huber_per_element = 0.5 * tf.square(quadratic) + self.delta * linear
        
        loss_per_task = tf.reduce_mean(huber_per_element, axis=0)
        base_loss = tf.reduce_sum(loss_per_task * tf.cast(self.task_weights, dtype=y_pred.dtype))

        mmd_loss_total = 0.0
        for i in range(4):
            y_p_task = y_pred[:, i:i+1]
            y_t_task = y_true[:, i:i+1]
            
            K_pp = self._rbf_kernel(y_p_task, y_p_task)
            K_tt = self._rbf_kernel(y_t_task, y_t_task)
            K_pt = self._rbf_kernel(y_p_task, y_t_task)
            
            task_mmd = tf.reduce_mean(K_pp) + tf.reduce_mean(K_tt) - 2.0 * tf.reduce_mean(K_pt)
            mmd_loss_total += self.task_weights[i] * task_mmd

        total_loss = base_loss + self.mmd_weight * mmd_loss_total
        return total_loss

    def get_config(self):
        config = super(MMDMultiTaskHuberLoss, self).get_config()
        weights_val = self.task_weights.numpy().tolist() if isinstance(self.task_weights, tf.Tensor) else list(self.task_weights)
        config.update({
            "weights": weights_val,
            "delta": self.delta,
            "mmd_weight": self.mmd_weight,
            "sigmas": self.sigmas
        })
        return config


class MMDMetric(tf.keras.metrics.Metric):
    def __init__(self, weights=[2.0, 1.0, 1.0, 3.0], sigmas=[0.1, 0.5, 1.0, 2.0, 5.0], name='mmd_loss', **kwargs):
        super(MMDMetric, self).__init__(name=name, **kwargs)
        if isinstance(weights, tf.Tensor):
            self.task_weights = weights
        else:
            self.task_weights = tf.constant(weights, dtype=tf.float32)
        self.sigmas = sigmas
        self.total_mmd = self.add_weight(name='total_mmd', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def _rbf_kernel(self, x, y):
        x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
        dist_sq = x_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(x, y, transpose_b=True)
        dist_sq = tf.maximum(0.0, dist_sq)
        
        kernel_val = 0.0
        for sigma in self.sigmas:
            gamma = 1.0 / (2.0 * (sigma ** 2))
            kernel_val += tf.exp(-gamma * dist_sq)
        return kernel_val

    def update_state(self, y_true, y_pred, sample_weight=None):
        mmd_loss_total = 0.0
        for i in range(4):
            y_p_task = y_pred[:, i:i+1]
            y_t_task = y_true[:, i:i+1]
            
            K_pp = self._rbf_kernel(y_p_task, y_p_task)
            K_tt = self._rbf_kernel(y_t_task, y_t_task)
            K_pt = self._rbf_kernel(y_p_task, y_t_task)
            
            task_mmd = tf.reduce_mean(K_pp) + tf.reduce_mean(K_tt) - 2.0 * tf.reduce_mean(K_pt)
            mmd_loss_total += self.task_weights[i] * task_mmd

        self.total_mmd.assign_add(mmd_loss_total)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_mmd / self.count

    def reset_state(self):
        self.total_mmd.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super(MMDMetric, self).get_config()
        weights_val = self.task_weights.numpy().tolist() if isinstance(self.task_weights, tf.Tensor) else list(self.task_weights)
        config.update({
            "weights": weights_val,
            "sigmas": self.sigmas
        })
        return config


class SingleTaskHuberMetric(tf.keras.metrics.Metric):
    def __init__(self, task_idx, delta=1.0, name='task_huber', **kwargs):
        super(SingleTaskHuberMetric, self).__init__(name=name, **kwargs)
        self.task_idx = task_idx
        self.delta = delta
        self.total_huber = self.add_weight(name='total_huber', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_p_task = y_pred[:, self.task_idx:self.task_idx+1]
        y_t_task = y_true[:, self.task_idx:self.task_idx+1]
        
        error = y_t_task - y_p_task
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        huber_per_element = 0.5 * tf.square(quadratic) + self.delta * linear
        
        task_huber = tf.reduce_mean(huber_per_element)
        self.total_huber.assign_add(task_huber)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_huber / self.count

    def reset_state(self):
        self.total_huber.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super(SingleTaskHuberMetric, self).get_config()
        config.update({
            "task_idx": self.task_idx,
            "delta": self.delta
        })
        return config


# --- 4. ViT Block modules ---
class PatchEmbedding(Layer):
    def __init__(self, patch_size=4, stride=2, embed_dim=80, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.projection = Conv2D(embed_dim, kernel_size=patch_size, strides=stride, padding='same')

    def call(self, x):
        patches = self.projection(x)
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, (batch_size, -1, self.embed_dim))
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size, 
            "stride": self.stride, 
            "embed_dim": self.embed_dim
        })
        return config


class PositionEmbedding(Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.pos_embeddings = self.add_weight(
            shape=(sequence_length, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embeddings")
    
    def call(self, inputs):
        return inputs + self.pos_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "embed_dim": self.embed_dim})
        return config


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim, "num_heads": self.num_heads,
            "ff_dim": self.ff_dim, "rate": self.rate
        })
        return config


# --- 5. ViT multi-task network architecture ---
class ViTMultiTask(Model):
    def __init__(self, patch_size, stride, embed_dim, num_patches, num_heads, ff_dim, 
                 num_transformer_blocks, mlp_head_units, n_outputs=4, **kwargs):
        super(ViTMultiTask, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_head_units = mlp_head_units
        self.n_outputs = n_outputs

        total_seq_length = num_patches + 4

        self.patch_embedding = PatchEmbedding(patch_size, stride, embed_dim)
        self.pos_emb = PositionEmbedding(total_seq_length, embed_dim)
        
        self.task_tokens = self.add_weight(
            shape=(1, 4, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="task_tokens"
        )
        self.transformer_layers = [
            TransformerBlock(embed_dim, num_heads, ff_dim) 
            for _ in range(num_transformer_blocks)
        ]
        
        self.head_d4000 = Sequential([
            Dense(256, activation='gelu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='gelu'),
            Dense(1, name="d4000_out")
        ], name="head_d4000")

        self.head_mass = Sequential([
            Dense(mlp_head_units[0], activation='relu'),
            Dropout(0.2),
            LayerNormalization(epsilon=1e-6),
            Dense(mlp_head_units[1], activation='relu'),
            Dropout(0.2),
            LayerNormalization(epsilon=1e-6),
            Dense(1, name="mass_out")
        ], name="head_mass")

        self.head_redshift = Sequential([
            Dense(mlp_head_units[0], activation='relu'),
            Dropout(0.2),
            LayerNormalization(epsilon=1e-6),
            Dense(mlp_head_units[1], activation='relu'),
            Dropout(0.2),
            LayerNormalization(epsilon=1e-6),
            Dense(1, name="redshift_out")
        ], name="head_redshift")

        self.head_ew_hd = Sequential([
            Dense(512, activation='gelu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='gelu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='gelu'),
            Dense(1, name="ew_hd_out")
        ], name="head_ew_hd")

    def call(self, inputs, training=False):
        x = self.patch_embedding(inputs)
        
        batch_size = tf.shape(inputs)[0]
        batch_task_tokens = tf.tile(self.task_tokens, [batch_size, 1, 1])
        x = tf.concat([batch_task_tokens, x], axis=1)
        x = self.pos_emb(x)
        
        for block in self.transformer_layers:
            x = block(x, training=training)
        
        feat_d4000   = x[:, 0, :]
        feat_mass    = x[:, 1, :]
        feat_z       = x[:, 2, :]
        feat_ew_hd   = x[:, 3, :]
        
        out_d4000 = self.head_d4000(feat_d4000, training=training)
        out_mass  = self.head_mass(feat_mass, training=training)
        out_z     = self.head_redshift(feat_z, training=training)
        out_ew    = self.head_ew_hd(feat_ew_hd, training=training)
        
        return tf.concat([out_d4000, out_mass, out_z, out_ew], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size, "stride": self.stride, 
            "num_patches": self.num_patches, "embed_dim": self.embed_dim, 
            "num_heads": self.num_heads, "ff_dim": self.ff_dim, 
            "num_transformer_blocks": self.num_transformer_blocks,
            "mlp_head_units": self.mlp_head_units, "n_outputs": self.n_outputs
        })
        return config


# --- 6. Execute data loading and pipeline setup ---
base_dir = '/data/zjluo/HSC_spec_fig/'
(X_train, Y_train_norm), (X_valid, Y_valid_norm), (X_test, Y_test_norm, Y_test_phys, test_ids), y_mean, y_std = load_datasets_split(base_dir)

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train_norm))
    .shuffle(10000)
    .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((X_valid, Y_valid_norm))
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)


# --- 7. Model building and compilation ---
IMG_SIZE, CHANNELS = 64, 5

patch_size = 4
stride = 4
embed_dim = patch_size * patch_size * CHANNELS   # 80
num_patches = (IMG_SIZE // stride) ** 2          # 256
num_heads, ff_dim = 4, 4 * embed_dim
num_layers = 8
mlp_units = [256, 192]
n_outputs = 4

checkpoint_dir = './checkpoints/desi_hsc_mmd_256bs'
os.makedirs('./checkpoints', exist_ok=True)

# Loss function and custom metric instances
mmd_loss_fn = MMDMultiTaskHuberLoss(
    weights=[2.0, 1.0, 1.0, 3.0], 
    delta=1.0, 
    mmd_weight=1.0, 
    sigmas=[0.1, 0.5, 1.0, 2.0, 5.0]
)

mmd_metric_fn = MMDMetric(
    weights=[2.0, 1.0, 1.0, 3.0], 
    sigmas=[0.1, 0.5, 1.0, 2.0, 5.0]
)

d4000_huber = SingleTaskHuberMetric(task_idx=0, delta=1.0, name='d4000_huber')
mass_huber  = SingleTaskHuberMetric(task_idx=1, delta=1.0, name='mass_huber')
z_huber     = SingleTaskHuberMetric(task_idx=2, delta=1.0, name='z_huber')
ew_hd_huber = SingleTaskHuberMetric(task_idx=3, delta=1.0, name='ew_hd_huber')

custom_objects = {
    'PatchEmbedding': PatchEmbedding,
    'PositionEmbedding': PositionEmbedding,
    'TransformerBlock': TransformerBlock,
    'ViTMultiTask': ViTMultiTask,
    'MMDMultiTaskHuberLoss': MMDMultiTaskHuberLoss,
    'MMDMetric': MMDMetric,
    'SingleTaskHuberMetric': SingleTaskHuberMetric
}

lr_schedule = CosineDecay(initial_learning_rate=1e-4, decay_steps=50000, alpha=0.01)
optimizer = Adam(learning_rate=lr_schedule)

all_metrics = ['mae', mmd_metric_fn, d4000_huber, mass_huber, z_huber, ew_hd_huber]

if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
    print(f"Loading existing model from {checkpoint_dir}...")
    ViT_model = load_model(checkpoint_dir, custom_objects=custom_objects, compile=False)
    ViT_model.compile(optimizer=optimizer, loss=mmd_loss_fn, metrics=all_metrics)
else:
    print("Creating new ViT model...")
    ViT_model = ViTMultiTask(patch_size, stride, embed_dim, num_patches, num_heads, ff_dim, num_layers, mlp_units, n_outputs)
    ViT_model.build((None, IMG_SIZE, IMG_SIZE, CHANNELS))
    ViT_model.compile(optimizer=optimizer, loss=mmd_loss_fn, metrics=all_metrics)

ViT_model.summary()


# --- 8. Callbacks and training execution ---
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        save_best_only=True,
        save_format='tf',
        monitor='val_loss',
        mode='min'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=500,
        restore_best_weights=True
    )
]

print("Starting training...")
history = ViT_model.fit(train_ds, epochs=301, validation_data=valid_ds, callbacks=callbacks)


# --- 9. Save training history (mmd_history_256bs.dat) ---
history_out_file = "mmd_history_256bs.dat"
print(f"Saving training history to {history_out_file}...")

history_keys = list(history.history.keys())
epochs_arr = np.arange(1, len(history.history[history_keys[0]]) + 1).reshape(-1, 1)
history_data = [epochs_arr] + [np.array(history.history[k]).reshape(-1, 1) for k in history_keys]
history_matrix = np.hstack(history_data)

header_str = "Epoch " + " ".join(history_keys)
fmt_list = ["%d"] + ["%.6e"] * len(history_keys)

np.savetxt(history_out_file, history_matrix, header=header_str, fmt=fmt_list, comments='')
print(f"History data successfully written to {history_out_file}.")


# --- 10. Test set predictions and save (mmd_pred_256bs.dat) ---
print("Evaluating on test dataset and saving predictions...")
y_pred_test_norm = ViT_model.predict(X_test)

# Inverse normalize
y_pred_test_phys = y_pred_test_norm * y_std + y_mean

# Physical redshift clipping (non-negative)
y_pred_test_phys[:, 2] = np.maximum(0.0, y_pred_test_phys[:, 2])

# Concatenate: col1 (ZCAT_TARGETID) | cols 2-5 (true) | cols 6-9 (pred)
pred_matrix = np.column_stack([
    test_ids.reshape(-1, 1),
    Y_test_phys,
    y_pred_test_phys
])

pred_out_file = "mmd_pred_256bs.dat"
pred_header = (
    "ZCAT_TARGETID "
    "TRUE_FIT_RES_D4000_n TRUE_FIT_RES_M TRUE_DESI_redshift TRUE_FIT_RES_EW_hd_a "
    "PRED_FIT_RES_D4000_n PRED_FIT_RES_M PRED_DESI_redshift PRED_FIT_RES_EW_hd_a"
)

# Save format: TARGETID as long integer (%d), other columns with 6 decimal places (%e)
fmt_spec = ['%d'] + ['%.6e'] * 8
np.savetxt(pred_out_file, pred_matrix, header=pred_header, fmt=fmt_spec, comments='')
print(f"Test predictions successfully written to {pred_out_file}.")
