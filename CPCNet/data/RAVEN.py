import os
from glob import glob
from os.path import join
from pathlib import Path
import numpy as np
import tensorflow as tf

from .const import ALL_CONFIGS


RULE_ATTR = [[0, 1, 0, 0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 1, 0, 0, 0],
             [1, 0, 0, 0, 1, 1, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0, 0, 1],
             [0, 1, 0, 0, 0, 0, 0, 0, 1]]


def read_RAVEN_file(x):
    """
    tf.py_function wrapper.
    """

    image, target, config = tf.py_function(_read_file, [x], (tf.float32, tf.int64, tf.int64))

    return image, target, config


def _read_file(file_path):

    file_path = file_path.numpy().decode()

    data = np.load(file_path)

    image = tf.convert_to_tensor(data["image"], dtype = tf.float32)
    label = tf.convert_to_tensor(data["target"], dtype = tf.int64)

    config = tf.convert_to_tensor(ALL_CONFIGS.index(Path(file_path).parts[-2]), dtype = tf.int64)

    del data

    return image, label, config


def read_file_test(x):
    """
    tf.py_function wrapper.
    """

    image, target, item, meta_target = tf.py_function(_read_file_test, [x], (tf.float32, tf.int64, tf.string, tf.uint8))

    return image, target, item, meta_target


def _read_file_test(file_path):

    file_path = file_path.numpy().decode()

    data = np.load(file_path)

    image = tf.convert_to_tensor(data["image"], dtype = tf.float32)
    label = tf.convert_to_tensor(data["target"], dtype = tf.int64)

    item = tf.convert_to_tensor(Path(file_path).parts[-1], dtype = tf.string)

    meta_target = np.zeros(18, dtype = np.uint)
    for rule_attr_row in data["meta_matrix"]:
        if ([0, 0, 0, 0, 0, 0, 0, 0, 0] != rule_attr_row).any():
            found = False
            for idx, b in enumerate(RULE_ATTR):
                if (rule_attr_row == b).all():
                    meta_target[idx] = 1
                    found = True
                    break
            if not found:
                raise ValueError(f"Unknown rule_attr: {rule_attr_row}.")

    meta_target = tf.convert_to_tensor(meta_target, dtype = tf.uint8)

    del data

    return image, label, item, meta_target


def get_RAVEN_dataset(dataset_root, split, spatial_configs, gain = None):

    if gain is None:
        gain = [1] * len(spatial_configs)

    file_names = []

    for idx, config in enumerate(spatial_configs):
        config_path = os.path.join(dataset_root, config)
        if not os.path.exists(config_path):
            raise ValueError(f"Cannot find config directory at {config_path}.")
        file_names.extend([file_name for file_name in glob(join(config_path, "*.npz")) if split in file_name] * gain[idx])

    ds = tf.data.Dataset.from_tensor_slices(file_names)

    return ds
