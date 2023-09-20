import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import sys
# if sys.gettrace() is not None:
#     tf.compat.v1.enable_eager_execution()
#     tf.config.run_functions_eagerly(True)
#     tf.data.experimental.enable_debug_mode()
#     tf.print("Eager Execution!!!!!!!!!!")
# else:
#     tf.config.run_functions_eagerly(False)
#     tf.compat.v1.disable_eager_execution()
#     tf.print("Graph Execution!!!!!!!!!!")
# tf.print(tf.executing_eagerly())

import argparse
import os

from .data.const import ALL_CONFIGS


parser = argparse.ArgumentParser(description = "Hahaha")

# train and test
parser.add_argument("--mode", type = str)
parser.add_argument("--model", type = str)
parser.add_argument("--channels", type = int, default = 64)
parser.add_argument("--ckpt-path", type = str, default = None)

parser.add_argument("--dataset", type = str, default = "RAVEN")
parser.add_argument("--dataset-path", type = str)
parser.add_argument("--image-size", type = int, default = 160)

# RAVEN specific
parser.add_argument("--train-configs", nargs = '+', default = ALL_CONFIGS)
parser.add_argument("--train-configs-proportion", type = int, nargs = '+', default = None)

parser.add_argument("--output-path", type = str)
parser.add_argument("--epoch-num", type = int, default = 2)
parser.add_argument("--batch-size", type = int, default = 4)
parser.add_argument("--num-workers", type = int, default = 0)
parser.add_argument("--learning-rate", type = float, default = None)
parser.add_argument("--weight-decay", type = float, default = 0.0)
parser.add_argument("--loss", type = str, default = "ce")
parser.add_argument("--seed", type = int, default = 2023)
parser.add_argument("--tag", type = str, default = "NA")

# hyper param search
parser.add_argument("--n-trials", type = int, default = 30)
parser.add_argument("--n-startup-trials", type = int, default = None)
parser.add_argument("--n-warmup-steps", type = int, default = None)
parser.add_argument("--interval-steps", type = int, default = None)
parser.add_argument("--learning-rate-range", nargs = '+', type = float, default = None)
parser.add_argument("--weight-decay-range", nargs = '+', type = float, default = None)

args = parser.parse_args()

# ################################# set seeds for tensorflow ######################################################
# random.seed(args.seed)
# np.random.seed(args.seed)
# tf.random.set_seed(args.seed)
# tf.experimental.numpy.random.seed(args.seed)
# tf.keras.utils.set_random_seed(args.seed)
#
# # tf.set_random_seed(args.seed)  # for tf.compact.v1
#
# # When running on the CuDNN backend, two further options must be set
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
#
# # Set a fixed value for the hash seed
# os.environ["PYTHONHASHSEED"] = str(args.seed)


if args.output_path is not None and "~" in args.output_path:
    args.output_path = os.path.expanduser(args.output_path)

if args.ckpt_path is not None and "~" in args.ckpt_path:
    args.ckpt_path = os.path.expanduser(args.ckpt_path)


