from os import mkdir
import csv
import os
import tensorflow as tf
from tqdm import tqdm


from .. import args
from ..data.const import DC, D9, D4, CC, UD, LR, CS
from ..data.RAVEN import get_RAVEN_dataset, read_file_test
from ..models.CPCNet import CPCNet, CPCNet_0_Contrasting_Layer, CPCNet_1_Contrasting_Layer, CPCNet_2_Contrasting_Layer, CPCNet_3_Contrasting_Layer, CPCNet_4_Contrasting_Layer, CPCNet_IC

model = None
loss_fn = None

acc_metric = None
loss_a_metric = None
loss_b_metric = None


def test_multi_path():

    global model, loss_fn, loss_a_metric, loss_b_metric, acc_metric

    if "CPCNet" == args.model:
        model = CPCNet(args.image_size, args.channels)
    elif "CPCNet_0_Contrasting_Layer" == args.model:
        model = CPCNet_0_Contrasting_Layer(args.image_size, args.channels)
    elif "CPCNet_1_Contrasting_Layer" == args.model:
        model = CPCNet_1_Contrasting_Layer(args.image_size, args.channels)
    elif "CPCNet_2_Contrasting_Layer" == args.model:
        model = CPCNet_2_Contrasting_Layer(args.image_size, args.channels)
    elif "CPCNet_3_Contrasting_Layer" == args.model:
        model = CPCNet_3_Contrasting_Layer(args.image_size, args.channels)
    elif "CPCNet_4_Contrasting_Layer" == args.model:
        model = CPCNet_4_Contrasting_Layer(args.image_size, args.channels)
    elif "CPCNet_IC" == args.model:
        model = CPCNet_IC(args.image_size, args.channels)
    else:
        raise ValueError("Unknown model: {}".format(args.model))

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True, reduction = "sum_over_batch_size")

    loss_a_metric = tf.keras.metrics.Mean()
    loss_b_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    ckpt = tf.train.Checkpoint(model = model)
    checkpoint_status = ckpt.restore(args.ckpt_path)
    args.output_path = args.ckpt_path+"-test"
    mkdir(args.output_path)

    test_split("train")
    test_split("val")
    test_split("test")

    checkpoint_status.expect_partial()


def test_split(split):

    cs_dataset = get_RAVEN_dataset(args.dataset_path, split, [CS]).map(map_func = read_file_test, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)
    d4_dataset = get_RAVEN_dataset(args.dataset_path, split, [D4]).map(map_func = read_file_test, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)
    d9_dataset = get_RAVEN_dataset(args.dataset_path, split, [D9]).map(map_func = read_file_test, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)
    cc_dataset = get_RAVEN_dataset(args.dataset_path, split, [CC]).map(map_func = read_file_test, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)
    dc_dataset = get_RAVEN_dataset(args.dataset_path, split, [DC]).map(map_func = read_file_test, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)
    lf_dataset = get_RAVEN_dataset(args.dataset_path, split, [LR]).map(map_func = read_file_test, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)
    ud_dataset = get_RAVEN_dataset(args.dataset_path, split, [UD]).map(map_func = read_file_test, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)

    loss_a_dc, loss_b_dc, acc_dc = test_config(dc_dataset, split, DC)
    loss_a_d9, loss_b_d9, acc_d9 = test_config(d9_dataset, split, D9)
    loss_a_d4, loss_b_d4, acc_d4 = test_config(d4_dataset, split, D4)
    loss_a_cc, loss_b_cc, acc_cc = test_config(cc_dataset, split, CC)
    loss_a_lr, loss_b_lr, acc_lr = test_config(lf_dataset, split, LR)
    loss_a_ud, loss_b_ud, acc_ud = test_config(ud_dataset, split, UD)
    loss_a_cs, loss_b_cs, acc_cs = test_config(cs_dataset, split, CS)

    acc = (acc_dc + acc_d9 + acc_d4 + acc_lr + acc_ud + acc_cc + acc_cs) / 7

    acc = f"{acc * 100.0:.2f}"
    acc_dc = f"{acc_dc * 100.0:.2f}"
    acc_d9 = f"{acc_d9 * 100.0:.2f}"
    acc_d4 = f"{acc_d4 * 100.0:.2f}"
    acc_cc = f"{acc_cc * 100.0:.2f}"
    acc_lr = f"{acc_lr * 100.0:.2f}"
    acc_ud = f"{acc_ud * 100.0:.2f}"
    acc_cs = f"{acc_cs * 100.0:.2f}"

    loss_a = (loss_a_dc + loss_a_d9 + loss_a_d4 + loss_a_cc + loss_a_lr + loss_a_ud + loss_a_cs) / 7
    loss_b = (loss_b_dc + loss_b_d9 + loss_b_d4 + loss_b_cc + loss_b_lr + loss_b_ud + loss_b_cs) / 7

    print(f"Test ckpt: {args.ckpt_path}")
    print(f"{split} Acc: DC:{acc_dc}, D9:{acc_d9}, D4:{acc_d4}, CC:{acc_cc}, "
          f"UD:{acc_ud}, LR:{acc_lr}, CS:{acc_cs},  ALL: {acc}, ")
    print(f"{split} loss_a: DC:{loss_a_dc:.4f}, D9:{loss_a_d9:.4f}, D4:{loss_a_d4:.4f}, CC:{loss_a_cc:.4f}, "
          f"UD:{loss_a_ud:.4f}, LR:{loss_a_lr:.4f}, CS:{loss_a_cs:.4f}, ALL:  {loss_a:.4f}")
    print(f"{split} loss_b: DC:{loss_b_dc:.4f}, D9:{loss_b_d9:.4f}, D4:{loss_b_d4:.4f}, CC:{loss_b_cc:.4f}, "
          f"UD:{loss_b_ud:.4f}, LR:{loss_b_lr:.4f}, CS:{loss_b_cs:.4f}, ALL:  {loss_b:.4f}")

    prediction_file = open(os.path.join(args.output_path, split + ".csv"), 'w')
    writer = csv.writer(prediction_file)
    writer.writerow(["acc_dc", "acc_d9", "acc_d4", "acc_cc", "acc_lr", "acc_ud", "acc_cs", "acc"])
    writer.writerow([acc_dc, acc_d9, acc_d4, acc_cc, acc_lr, acc_ud, acc_cs, acc])
    writer.writerow(["loss_a_dc", "loss_a_d9", "loss_a_d4", "loss_a_cc", "loss_a_lr", "loss_a_ud", "loss_a_cs", "loss_a"])
    writer.writerow([loss_a_dc, loss_a_d9, loss_a_d4, loss_a_cc, loss_a_lr, loss_a_ud, loss_a_cs, loss_a])
    writer.writerow(["loss_b_dc", "loss_b_d9", "loss_b_d4", "loss_b_cc", "loss_b_lr", "loss_b_ud", "loss_b_cs", "loss_b"])
    writer.writerow([loss_b_dc, loss_b_d9, loss_b_d4, loss_b_cc, loss_b_lr, loss_b_ud, loss_b_cs, loss_b])
    prediction_file.close()


def test_config(config_dataset, split, config):

    global loss_a_metric, loss_b_metric, acc_metric

    loss_a_metric.reset_state()
    loss_b_metric.reset_state()
    acc_metric.reset_state()

    prediction_file = open(os.path.join(args.output_path, split + "_" +config + ".csv"), 'w')
    writer = csv.writer(prediction_file)
    writer.writerow(["item", "target", "predict", "correctness",
                     "Progression-Number", "Progression-Position", "Arithmetic-Number", "Arithmetic-Position",
                     "Distribute_Three-Number", "Distribute_Three-Position", "Constant-Number/Position",
                     "Constant-Type", "Progression-Type", "Distribute_Three-Type", "Constant-Size", "Arithmetic-Size",
                     "Distribute_Three-Size", "Progression-Size", "Constant-Color", "Arithmetic-Color",
                     "Distribute_Three-Color", "Progression-Color"])

    pbar = tqdm(config_dataset)
    for images, targets, items, meta_targets in pbar:

        pred, correctness = tf_func_test(images, targets)

        for i, j, k, l, m in zip(items.numpy(), targets.numpy(), pred.numpy(), correctness.numpy(), meta_targets.numpy()):
            writer.writerow([i, j, k, l] + m.tolist())

    prediction_file.close()
    pbar.close()

    return loss_a_metric.result().numpy(), loss_b_metric.result().numpy(), acc_metric.result().numpy()


@tf.function(jit_compile = True)
def tf_func_test(images, targets):

    target_one_hots = tf.one_hot(indices = targets, depth = 8, on_value = 1, off_value = 0, axis = -1, dtype = tf.int64)

    logits_a, logits_b = model(images, training = False)
    logits = logits_a + logits_b
    loss_a = loss_fn(y_true = target_one_hots, y_pred = logits_a)
    loss_b = loss_fn(y_true = target_one_hots, y_pred = logits_b)

    loss_a_metric.update_state(loss_a)
    loss_b_metric.update_state(loss_b)
    acc_metric.update_state(targets, logits)

    pred = tf.math.argmax(logits, axis = -1)
    correctness = tf.cast((targets == pred), dtype = tf.uint8)

    return pred, correctness
