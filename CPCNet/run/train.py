import tensorflow as tf

from os.path import join, exists
from os import makedirs
from datetime import datetime
from tqdm import tqdm

from .run_utils import get_args_str
from .Logger import Logger

from .. import args
from ..models.CPCNet import CPCNet_LP, CPCNet_UP, CPCNet_UC, CPCNet_LC

from ..data.RAVEN import get_RAVEN_dataset, read_RAVEN_file
from ..data.const import DC_index, D9_index, D4_index, CC_index, UD_index, LR_index, CS_index

model = None
loss_fn = None
optimizer = None

train_dataset = None
val_dataset = None

avg_loss = None
avg_acc = None
avg_acc_dc = None
avg_acc_d9 = None
avg_acc_d4 = None
avg_acc_cc = None
avg_acc_lr = None
avg_acc_ud = None
avg_acc_cs = None


def train():
    """
    Train from scratch or from checkpoints.
    """

    global model, loss_fn, optimizer, train_dataset, val_dataset, avg_loss, \
        avg_acc, avg_acc_dc, avg_acc_d9, avg_acc_d4, avg_acc_cc, avg_acc_lr, avg_acc_ud, avg_acc_cs

    if "CPCNet_LP" == args.model:
        model = CPCNet_LP(args.image_size, args.channels)
    elif "CPCNet_UP" == args.model:
        model = CPCNet_UP(args.image_size, args.channels)
    elif "CPCNet_UC" == args.model:
        model = CPCNet_UC(args.image_size, args.channels)
    elif "CPCNet_LC" == args.model:
        model = CPCNet_LC(args.image_size, args.channels)
    else:
        raise ValueError("Unknown model: {}".format(args.model))

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True, reduction = "sum_over_batch_size")

    train_dataset = get_RAVEN_dataset(dataset_root = args.dataset_path, split = "train", spatial_configs = args.train_configs, gain = args.train_configs_proportion)
    train_dataset = train_dataset.shuffle(buffer_size = len(train_dataset)).map(map_func = read_RAVEN_file, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)
    val_dataset = get_RAVEN_dataset(args.dataset_path, "val", args.train_configs).map(map_func = read_RAVEN_file, num_parallel_calls = args.num_workers).batch(batch_size = args.batch_size)

    avg_loss = tf.keras.metrics.Mean()
    avg_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    avg_acc_dc = tf.keras.metrics.SparseCategoricalAccuracy()
    avg_acc_d9 = tf.keras.metrics.SparseCategoricalAccuracy()
    avg_acc_d4 = tf.keras.metrics.SparseCategoricalAccuracy()
    avg_acc_cc = tf.keras.metrics.SparseCategoricalAccuracy()
    avg_acc_lr = tf.keras.metrics.SparseCategoricalAccuracy()
    avg_acc_ud = tf.keras.metrics.SparseCategoricalAccuracy()
    avg_acc_cs = tf.keras.metrics.SparseCategoricalAccuracy()

    ckpt_path = join(args.output_path, "model_ckpt")
    log_path = join(args.output_path, "log")

    # train from checkpoints
    if exists(ckpt_path) and exists(log_path):

        train_from_checkpoint = True

        optimizer = tf.keras.optimizers.Adam()
        ckpt = tf.train.Checkpoint(optimizer = optimizer, model = model)
        manager = tf.train.CheckpointManager(ckpt, directory = ckpt_path, max_to_keep = 50)
        checkpoint_status = ckpt.restore(manager.latest_checkpoint)

        start_epoch = ckpt.save_counter.numpy()

        logger = Logger(log_path)

        if args.learning_rate is not None:
            optimizer.learning_rate = args.learning_rate

    # train from scratch
    else:
        train_from_checkpoint = False
        checkpoint_status = None

        args.output_path = join(args.output_path, datetime.now().strftime("%G_%m_%d_%H_%M_%S_") + get_args_str(args))
        makedirs(args.output_path)

        ckpt_path = join(args.output_path, "model_ckpt")
        makedirs(ckpt_path)

        optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate, weight_decay = args.weight_decay, epsilon = 1e-8)

        ckpt = tf.train.Checkpoint(optimizer = optimizer, model = model)
        manager = tf.train.CheckpointManager(ckpt, directory = ckpt_path, max_to_keep = 50)

        start_epoch = 0

        log_path = join(args.output_path, "log")
        makedirs(log_path)
        logger = Logger(log_path)

    end_epoch = start_epoch + args.epoch_num
    for epoch in range(start_epoch, end_epoch):

        print("**************************************** Start Epoch {} / {} ***************************************".format(epoch, end_epoch))
        train_epoch()
        loss = {'train': avg_loss.result().numpy()}
        acc = {'train': avg_acc.result().numpy()}
        train_config_acc = {"dc": avg_acc_dc.result().numpy(), "d9": avg_acc_d9.result().numpy(), "d4": avg_acc_d4.result().numpy(),
                            "cc": avg_acc_cc.result().numpy(), "ud": avg_acc_ud.result().numpy(), "lr": avg_acc_lr.result().numpy(),
                            "cs": avg_acc_cs.result().numpy()}
        logger.add_scalars("/train_config_acc", train_config_acc, epoch)

        validate_epoch()
        loss['val'] = avg_loss.result().numpy()
        acc['val'] = avg_acc.result().numpy()
        val_config_acc = {"dc": avg_acc_dc.result().numpy(), "d9": avg_acc_d9.result().numpy(), "d4": avg_acc_d4.result().numpy(),
                          "cc": avg_acc_cc.result().numpy(), "ud": avg_acc_ud.result().numpy(), "lr": avg_acc_lr.result().numpy(),
                          "cs": avg_acc_cs.result().numpy()}
        logger.add_scalars("/val_config_acc", val_config_acc, epoch)

        logger.add_scalars('/avg_loss', loss, epoch)
        logger.add_scalars('/acc', acc, epoch)
        logger.add_scalar('/lr', optimizer.learning_rate.numpy(), epoch)

        saved_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(epoch, saved_path))

        print(f"Train/Val: \nLoss: {loss['train']:.4f} / {loss['val']:.4f} \nAcc: {acc['train']:.4f} / {acc['val']:.4f}")
        for cfg in train_config_acc.keys():
            print(cfg, f": {train_config_acc.get(cfg):.4f} / {val_config_acc.get(cfg):.4f}")

    if train_from_checkpoint:
        checkpoint_status.assert_existing_objects_matched()
        checkpoint_status.assert_consumed()

    logger.close()


def train_epoch():
    """
    traverse the training set.
    """

    global train_dataset, avg_loss, avg_acc, avg_acc_dc, avg_acc_d9, avg_acc_d4, avg_acc_cc, avg_acc_lr, avg_acc_ud, avg_acc_cs

    avg_loss.reset_state()
    avg_acc.reset_state()
    avg_acc_dc.reset_state()
    avg_acc_d9.reset_state()
    avg_acc_d4.reset_state()
    avg_acc_cc.reset_state()
    avg_acc_lr.reset_state()
    avg_acc_ud.reset_state()
    avg_acc_cs.reset_state()

    pbar = tqdm(train_dataset)
    for images, targets, configs in pbar:

        loss = tf_func(images, targets, configs)

        pbar.set_postfix(loss = f"{loss.numpy():.4f}")

    pbar.close()


@tf.function
def tf_func(images, targets, configs):
    """
    tf.function wrapper.
    """

    target_one_hots = tf.one_hot(indices = targets, depth = 8, on_value = 1, off_value = 0, axis = -1, dtype = tf.int64)

    with tf.GradientTape() as tape:
        logits = model(images, training = True)
        loss = loss_fn(y_true = target_one_hots, y_pred = logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    avg_loss.update_state(loss)
    avg_acc.update_state(targets, logits)

    dc_indices = (DC_index == configs)
    if tf.math.reduce_any(dc_indices):
        avg_acc_dc.update_state(targets[dc_indices], logits[dc_indices])
    d9_indices = (D9_index == configs)
    if tf.math.reduce_any(d9_indices):
        avg_acc_d9.update_state(targets[d9_indices], logits[d9_indices])
    d4_indices = (D4_index == configs)
    if tf.math.reduce_any(d4_indices):
        avg_acc_d4.update_state(targets[d4_indices], logits[d4_indices])
    cc_indices = (CC_index == configs)
    if tf.math.reduce_any(cc_indices):
        avg_acc_cc.update_state(targets[cc_indices], logits[cc_indices])
    lr_indices = (LR_index == configs)
    if tf.math.reduce_any(lr_indices):
        avg_acc_lr.update_state(targets[lr_indices], logits[lr_indices])
    ud_indices = (UD_index == configs)
    if tf.math.reduce_any(ud_indices):
        avg_acc_ud.update_state(targets[ud_indices], logits[ud_indices])
    cs_indices = (CS_index == configs)
    if tf.math.reduce_any(cs_indices):
        avg_acc_cs.update_state(targets[cs_indices], logits[cs_indices])

    return loss


def validate_epoch():

    global val_dataset, avg_loss, avg_acc, avg_acc_dc, avg_acc_d9, avg_acc_d4, avg_acc_cc, avg_acc_lr, avg_acc_ud, avg_acc_cs

    avg_loss.reset_state()
    avg_acc.reset_state()
    avg_acc_dc.reset_state()
    avg_acc_d9.reset_state()
    avg_acc_d4.reset_state()
    avg_acc_cc.reset_state()
    avg_acc_lr.reset_state()
    avg_acc_ud.reset_state()
    avg_acc_cs.reset_state()

    pbar = tqdm(val_dataset)
    for images, targets, configs in pbar:

        tf_func_validate(images, targets, configs)

    pbar.close()


@tf.function(jit_compile = True)
def tf_func_validate(images, targets, configs):

    target_one_hots = tf.one_hot(indices = targets, depth = 8, on_value = 1, off_value = 0, axis = -1, dtype = tf.int64)

    logits = model(images, training = False)
    loss = loss_fn(y_true = target_one_hots, y_pred = logits)

    avg_loss.update_state(loss)
    avg_acc.update_state(targets, logits)

    dc_indices = (DC_index == configs)
    if tf.math.reduce_any(dc_indices):
        avg_acc_dc.update_state(targets[dc_indices], logits[dc_indices])
    d9_indices = (D9_index == configs)
    if tf.math.reduce_any(d9_indices):
        avg_acc_d9.update_state(targets[d9_indices], logits[d9_indices])
    d4_indices = (D4_index == configs)
    if tf.math.reduce_any(d4_indices):
        avg_acc_d4.update_state(targets[d4_indices], logits[d4_indices])
    cc_indices = (CC_index == configs)
    if tf.math.reduce_any(cc_indices):
        avg_acc_cc.update_state(targets[cc_indices], logits[cc_indices])
    lr_indices = (LR_index == configs)
    if tf.math.reduce_any(lr_indices):
        avg_acc_lr.update_state(targets[lr_indices], logits[lr_indices])
    ud_indices = (UD_index == configs)
    if tf.math.reduce_any(ud_indices):
        avg_acc_ud.update_state(targets[ud_indices], logits[ud_indices])
    cs_indices = (CS_index == configs)
    if tf.math.reduce_any(cs_indices):
        avg_acc_cs.update_state(targets[cs_indices], logits[cs_indices])
