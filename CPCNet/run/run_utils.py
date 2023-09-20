import os.path


def get_args_str(args):

    if "train" in args.mode:
        args_str = args.mode + "_" + args.model + "_ch_" + str(args.channels) + "_bs_" + str(args.batch_size)
        args_str += "_dp_" + os.path.basename(os.path.normpath(args.dataset_path))
        args_str += "_ls_" + args.loss
        args_str += "_sd_" + str(args.seed)
    elif "test" == args.mode:
        args_str = args.mode + "_" + args.model + "_"
    elif "hyper_param_search" == args.mode:
        args_str = args.mode + "_" + args.model + "_en_" + str(args.epoch_num) \
                   + "_bs_" + str(args.batch_size) + "_nw_" + str(args.num_workers) + "_s_" + str(args.seed)
        args_str += "_lr_" + str(args.learning_rate_range) if args.learning_rate_range is not None else "_lr_" + str(args.learning_rate)
        args_str += "_wd_" + str(args.weight_decay_range) if args.weight_decay_range is not None else "_wd_" + str(args.weight_decay)
        args_str += "_nt_" + str(args.n_trials)
        args_str += "_nst_" + str(args.n_startup_trials) if args.n_startup_trials is not None else ""
        args_str += "_nwt_" + str(args.n_warmup_steps) if args.n_warmup_steps is not None else ""
        args_str += "_is_" + str(args.interval_steps) if args.interval_steps is not None else ""
        args_str += "_"
    elif "train_number_matrix" == args.mode:
        args_str = args.mode + "_" + args.model +"_lr_" + str(args.learning_rate) + "_bs_" + str(args.batch_size) + "_nw_" + str(args.num_workers)
    else:
        raise ValueError("Unknown mode: {}.".format(args.mode))

    return args_str
