from CPCNet import args
from CPCNet.run.train import train
from CPCNet.run.test import test
from CPCNet.run.train_multi_path import train_multi_path
from CPCNet.run.test_multi_path import test_multi_path


def main():
    if "train" == args.mode:
        train()
    elif "test" == args.mode:
        test()
    elif "train_multi_path" == args.mode:
        train_multi_path()
    elif "test_multi_path" == args.mode:
        test_multi_path()
    else:
        raise ValueError("Unknown mode!")


if __name__ == "__main__":
    main()
