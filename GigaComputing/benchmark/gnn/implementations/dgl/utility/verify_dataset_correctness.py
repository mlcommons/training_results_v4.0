import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/data/igbh")
    parser.add_argument("--dataset_size", type=str, default="full")

    args = parser.parse_args()
    size = args.dataset_size
    path = args.data_dir

    # check train_idx and val_idx size correctness

    expected_size = {'tiny':100000, 'small':1000000, 'medium':10000000, 'large':100000000, 'full':157675969}[size]
    expected_train_size = int(expected_size * 0.6)
    expected_val_size = int(expected_size * 0.005)

    train_idx = torch.load(f"{path}/{size}/processed/train_idx.pt")
    assert train_idx.shape[0] == expected_train_size, f"Expecting {expected_train_size} train indices, found {train_idx.shape[0]}"

    val_idx = torch.load(f"{path}/{size}/processed/val_idx.pt")
    assert val_idx.shape[0] == expected_val_size, f"Expecting {expected_val_size} val indices, found {val_idx.shape[0]}"
