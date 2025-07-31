import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import os
import numpy as np
import random

from config import Config
from dataset import GARSensorDataset, load_dataset
from models.asg import ASG
from utils import ASGTrainer, evaluate_model, cross_validation


def setseed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def data_loaders(config):

    data, group_labels, person_labels, coord = load_dataset(config.data_dir, config)

    config.num_group_act = len(np.unique(group_labels))
    config.num_ind_actions = len(np.unique(person_labels))

    full_dataset = GARSensorDataset(data, group_labels, person_labels, coord, config, is_train=True)

    train_dataset, val_dataset, test_dataset = cross_validation(data, group_labels, person_labels, coord,config)

    #loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='ASG Training and Testing')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (required for test mode)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory path')

    args = parser.parse_args()

    setseed(42)

    #configuration
    config = Config()
    config.data_dir = args.data_dir

    os.makedirs(f"{config.output_dir}/models", exist_ok=True)
    os.makedirs(f"{config.output_dir}/results", exist_ok=True)

    # loaders
    train_loader, val_loader, test_loader = data_loaders(config)

    # initialize
    model = ASG(config).to(config.device)
    trainer = ASGTrainer(config)

    if args.mode == 'train':

        best_acc = trainer.train(model, train_loader, val_loader)
        model.load_state_dict(torch.load(f"{config.output_dir}/models/model.pt"))
        results = evaluate_model(model, test_loader, config.device, config.num_group_act)

        import json
        with open(f"{config.output_dir}/results/training_results.json", 'w') as f:
            json.dump({
                'acc': float(best_acc),
                'pre': results['pre'],
                're': results['re'],
                'f1': results['f1'],
                'acc': results['acc'],
            }, f, indent=2)

    elif args.mode == 'test':

        if not args.model_path:
            return

        if not os.path.exists(args.model_path):
            return

        # load
        model.load_state_dict(torch.load(args.model_path))

        # Evaluate
        results = evaluate_model(model, test_loader, config.device, config.num_group_act)

        # results
        import json
        with open(f"{config.output_dir}/results/test_results.json", 'w') as f:
            json.dump({
                'model_path': args.model_path,
                'pre': results['pre'],
                're': results['re'],
                'f1': results['f1'],
                'acc': results['acc'],
            }, f, indent=2)

        print(f"test results: {config.output_dir}/results/test_results.json")


if __name__ == "__main__":
    main()