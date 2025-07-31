import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, pre_re_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ASGTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

    def train(self, model, train_loader, val_loader):
        """Train ASG for GAR"""
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )

        # Loss
        crit = nn.CrossEntropyLoss()

        best_acc = 0
        best_epoch = 0

        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs}')

            for data, group_y, person_y, coord in pbar:
                data = data.to(self.device).float()
                group_y = group_y.to(self.device)
                person_y = person_y.to(self.device)
                coord = coord.to(self.device).float()

                outputs = model(data, coord, person_y)
                loss = crit(outputs, group_y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # com
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += group_y.size(0)
                train_correct += predicted.eq(group_y).sum().item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * train_correct / train_total:.2f}%'
                })

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, group_y, person_y, coord in val_loader:
                    data = data.to(self.device).float()
                    group_y = group_y.to(self.device)
                    person_y = person_y.to(self.device)
                    coord = coord.to(self.device).float()

                    outputs = model(data, coord, person_y)
                    loss = crit(outputs, group_y)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += group_y.size(0)
                    val_correct += predicted.eq(group_y).sum().item()

            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f'Epoch {epoch + 1}/{self.config.num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                model_save_path = f"{self.config.output_dir}/models/model.pt"
                torch.save(model.state_dict(), model_save_path)

            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"{self.config.output_dir}/models/checkpoint_epoch_{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'val_acc': val_acc,
                }, checkpoint_path)

        return best_acc

    def resume_training(self, model, train_loader, val_loader, checkpoint_path):

        if not os.path.exists(checkpoint_path):
            return None

        # Load
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']


        #training
        crit = nn.CrossEntropyLoss()
        best_epoch = start_epoch

        for epoch in range(start_epoch, self.config.num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs}')

            for data, group_y, person_y, coord in pbar:
                data = data.to(self.device).float()
                group_y = group_y.to(self.device)
                person_y = person_y.to(self.device)
                coord = coord.to(self.device).float()

                outputs = model(data, coord, person_y)
                loss = crit(outputs, group_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += group_y.size(0)
                train_correct += predicted.eq(group_y).sum().item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * train_correct / train_total:.2f}%'
                })

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, group_y, person_y, coord in val_loader:
                    data = data.to(self.device).float()
                    group_y = group_y.to(self.device)
                    person_y = person_y.to(self.device)
                    coord = coord.to(self.device).float()

                    outputs = model(data, coord, person_y)
                    loss = crit(outputs, group_y)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += group_y.size(0)
                    val_correct += predicted.eq(group_y).sum().item()

            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f'Epoch {epoch + 1}/{self.config.num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f"{self.config.output_dir}/models/model.pt")

        return best_acc


def compute_acc(predictions, y):
    return acc_score(y, predictions)


def compute_pre_re_f1(predictions, y, average='macro'):
    pre, re, f1, _ = pre_re_fscore_support(
        y, predictions, average=average, zero_division=0
    )
    return pre, re, f1


def compute_confusion_matrix(predictions, y, num_classes, save_path=None):
    cm = confusion_matrix(y, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticky=range(num_classes), yticky=range(num_classes))
    plt.title('Confusion Matrix')
    plt.xy('Predicted y')
    plt.yy('True y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return cm


def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    all_predictions = []
    all_y = []

    with torch.no_grad():
        for data, group_y, person_y, coord in data_loader:
            data = data.to(device)
            group_y = group_y.to(device)
            person_y = person_y.to(device)
            coord = coord.to(device)

            outputs = model(data, coord, person_y)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_y.extend(group_y.cpu().numpy())

    # Compute metrics
    acc = compute_acc(all_predictions, all_y)
    pre, re, f1 = compute_pre_re_f1(all_predictions, all_y)

    print(f"\nacc: {acc:.4f}, pre: {pre:.4f}, re: {re:.4f}, F1: {f1:.4f}")
    return {
        'acc': acc,
        'pre': pre,
        're': re,
        'f1': f1,
        'predictions': all_predictions,
        'y': all_y
    }


def cross_validation(data, group_y, person_y, coord, config, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
        train_data = data[train_idx]
        test_data = data[test_idx]
        train_group_y = group_y[train_idx]
        test_group_y = group_y[test_idx]
        train_person_y = person_y[train_idx]
        test_person_y = person_y[test_idx]
        train_coords = coord[train_idx]
        test_coords = coord[test_idx]

        val_size = int(0.2 * len(train_data))
        val_data = train_data[:val_size]
        val_group_y = train_group_y[:val_size]
        val_person_y = train_person_y[:val_size]
        val_coords = train_coords[:val_size]

        train_data = train_data[val_size:]
        train_group_y = train_group_y[val_size:]
        train_person_y = train_person_y[val_size:]
        train_coords = train_coords[val_size:]

        from data.dataset import GARSensorDataset
        train_dataset = GARSensorDataset(train_data, train_group_y, train_person_y, train_coords, config, is_train=True)
        val_dataset = GARSensorDataset(val_data, val_group_y, val_person_y, val_coords, config, is_train=False)
        test_dataset = GARSensorDataset(test_data, test_group_y, test_person_y, test_coords, config, is_train=False)

        break

    return train_dataset, val_dataset, test_dataset











