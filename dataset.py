import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from scipy import stats
import random
import os


class GARSensorDataset(Dataset):
    def __init__(self, data, group_y, person_y, coord, config, is_train=True):
        self.data = torch.FloatTensor(data)
        self.group_y = torch.LongTensor(group_y)
        self.person_y = torch.LongTensor(person_y)
        self.coord = torch.FloatTensor(coord)
        self.config = config
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]  # (5, N_TIME_STEPS, 12)
        group_y = self.group_y[idx]
        person_y = self.person_y[idx]  # (5,)
        sample_coords = self.coord[idx]  # (5, 2)

        win_data = self.sliding_win(sample_data)

        if not self.is_train:
            win_data = self.asyn(win_data)

        return win_data, group_y, person_y, sample_coords

    def sliding_win(self, data):
        ind, time_steps, ch = data.shape
        win_size = self.config.sub_win_len
        num_win = self.config.num_time_win

        win_data = torch.zeros(ind, num_win, ch, win_size)

        for i in range(num_win):
            start_idx = i * win_size
            end_idx = start_idx + win_size
            if end_idx <= time_steps:
                win_data[:, i, :, :] = data[:, start_idx:end_idx, :].transpose(1, 2)
            else:
                actual_size = time_steps - start_idx
                if actual_size > 0:
                    win_data[:, i, :, :actual_size] = data[:, start_idx:, :].transpose(1, 2)
                    win_data[:, i, :, actual_size:] = data[:, -1:, :].transpose(1, 2).expand(-1, -1,
                                                                                                  win_size - actual_size)

        return win_data

    def asyn(self, data):
        ind, time_win, ch, win_size = data.shape
        perturbed_data = data.clone()

        num_async = random.randint(1, 4)
        async_ind = random.sample(range(ind), num_async)

        for ind in async_ind:
            delay = random.choice(self.config.temporal_delay[1:])  # Exclude 0 delay
            if delay < time_win:
                shifted_data = torch.roll(perturbed_data[ind], shifts=delay, dims=0)
                perturbed_data[ind] = shifted_data

        return perturbed_data


def load_dataset(data_dir, config):
    segments, y_p, y_g, coord = [], [], [], []

    columns = [
        "pocket_acc_x", "pocket_acc_y", "pocket_acc_z",
        "pocket_gyro_x", "pocket_gyro_y", "pocket_gyro_z",
        "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
        "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z",
        "coordinate_x", "coordinate_y",
        "y_person", "y_group"
    ]

    for i in range(10):  # Assuming 10 files
        file_path = os.path.join(data_dir, f"act{i}.txt")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, sep="\t", header=None, names=columns)

        N_TIME_STEPS = config.win_len
        STRIDE = N_TIME_STEPS // 2

        for start in range(0, len(df) - N_TIME_STEPS, STRIDE):
            end = start + N_TIME_STEPS
            win = df.iloc[start:end]

            # Extract  feas
            fea_cols = [
                "pocket_acc_x", "pocket_acc_y", "pocket_acc_z",
                "pocket_gyro_x", "pocket_gyro_y", "pocket_gyro_z",
                "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
                "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z"
            ]
            feas = win[fea_cols].values

            y_p = win["y_person"].values[0]
            y_g = stats.mode(win["y_group"].values, keepdims=True)[0][0]
            coords = win[["coordinate_x", "coordinate_y"]].values

            segments.append(feas)
            y_p.append(y_p)
            y_g.append(y_g)
            coord.append(coords)

    # convert
    segments = np.array(segments).astype(np.float32)
    y_p = np.array(y_p)
    y_g = np.array(y_g)
    coord = np.array(coord).astype(np.float32)

    grouped_data, grouped_y_p, grouped_y_g, grouped_coord = [], [], [], []

    for i in range(0, len(segments) - 4, 5):
        group_data = segments[i:i + 5]
        group_y_p = y_p[i:i + 5]
        group_y_g = y_g[i + 2]
        group_coords = coord[i:i + 5]

        avg_coords = np.mean(group_coords, axis=1)

        grouped_data.append(group_data)
        grouped_y_p.append(group_y_p)
        grouped_y_g.append(group_y_g)
        grouped_coord.append(avg_coords)

    return (
        np.array(grouped_data),
        np.array(grouped_y_g),
        np.array(grouped_y_p),
        np.array(grouped_coord)
    )


