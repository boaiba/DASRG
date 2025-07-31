import torch


class Config:
    def __init__(self):
        # Dataset parameters
        self.dataset_name = "Garsensors"#UT-Data-gar
        self.win_len = 300
        self.overlap = 0.5
        self.sub_win_len = 50
        self.num_time_win = 6
        self.sensor_ch = 12  # pocket(6) + wrist(6)
        self.num_ind = 5
        self.num_group_act = 10
        self.num_ind_actions = 11

        # Model parameters
        self.cnn_filters = [64, 128, 256]
        self.lstm_hid_dim = 128
        self.fea_dim = 256
        self.dropout = 0.2
        self.num_heads = 3

        # ASG specific parameters
        self.temporal_delays = [0, 1, 2, 3]
        self.ch_ratio = 6
        self.dist_thd = 5.0
        self.proba_thd = 0.8

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 30
        self.weight_decay = 1e-5


        # Device and paths
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = "path/data"
        self.output_dir = "./outputs"

