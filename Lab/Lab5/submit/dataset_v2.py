import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import torch.utils.data as tu

gpu = torch.device("cuda")
cpu = torch.device("cpu")
default_transform = transforms.Compose([
    transforms.ToTensor(),
])


class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='trian', transform=default_transform, frame_num=12, frame_in=2):
        assert mode == 'train' or mode == 'test' or mode == 'validate' or mode == 'trial'
        self.transform = transform
        # self.seed = args.seed
        self.seed_is_set = False
        self.frame_num = frame_num
        self.frame_in_num = frame_in
        self.batches_len = 0
        self.dirpath = '%s/%s' % (args.data_root, mode)
        self.dirs = []
        # self.dir_index = 0
        for dir in os.listdir(self.dirpath):
            for sub_dir in os.listdir('%s/%s' % (self.dirpath, dir)):
                self.dirs.append('%s/%s/%s' % (self.dirpath, dir, sub_dir))
                # print('%s/%s/%s' % (self.dirpath, dir, sub_dir))

        # print(self.dirpath, self.batches, self.video_file, self.data, self.frames_name, self.csv_name, sep='\n')
        # raise NotImplementedError

    def set_seed(self, seed):
        # print(self.seed_is_set)
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
        # raise StopIteration()

    def __len__(self):
        return len(self.dirs)
        # raise NotImplementedError

    def get_seq(self, index):
        frames = torch.tensor([])  # .to(gpu, dtype=torch.float)
        for frame in range(self.frame_num):
            path = '%s/%d.png' % (self.dirs[index], frame)
            img = Image.open(path)
            img_tensor = self.transform(img)
            img_tensor = img_tensor  # .to(gpu, dtype=torch.float)
            frames = torch.cat((frames, img_tensor))
        # frame_len = int(frames.size()[0] / (self.frame_num * 3))
        # print('frames: ', frames.size())
        frames = torch.reshape(frames, (self.frame_num, 3, 64, 64))  # 12, 3, 64, 64
        # frames_in = frames[:, :self.frame_in_num, :, :, :]  # 2, 3, 64, 64
        # frames_out = frames[:, self.frame_in_num:, :, :, :]  # 10, 3, 64, 64
        # print('frames total: ', frames.size())  # , 'frames_in: ', frames_in.size(), 'frames_out: ', frames_out.size())
        return frames
        # raise NotImplementedError

    def get_csv(self, index):
        excel_action = torch.tensor([])  # .to(gpu, dtype=torch.float)
        excel_end = torch.tensor([])  # .to(gpu, dtype=torch.float)
        excel_action_total = torch.tensor([])  # .to(gpu, dtype=torch.float)
        excel_end_total = torch.tensor([])  # .to(gpu, dtype=torch.float)
        excel_action = get_text(self.dirs[index], 'actions.csv', self.frame_num)
        excel_action_total = torch.cat((excel_action_total, excel_action))
        excel_end = get_text(self.dirs[index], 'endeffector_positions.csv', self.frame_num)
        excel_end_total = torch.cat((excel_end_total, excel_end))
        # print('before action total: ', excel_action_total.size(), 'end: ', excel_end_total.size())
        excel_total = torch.cat((excel_action_total, excel_end_total), 1)
        # print(excel_total.size())
        action_len = int(excel_action_total.size()[0] / self.frame_num)
        # self.batches_len = action_len
        excel_total = torch.reshape(excel_total, (self.frame_num, 7))
        # excel_total_in = excel_total[:, :self.frame_in_num, :]
        # excel_total_out = excel_total[:, self.frame_in_num:, :]
        # print(excel_total.size())
        # excel_action_total = torch.reshape(excel_action_total, (action_len, 30, 4))
        # excel_end_total = torch.reshape(excel_end_total, (action_len, 30, 3))
        # print('action total: ', excel_action_total.size(), 'end: ', excel_end_total.size())
        return excel_total  # excel_action_total[index], excel_end_total[index]

    def __getitem__(self, index):
        # print(index)
        self.set_seed(index)
        seq = self.get_seq(index)
        cond = self.get_csv(index)
        # print('seq: ', seq.size())
        # print('cond: ', cond[0].size(), cond[1].size())
        return seq, cond


def get_text(dirpath, filename, frame_num):
    excel = torch.tensor([])  # .to(gpu, dtype=torch.float)
    row_num = 0
    with open('%s/%s' % (dirpath, filename), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            # print(row_num)
            if row_num < frame_num:
                text = row[0].split(",")  # [', '.join(row)]
                text_arr = np.array([text], dtype=float)
                # print(text_arr, type(text_arr), len(text_arr))
                text_tns = torch.Tensor(text_arr)
                text_tns = text_tns  # .to(gpu, dtype=torch.float32)
                excel = torch.cat((excel, text_tns))
                row_num += 1
            else:
                break
    return excel


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
#     parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
#     parser.add_argument('--batch_size', default=12, type=int, help='batch size')
#     parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
#     parser.add_argument('--model_dir', default='', help='base directory to save logs')
#     parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
#     parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
#     parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
#     parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
#     parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
#     parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
#     parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
#     parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
#     parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
#     parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
#     parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing (if use cyclical mode)')
#     parser.add_argument('--seed', default=1, type=int, help='manual seed')
#     parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
#     parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
#     parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
#     parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
#     parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
#     parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
#     parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
#     parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
#     parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
#     parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
#     parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
#     parser.add_argument('--cuda', default=False, action='store_true')
#     parser.add_argument('--cond_size', type=int, default=7, help='dimension of condition')
#     args = parser.parse_args([])
#     return args


#
# args = parse_args()
# path = parse_args().data_root  # default='./data/processed_data'  "./data/Processed_data/"  "traj_512_to_767.tfrecords/0/"
# mode = 'trial'
# data_path = path + mode + '/'
#
# data = bair_robot_pushing_dataset(args, mode=mode)
# # print(cond)
#
# count = 0
# print('start loop')
# for x, y in data:
#     # print('data: ', x)
#     print(count, x.size(), y.size(), len(data))
#     count += 1
#     break
#
#
# batches = 4
# train_data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
# i=0
#
# for x, y in train_data_loader:
#     # print(x, y)
#     print('batch: ', i, x.size(), y.size(), len(data))
#     i += 1
#
# args = parse_args()
# train_data = bair_robot_pushing_dataset(args, 'trial')
# # train_data = torch.tensor(train_data, device='cpu')
# train_loader = DataLoader(train_data,
#                           batch_size=4,
#                           shuffle=True,
#                           drop_last=True)
# train_iterator = iter(train_loader)
#
# i = 0
# print(args.epoch_size)
# for _ in range(args.epoch_size):
#     try:
#         seq, cond = next(train_iterator)  # sequence of frames, condition from csv
#         print(i, seq.size(), cond.size())
#         i += 1
#     except StopIteration:
#         print(i)
#         i += 1
#         train_iterator = iter(train_loader)

