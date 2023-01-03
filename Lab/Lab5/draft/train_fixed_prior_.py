import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, finn_eval_seq, plot_output, pred_rec, pred_norm, reparameter
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import gc

# print(torch.cuda.memory_stats())
# print(torch.cuda.memory_summary())
# print(torch.cuda.memory_allocated())
# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()
#
#     torch.cuda.empty_cache()
#
#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)
#
#     print("GPU Usage after emptying the cache")
#     gpu_usage()
#
#
# free_gpu_cache()

torch.backends.cudnn.benchmark = True  # True
# torch.backends.cudnn.enabled = False  # addition
# print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
# print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name(0))


def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')  # 0.002
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=2, type=int, help='size of every batch')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')  # 'adam', 'rmsprop', 'sgd'
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')  # 300
    parser.add_argument('--epoch_size', type=int, default=200, help='epoch size')  # 600
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=100, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing (if use cyclical mode)')
    parser.add_argument('--kl_beta', type=float, default=0, help='kl beta/weight for each epoch')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=10, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--cond_size', type=int, default=7, help='dimension of condition')
    parser.add_argument('--var', type=float, default=0.0, help='var')
    parser.add_argument('--mean', type=float, default=1.0, help='mean')
    args = parser.parse_args([])
    return args


def train(x, cond, modules, optimizer, args, device):
    criterion = nn.MSELoss()
    # modules['frame_predictor'].zero_grad()
    # modules['posterior'].zero_grad()
    # modules['encoder'].zero_grad()
    # modules['decoder'].zero_grad()
    optimizer.zero_grad()
    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False
    # print('use_teacher_forcing: ', use_teacher_forcing, end=' ')
    # print(encoder)
    # frame_predictor = lstm(args.g_dim + args.z_dim + args.cond_size, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
    # posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
    # encoder = vgg_encoder(args.g_dim)
    # decoder = vgg_decoder(args.g_dim)
    decoder_output_total = torch.tensor(x[0].unsqueeze(0)).to(device)
    # encoder_output_total = []
    # for i in range(args.n_past):
    encoder_output_total = [modules['encoder'](x[0])]  # h5.view(-1, self.dim), [h1, h2, h3, h4]
    # print('encoder_output_total: ', encoder_output_total[0][0].requires_grad)
    # print('seq_x: ', x.size(), 'seq_x[0]: ', x[0].size(), 'use_teacher_forcing: ', use_teacher_forcing)
    # print('input: ', x.requires_grad)

    for i in range(1, args.n_past + args.n_future):  # frames_size
        if i == 1 or use_teacher_forcing:  # use_teacher_forcing
            # print('type: ', type(encoder_output_total), type(modules['encoder'](x[i])))
            encoder_output_total.append(modules['encoder'](x[i]))
            # for name, para in modules['encoder'].named_parameters():
            #     print('{}: {}'.format(name, para.c5.weight.grad))
            # print('weight: ', modules['encoder'].c5[0].weight.grad)
            # print('encoder_output_total: ', len(encoder_output_total), 'encoder_output_total[0]: ', len(encoder_output_total[0]))
            if args.last_frame_skip or i < args.n_past:
                encoder_output_past, remain = encoder_output_total[i - 1]  # h5, [h1, h2, h3, h4]
            else:
                encoder_output_past = encoder_output_total[i - 1][0]
            z, mean, logvar = modules['posterior'](encoder_output_total[i][0])
            # print('z: ', z.requires_grad)
            # print('z: {}, mu: {}, logvar: {}, z_max: {}, z_min: {}'.format(z.size(), mu.size(), logvar.size(), z.max(), z.min()))
        else:
            encoder_output_past = modules['encoder'](decoder_output_total[i - 1])[0]  # h5, [h1, h2, h3, h4] take h5 only
            mean, logvar = args.mean, torch.tensor(args.var).to(device)
            z = reparameter(mean, logvar, (args.batch_size, args.z_dim), device)
            # print('z: {}, z_max: {}, z_min: {}'.format(z.size(), z.max(), z.min()))

        lstm_input = torch.cat((encoder_output_past, cond[i - 1], z), 1)
        # print('lstm_input_size: ', lstm_input.size(), 'cond[i - 1]: ', cond[i - 1].size())
        lstm_output = modules['frame_predictor'](lstm_input)
        # print('lstm_output: ', lstm_output.requires_grad)
        # print('rnn_output: ', lstm_output.size())
        decoder_output = modules['decoder']((lstm_output, remain))
        # print('decoder_output: ', decoder_output.requires_grad)
        # print('output: ', decoder_output.size())
        decoder_output_total = torch.cat((decoder_output_total, decoder_output.unsqueeze(0)))
        # print('decoder_output_total: ', decoder_output_total.size(), 'decoder_output_total[i - 1, 0]: ', decoder_output_total[i - 1, 0].size())
        mse += criterion(decoder_output, x[i])
        kld += kl_criterion(mean, logvar, args)
        # raise NotImplementedError

    loss = mse + kld * args.kl_beta
    # print('loss: {}, mse: {}, kld: {}, beta: {}'.format(loss, mse, kld, args.kl_beta))
    loss.backward()
    optimizer.step()
    # print('\r', i, 'weight_after: ', modules['encoder'].c5[0].weight[0, 0, 0, 0], end=' ')
    # print('grad_after: ', modules['encoder'].c5[0].weight.grad[0, 0, 0], end=' ')
    # return loss, mse, kld
    # return_loss = loss.detach().cpu().numpy() / (args.n_past + args.n_future)
    # return_mse = mse.detach().cpu().numpy() / (args.n_past + args.n_future)
    # return_kld = kld.detach().cpu().numpy() / (args.n_future + args.n_past)
    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)


class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.n_epoch = args.epoch_size
        self.period = self.n_epoch / args.kl_anneal_cycle  # 600/3 = 200
        self.ratio = args.kl_anneal_ratio / 2
        self.step = 1 / (self.period * self.ratio)  # 1/(200*0.5) = 0.01
        # self.start = 0.0
        # self.stop = 1.0
        # print('kl_annealing_initial, epoch: ', self.epoch)

    def update(self, cyclical):
        # beta_total = np.ones(self.n_epoch)  # initial 1
        # period = self.n_epoch / self.n_cycle
        # step = (self.stop - self.start) / (period * self.n_ratio)  # linear schedule
        # print('step: ', step)
        #
        # for cycle in range(self.n_cycle):
        #     beta, epoch = self.start, 0
        #     while beta <= self.stop and (int(epoch + cycle * period) < self.n_epoch):
        #         print('\r', int(epoch + cycle * period), self.n_epoch, beta, end=' ')
        #         beta_total[int(epoch + cycle * period)] = beta  # replace value when it is not 1
        #         beta += step
        #         epoch += 1
        # return beta_total
        if cyclical:
            self.epoch %= self.period
            beta = self.epoch * self.step
        else:
            beta = self.epoch * (self.step / 2)

        return 1 if beta > 1 else beta

    def get_beta(self, epoch, cyclical):
        self.epoch = epoch
        # self.epoch = epoch
        # beta_total = self.update()
        # beta = beta_total[self.epoch]
        # print('epoch: ', self.epoch, beta, max(beta_total))
        return self.update(cyclical)


def main():
    args = parse_args()
    fix_seed(args.seed)
    # args.epoch_size = 10
    args.tfr_decay_step = 0.02
    args.cuda = True
    #     args.batch_size = 2
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'

    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f' \
               % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    # print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))

    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------
    # print('batch_size: ', args.batch_size)
    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim + args.z_dim + args.cond_size, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)

    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)

    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)
    # print('modules[frame_predictor]: ', next(frame_predictor.parameters()).is_cuda)
    # print('modules[posterior]: ', next(posterior.parameters()).is_cuda)
    # print('modules[encoder]: ', next(encoder.parameters()).is_cuda)
    # print('modules[decoder]: ', next(decoder.parameters()).is_cuda)
    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'trial')  # train
    validate_data = bair_robot_pushing_dataset(args, 'trial')  # validate
    count = 0
    # print('os.cpu_count(): ', os.cpu_count())
    # for x_batch, y_batch in train_data:
    #     # print('data: ', x)
    #     print(count, x_batch.size(), y_batch[0].size(), y_batch[1].size(), len(train_data))
    #     count += 1
    #     break
    # for x_batch, y_batch in validate_data:
    #     # print('data: ', x)
    #     print(count, x_batch.size(), y_batch[0].size(), y_batch[1].size(), len(train_data))
    #     count += 1
    #     break
    # train_loader = DataLoader(train_data, batch_size=args.batch_size)
    # i=0
    # for x, y in train_loader:
    #     # print(x, y)
    #     print('batch: ', i, x.size())
    #     i += 1
    train_loader = DataLoader(train_data,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)  # num_workers=args.num_workers, pin_memory=True
    train_iterator = iter(train_loader)
    # validate_data = train_data + label
    validate_loader = DataLoader(validate_data,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=True)

    validate_iterator = iter(validate_loader)  # iter from loader, every element(frame_seq, csv_cond) from loader

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    # optimizer = args.optimizer(params, lr=args.lr, momentum=0.9)
    # frame_predictor_optimizer = args.optimizer(frame_predictor.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # posterior_optimizer = args.optimizer(posterior.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # encoder_optimizer = args.optimizer(encoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # decoder_optimizer = args.optimizer(decoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # print('modules[frame_predictor]: ', next(modules['decoder'].parameters()).is_cuda)
    # print('modules[posterior]: ', next(modules['decoder'].parameters()).is_cuda)
    # print('modules[encoder]: ', next(modules['decoder'].parameters()).is_cuda)
    # print('modules[decoder]: ', next(modules['decoder'].parameters()).is_cuda)
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    loss_total, mse_total, kld_total, beta_total, tfr_total, ave_psnr_total = [], [], [], [], [], []

    for epoch in range(start_epoch, start_epoch + niter):  # epoch mean niter number (total 300)
        args.kl_beta = kl_anneal.get_beta(epoch, args.kl_anneal_cyclical)
        # print('epoch_start: ', epoch)
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for _ in range(args.epoch_size):  # every niter run 600(epochs) times
            try:
                seq, cond = next(train_iterator)  # sequence of frames, condition from csv
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)

            # print('\r_: ', _, end=' ')
            seq, cond = seq.to(device), cond.to(device)
            seq.transpose_(0, 1)  # transpose batch and frame dimension
            cond.transpose_(0, 1)
            # args.iter_count = _
            loss, mse, kld = train(seq, cond, modules, optimizer, args, device)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
            del seq, cond
            gc.collect()
            torch.cuda.empty_cache()

        # print('done 1 epoch train')
        if epoch >= args.tfr_start_decay_epoch and args.tfr >= args.tfr_lower_bound:
            ### Update teacher forcing ratio ###
            args.tfr = args.tfr - args.tfr_decay_step

        epoch_loss_ave = epoch_loss / args.epoch_size
        epoch_mse_ave = epoch_mse / args.epoch_size
        epoch_kld_ave = epoch_kld / args.epoch_size
        tfr_total.append(args.tfr)
        loss_total.append(epoch_loss_ave)
        mse_total.append(epoch_mse_ave)
        kld_total.append(epoch_kld_ave)
        beta_total.append(args.kl_beta)
        # print('epoch_loss_ave: ', epoch_loss_ave, 'epoch_mse_ave: ', epoch_mse_ave, 'epoch_kld_ave: ', epoch_kld_ave)
        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(
                ('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss_ave, epoch_mse_ave, epoch_kld_ave)))

        # del epoch_loss, epoch_mse, epoch_kld
        # gc.collect()

        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if epoch % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):  # batch for batches
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)

                # print('\r_: ', _, end=' ')

                validate_seq.transpose_(0, 1)  # transpose batch and frame dimension
                validate_cond.transpose_(0, 1)
                validate_seq, validate_cond = validate_seq.to(device), validate_cond.to(device)
                pred_seq = pred_norm(validate_seq, validate_cond, modules, args, device)
                # print('len(validate_data) // args.batch_size: ', len(validate_data), args.batch_size, len(validate_data) // args.batch_size, )
                # print('_: ', _, 'validate_seq__: ', validate_seq.size(), 'validate_seq: ', validate_seq.size(), 'pred_seq: ', pred_seq.size())
                # print('validate_seq[args.n_past:]: ', validate_seq[:, args.n_past:].size(), 'pred_seq[args.n_past:] ', pred_seq[:, args.n_past:].size())
                # pred_seq_norm = pred_norm(validate_seq, validate_cond, modules, args, device)
                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
                psnr_list.append(psnr)
                del _, psnr, pred_seq, validate_seq, validate_cond
                gc.collect()
                torch.cuda.empty_cache()

            # print('done 1 psnr')
            ave_psnr = np.mean(np.concatenate(psnr_list))
            ave_psnr_total.append(ave_psnr)

            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)

        if epoch % 20 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)

            validate_seq.transpose_(0, 1)
            validate_cond.transpose_(0, 1)
            validate_seq, validate_cond = validate_seq.to(device), validate_cond.to(device)
            rec_seq = pred_rec(validate_seq, validate_cond, modules, args, device)
            pred_seq = pred_norm(validate_seq, validate_cond, modules, args, device)
            plot_output(validate_seq, pred_seq, rec_seq, args, spt=1)
            # print('done epoch % 20 ')
            del validate_seq, validate_cond, rec_seq, pred_seq
            gc.collect()
            torch.cuda.empty_cache()
    print('beta_total: ', len(beta_total), beta_total)
    print('tfr_total: ', len(tfr_total), tfr_total)
    print('loss_total: ', len(loss_total), loss_total)
    print('mse_total: ', len(mse_total), mse_total)
    print('kld_total: ', len(kld_total), kld_total)
    print('ave_psnr_total: ', len(ave_psnr_total), ave_psnr_total)
    show_result(loss_total, mse_total, kld_total, ave_psnr_total, beta_total, tfr_total)


def show_result(loss_total, mse_total, kld_total, ave_psnr_total, beta_total, tfr_total):
    loss = torch.tensor(loss_total, device='cpu')
    mse = torch.tensor(mse_total, device='cpu')
    kld = torch.tensor(kld_total, device='cpu')
    psnr = torch.tensor(ave_psnr_total, device='cpu')
    beta = torch.tensor(beta_total, device='cpu')
    tfr = torch.tensor(tfr_total, device='cpu')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title('Training Loss/Ratio curve', fontsize=18)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_xlabel('epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)

    lns1 = ax1.plot(loss, 'g-', label='loss')
    lns2 = ax1.plot(mse, 'r-', label='mse')
    lns3 = ax1.plot(kld, 'b-', label='kld')
    lns4 = ax1.plot(psnr, 'ko', label='psnr')
    lns5 = ax2.plot(beta, 'c--', label='beta')
    lns6 = ax2.plot(tfr, 'm--', label='tfr')
    lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

# print('Done!')
