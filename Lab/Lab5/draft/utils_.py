import math
import argparse
from operator import pos
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as Img
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c])
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr


def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err


# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    # print('gt.type: {}, pred: {}'.format(type(gt), type(pred)))
    # gt = torch.reshape(gt, (gt.size()[1], gt.size()[2], gt.size()[3], gt.size()[4]))
    # print('gt.shape: {}, pred: {}'.format(gt.size(), pred.size()))
    # T = len(gt[0])
    # bs = gt[0, 0].shape[0]
    T = len(gt)  # frame?
    bs = len(gt[0])  # channel?
    # print('T: {}, bs: {}'.format(T, bs))
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))

    for i in range(bs):
        for t in range(T):
            # print('i: {}, t: {}'.format(i, t))
            origin = gt[t][i].detach().cpu().numpy()  # gt[t][i]
            predict = pred[t][i].detach().cpu().numpy()
            # print('origin.shape: {}, predict: {}'.format(origin.shape, predict.shape))
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr


def finn_psnr(x, y, data_range=1.):
    mse = ((x - y) ** 2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    # print('type img1: {}, img2: {}, window: {}'.format(type(img1), type(img2), type(window)))
    # print('img1: {}, img2: {}, window: {}'.format(img1.shape, img2.shape, window.shape))
    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def kl_criterion(mean, logvar, args):
    KLD = torch.sum(0.5 * (mean**2 + logvar.exp() - logvar) - 0.5)     # 0.5 * sum(1 + log(logvar^2) - mean^2 - sigma^2)
    KLD /= args.batch_size
    return KLD


def show_from_tensor(tensor, path, title=None):
    img = tensor.clone()
    # print('tensor: ', img.size())
    img = tensor_to_np(img)
    # print('img: ', img.shape)
    plt.figure()
    plt.axis("off")
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)
    plt.savefig(path + title)
    # plt.show()
    # plt.close('all')


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img


def reparameter(mean, logvar, size, device):
    var = torch.ones(size).to(device) * logvar
    mean = torch.zeros(size).to(device) + mean
    std = torch.exp(0.5 * var)
    eps = torch.randn_like(std)
    return mean + eps * std


def plot_output(ref_seq, pred_seq, rec_seq, args, spt=1):
    # print('=========================start_plot_rec===================================')
    path = './gif/'
    frame_num = args.n_past + args.n_future

    for batch in range(args.batch_size):
        filenames1, filenames2, filenames3 = [], [], []
        for frame in range(frame_num):
            filename1 = f'{str(1)}_{batch}_{frame}.png'
            # print('plot_size: ', ref_seq.size(), ref_seq[batch, frame].size())
            show_from_tensor(ref_seq[frame, batch], path, filename1)  # save fig
            # create file name and append it to a list
            filenames1.append(path + filename1)

            filename2 = f'{str(2)}_{batch}_{frame}.png'
            # print('plot_size: ', pred_seq.size(), pred_seq[batch, frame].size())
            show_from_tensor(pred_seq[frame, batch], path, filename2)  # save fig
            # create file name and append it to a list
            filenames2.append(path + filename2)

            filename3 = f'{str(3)}_{batch}_{frame}.png'
            # print('plot_size: ', pred_seq_norm.size(), pred_seq_norm[batch, frame].size())
            show_from_tensor(rec_seq[frame, batch], path, filename3)  # save fig
            # create file name and append it to a list
            filenames3.append(path + filename3)
            # plt.axis("off")
            # plt.yticks([])
            # plt.imshow(image)
            # # save frame
            # plt.savefig(filename)
            plt.close('all')

        # build gif
        with imageio.get_writer(path + 'ref_seq_' + str(batch) + '.gif', mode='I') as writer:
            for filename in filenames1:
                image1 = imageio.imread(filename)
                writer.append_data(image1)

        with imageio.get_writer(path + 'pred_seq_' + str(batch) + '.gif', mode='I') as writer:
            for filename in filenames2:
                image2 = imageio.imread(filename)
                writer.append_data(image2)

        with imageio.get_writer(path + 'rec_seq_' + str(batch) + '.gif', mode='I') as writer:
            for filename in filenames3:
                image3 = imageio.imread(filename)
                writer.append_data(image3)

        # fig = plt.figure()
        # plt.subplot(1, 3, spt)
        # plt.title(str(spt))
        # plt.axis("off")
        # # plt.yticks([])
        # ims = []
        # for frame in range(frame_num):
        #     img = Img.imread(path + f'{str(spt)}_{batch}_{frame}.png')
        #     # print('img: ', img.shape)
        #     im = plt.imshow(img, animated=True)
        #     ims.append([im])
        #
        # spt = 2
        # plt.subplot(1, 3, spt)
        # plt.title(str(spt))
        # plt.axis("off")
        # ims2 = []
        # for frame in range(frame_num):
        #     img2 = Img.imread(path + f'{str(spt)}_{batch}_{frame}.png')
        #     # print('img: ', img.shape)
        #     im2 = plt.imshow(img2, animated=True)
        #     ims2.append([im2])
        #
        # spt = 3
        # plt.subplot(1, 3, spt)
        # plt.title(str(spt))
        # plt.axis("off")
        # ims3 = []
        # for frame in range(frame_num):
        #     img3 = Img.imread(path + f'{str(spt)}_{batch}_{frame}.png')
        #     im3 = plt.imshow(img3, animated=True)
        #     ims3.append([im3])
        #
        # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        # ani2 = animation.ArtistAnimation(fig, ims2, interval=50, blit=True, repeat_delay=1000)
        # ani3 = animation.ArtistAnimation(fig, ims3, interval=50, blit=True, repeat_delay=1000)
        # plt.show()
        # plt.close(fig)
        # print('=========================end_plot_rec===================================')


def pred_norm(validate_seq, validate_cond, modules, args, device):
    # print('=========================start_pred_norm===================================')
    # print('seq_x: ', validate_seq.size(), 'seq_validate_seq[0]: ', validate_seq[:2].size())
    decoder_output_total = torch.tensor(validate_seq[0].unsqueeze(0)).to(device)
    # encoder_output_total = []
    # for i in range(args.n_past):
    encoder_output_total = [modules['encoder'](validate_seq[0])]  # h5.view(-1, self.dim), [h1, h2, h3, h4]
    # print('seq_x: ', validate_seq.size(), 'seq_validate_seq[0]: ', validate_seq[0].size())
    for i in range(1, args.n_past + args.n_future):  # frames_size
        # print('i: ', i)
        if i == 1:
            encoder_output_total.append(modules['encoder'](validate_seq[i]))
            z, mean, logvar = modules['posterior'](encoder_output_total[i][0])
            # print('z: {}, mean: {}, logvar: {}, z_max: {}, z_min: {}'.format(z.size(), mean.size(), logvar.size(), z.max(), z.min()))
            if args.last_frame_skip or i < args.n_past:
                encoder_output_past, remain = encoder_output_total[i - 1]  # h5, [h1, h2, h3, h4]
            else:
                encoder_output_past = encoder_output_total[i - 1][0]
        else:
            encoder_output_past = modules['encoder'](decoder_output_total[i - 1])[0]  # [0]  # h5, [h1, h2, h3, h4] take h5 only
            mean, logvar = args.mean, torch.tensor(args.var).to(device)
            z = reparameter(mean, logvar, (args.batch_size, args.z_dim), device)
            # print('z: {}, z_max: {}, z_min: {}'.format(z.size(), z.max(), z.min()))
        # print('z: {}, z_max: {}, z_min: {}'.format(z.size(), z.max(), z.min()))
        # print('encoder_output_past: ', encoder_output_past.size(), 'validate_cond[i - 1]: ', validate_cond[i - 1].size())
        lstm_input = torch.cat((encoder_output_past, validate_cond[i - 1], z), 1)
        # print('lstm_input_size: ', lstm_input.size())
        lstm_output = modules['frame_predictor'](lstm_input)
        # print('rnn_output: ', lstm_output.size())
        decoder_output = modules['decoder']((lstm_output, remain))
        # print('output: ', decoder_output.size())
        decoder_output_total = torch.cat((decoder_output_total, decoder_output.unsqueeze(0)))
        # print('decoder_output_total: ', decoder_output_total.size(), 'decoder_output_total[i - 1, 0]: ', decoder_output_total[i - 1, 0].size())
    # print('=========================end_pred_norm===================================')
    return decoder_output_total


def pred_rec(validate_seq, validate_cond, modules, args, device):
    # print('=========================start_pred_rec===================================')
    decoder_output_total = torch.tensor(validate_seq[0].unsqueeze(0)).to(device)
    # encoder_output_total = []
    # for i in range(args.n_past):
    encoder_output_total = [modules['encoder'](validate_seq[0])]  # h5.view(-1, self.dim), [h1, h2, h3, h4]
    # print('seq_x: ', validate_seq.size(), 'seq_validate_seq[0]: ', validate_seq[0].size())
    for i in range(1, args.n_past + args.n_future):  # frames_size
        encoder_output_total.append(modules['encoder'](validate_seq[i]))
        z, mean, logvar = modules['posterior'](encoder_output_total[i][0])
        # print('z: {}, mean: {}, logvar: {}, z_max: {}, z_min: {}'.format(z.size(), mean.size(), logvar.size(), z.max(), z.min()))
        if i == 1:
            if args.last_frame_skip or i < args.n_past:
                encoder_output_past, remain = encoder_output_total[i - 1]  # h5, [h1, h2, h3, h4]
            else:
                encoder_output_past = encoder_output_total[i - 1][0]
        else:
            encoder_output_past = modules['encoder'](decoder_output_total[i - 1])[0]  # h5, [h1, h2, h3, h4] take h5 only
        lstm_input = torch.cat((encoder_output_past, validate_cond[i-1], z), 1)
        # print('lstm_input_size: ', lstm_input.size(), 'validate_cond[i - 1]: ', validate_cond[i - 1].size())
        lstm_output = modules['frame_predictor'](lstm_input)
        # print('rnn_output: ', lstm_output.size())
        decoder_output = modules['decoder']((lstm_output, remain))
        # print('output: ', decoder_output.size())
        decoder_output_total = torch.cat((decoder_output_total, decoder_output.unsqueeze(0)))
        # print('decoder_output_total: ', decoder_output_total.size(), 'decoder_output_total[i - 1, 0]: ', decoder_output_total[i - 1, 0].size())
    # print('=========================end_pred_rec===================================')
    return decoder_output_total
