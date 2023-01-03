from dataset import ICLEVRLoader
from evaluator import evaluation_model
import torch
import torch.nn as nn
import torch.utils.data as tud
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import os
import random
from tqdm import tqdm


class DCGAN(nn.Module):
    def __init__(self, z_size=100, cond_size=24, cond_fc_size=156):
        super(DCGAN, self).__init__()  #
        self.z_size = z_size
        self.cond_size = cond_size
        self.cond_fc_size = cond_fc_size
        self.fc = nn.Sequential(
            nn.Linear(cond_size, self.cond_fc_size),  # 24 --> 156 fit first layer to 156+100(z_size)=256
            nn.ReLU(True)
        )
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.z_size+self.cond_fc_size, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512),  # out_channel_num
            nn.ReLU(True)  # save memory
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, z, cond):
        cond = self.fc(cond).view(-1, self.cond_fc_size, 1, 1)  # reshape to be same size of z
        z = torch.cat((z, cond), dim=1)  # attach cond to z
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, img_size=64, cond_size=24):
        super(Discriminator, self).__init__()
        self.cond_size = cond_size
        self.img_size = img_size
        self.fc = nn.Sequential(
            nn.Linear(self.cond_size, self.img_size * self.img_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(img_size, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()  # WGAN1: remove sigmoid
        )

    def forward(self, img, cond):
        cond = self.fc(cond).view(-1, 1, self.img_size, self.img_size)
        # print('size:', img.size(), cond.size())
        img = torch.cat((img, cond), dim=1)
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        img = self.layer5(img)
        return img


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # init as normal distribution (tensor, mean=0, std=1)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)  # init as constant val (tensor, val)


def train(model_g, model_d, args):
    if args.model_dir != '':
        # load model and continue training from checkpoint
        print('loaded model')
        saved_model = torch.load('%s/%s' % (args.model_dir, args.load_model_name))
        model_dir = args.model_dir
        iter_g = args.iter_g
        iter_d = args.iter_d
        args = saved_model['args']
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epochs = 0  # saved_model['last_epoch']
        # read = pd.read_csv('%s/train.csv' % args.model_dir)
        # loss_real_total, loss_fake_total, loss_g_total, acc_real_total, acc_fake_total, acc_g_total = \
        #     read['loss_real_total'].tolist(), read['loss_fake_total'].tolist(), read['loss_g_total'].tolist(), \
        #     read['acc_real_total'].tolist(), read['acc_fake_total'].tolist(), read['acc_g_total'].tolist()
        # print('loss_real_total: ', len(loss_real_total), loss_real_total)
        # print('loss_fake_total: ', len(loss_fake_total), loss_fake_total)
        # print('loss_g_total: ', len(loss_g_total), loss_g_total)
        # print('acc_real_total: ', len(acc_real_total), acc_real_total)
        # print('acc_fake_total: ', len(acc_fake_total), acc_fake_total)
        # print('acc_g_total: ', len(acc_g_total), acc_g_total)
        # print('start_epochs: ', start_epochs)
    else:
        name = '%d_train_lrg=%.5f_lrd=%.5f_iter_g=%d_iter_d=%d_clip=%.3f_gamma=%.5f_gamma_step=%.5f' \
               % (args.train_count, args.lrg, args.lrd, args.iter_g, args.iter_d, args.clp, args.gamma, args.gamma_step)
        args.log_dir = '%s/%s' % (args.log_dir, name)
        iter_g = args.iter_g
        iter_d = args.iter_d
        start_epochs = 0

    pic_path = '%s/pic/' % args.log_dir
    pic_path_new = '%s/new_pic/' % args.log_dir
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(pic_path, exist_ok=True)
    os.makedirs(pic_path_new, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    print(args)
    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    batch_size, device, lrg, lrd, gamma, gamma_step, beta, total_epochs = \
        args.batch_size, args.device, args.lrg, args.lrd, args.gamma, args.gamma_step, args.beta, args.epochs
    if args.model_dir != '':
        model_g = saved_model['model_g']
        model_d = saved_model['model_d']
    else:
        model_g.apply(weights_init)
        model_d.apply(weights_init)
    model_g.to(device)
    model_d.to(device)
    model_e = evaluation_model('./checkpoint.pth')
    # print(model_g)
    # print(model_d)
    # loss_function = nn.BCELoss()  # WGAN2: use RMS loss, remove log loss function
    opt_g = torch.optim.RMSprop(model_g.parameters(), lr=lrg)  # WGAN2: use RMS loss
    opt_d = torch.optim.RMSprop(model_d.parameters(), lr=lrd)
    # opt_g = torch.optim.Adam(model_g.parameters(), lr=lrg, betas=(beta, 0.999))  # WGAN2: use RMS loss
    # opt_d = torch.optim.Adam(model_d.parameters(), lr=lrd, betas=(beta, 0.999))
    g_lr_step = torch.optim.lr_scheduler.StepLR(opt_g, step_size=gamma_step, gamma=gamma)
    d_lr_step = torch.optim.lr_scheduler.StepLR(opt_d, step_size=gamma_step, gamma=gamma)

    best_score, best_new_score = 0, 0
    epoch_score, epoch_new_score, loss_real_total, loss_fake_total, loss_g_total, acc_real_total, loss_d_total,\
        acc_fake_total, acc_g_total = [], [], [], [], [], [], [], [], []
    total_epochs = total_epochs - start_epochs
    progress = tqdm(total=total_epochs)

    for epoch in range(total_epochs):
        score, score_new = 0, 0
        loss_real_batch, loss_fake_batch, loss_g_batch, acc_real_batch, loss_d_batch, acc_fake_batch,\
            acc_g_batch, img_batch, img_total, img_new_total = [], [], [], [], [], [], [], [], [], []
        for data in data_loader_train:
            img = data[0].to(device)
            cond = data[1].to(device, dtype=torch.float)
            size = img.size()[0]
            one = torch.ones(size).to(device)
            fake_label = torch.zeros(size).to(device)  # WGAN3: use -1 instead of 0
            mone = -1 * one
            z = torch.randn(size, 100, 1, 1).to(device)
            loss_real_sum, acc_real_sum, loss_fake_sum, loss_d_sum, acc_fake_sum, \
                loss_g_sum, acc_g_sum = 0, 0, 0, 0, 0, 0, 0
            model_d.train()
            for iter in range(iter_d):
                model_d.zero_grad()
                pred_label = model_d(img, cond).view(-1)
                pred_label.backward(one)   # WGAN3: remove loss function and directly backward one and mone
                # loss_real = loss_function(pred_label, real_label)  # pred_label should be near 1
                acc_real = one - pred_label
                # loss_real_sum += loss_real.item()
                acc_real_sum += acc_real.mean().item()
                loss_real = -torch.mean(pred_label)
                loss_real_sum += loss_real.sum().item()
                # loss_real = one - pred_label  # original
                # loss_real_sum += loss_real.sum().abs().item()  # original

                pred_img = model_g(z, cond).detach()  # fix Gen, prevent bp
                pred_label_fake = model_d(pred_img, cond).view(-1)
                pred_label_fake.backward(mone)  # WGAN3: remove loss function and directly backward one and mone
                # loss_fake = loss_function(pred_label_fake, fake_label)  # pred_label should be near 0
                acc_fake = pred_label_fake  # edit
                # loss_fake_sum += loss_fake.item()
                acc_fake_sum += acc_fake.mean().item()
                loss_fake = torch.mean(pred_label_fake)
                loss_fake_sum += loss_fake.sum().item()
                # loss_fake = pred_label_fake  # original
                # loss_fake_sum += loss_fake.sum().abs().item()  # original
                loss_d = pred_label_fake - pred_label
                loss_d_sum += loss_d.sum()
                # loss_d = loss_real + loss_fake
                # loss_d.backward()  # WGAN3: remove loss function and directly backward one and mone
                opt_d.step()
            loss_size = size * iter_d
            loss_d_batch.append(loss_d_sum / loss_size)
            loss_real_batch.append(loss_real_sum / loss_size)
            # print('real: ', loss_real, loss_real_sum, loss_size)
            # print('fake: ', loss_fake, loss_fake_sum, loss_size)
            acc_real_batch.append(acc_real_sum / loss_size)
            loss_fake_batch.append(loss_fake_sum / loss_size)
            acc_fake_batch.append(acc_fake_sum / loss_size)
            for para in model_d.parameters():  # WGAN3: limit model_d
                para.data.clamp_(-args.clp, args.clp)

            model_g.train()
            for iter in range(iter_g):
                model_g.zero_grad()
                pred_img = model_g(z, cond)
                pred_label = model_d(pred_img, cond).view(-1)
                # loss_g = loss_function(pred_label, real_label)  # pred_label should be near 1
                # loss_g_sum += loss_g.item()
                loss_g = -torch.mean(pred_label)
                loss_g_sum += loss_g.sum()
                loss_g = pred_label  # org
                loss_g_sum += loss_g.sum().abs().item()  # org
                pred_label.backward(one)  # WGAN3: remove loss function and directly backward one and mone
                # loss_g.backward()
                opt_g.step()
                acc_g = pred_label.mean().item()
                acc_g_sum += acc_g
                # if (iter+1) == iter_g and loss_d > loss_g:
                #     iter -= 1
            loss_size = size * iter_g
            loss_g_batch.append(loss_g_sum / loss_size)
            # print('loss_g: ', loss_g, loss_g_sum, loss_size)
            acc_g_batch.append(acc_g_sum / loss_size)
        batches_size = len(data_loader_train)
        loss_d_total.append(sum(loss_real_batch) / batches_size)
        loss_real_total.append(sum(loss_real_batch) / batches_size)
        acc_real_total.append(sum(acc_real_batch) / batches_size)
        loss_fake_total.append(sum(loss_fake_batch) / batches_size)
        acc_fake_total.append(sum(acc_fake_batch) / batches_size)
        loss_g_total.append(sum(loss_g_batch) / batches_size)
        acc_g_total.append(sum(acc_g_batch) / batches_size)
        # with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        #         train_record.write(
        #             ('[epoch: %02d] acc_real_total: %s | acc_fake_total: %.5f | acc_g_total: %.5f\n' %
        #              (epoch, acc_real_total[-1], acc_fake_total[-1], acc_g_total[-1])))

        model_g.eval()
        model_d.eval()
        for data in data_loader_test:
            cond = data[1].to(device, dtype=torch.float)
            # print('cond :', cond)
            size = cond.size()[0]
            z = torch.randn(size, 100, 1, 1).to(device)
            with torch.no_grad():
                img = model_g(z, cond)
                img_total.append(img)
                score += model_e.eval(img, cond)
        batch_score = score / len(data_loader_test)
        epoch_score.append(batch_score)

        for data in data_loader_new_test:
            cond = data[1].to(device, dtype=torch.float)
            # print('cond_new :', cond)
            size = cond.size()[0]
            z = torch.randn(size, 100, 1, 1).to(device)
            with torch.no_grad():
                img_new = model_g(z, cond)
                img_new_total.append(img_new)
                score_new += model_e.eval(img_new, cond)
        batch_new_score = score_new / len(data_loader_new_test)
        epoch_new_score.append(batch_new_score)

        if batch_score > best_score or batch_new_score > best_new_score:
            if batch_score > best_score and batch_new_score > best_new_score:
                best_epoch = epoch
                best_score = batch_score
                best_new_epoch = epoch
                best_new_score = batch_new_score
            elif batch_score > best_score:
                best_epoch = epoch
                best_score = batch_score
            else:
                best_new_epoch = epoch
                best_new_score = batch_new_score
                # if batch_new_score > best_new_score:
                #     best_new_score = batch_new_score
            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(
                    ('[epoch: %02d] loss_real_total: %s | loss_fake_total: %s | loss_g_total: %.5f | acc_real_total: %s | acc_fake_total: %.5f | acc_g_total: %.5f | test score = %.5f | new test score = %.5f save model\n' %
                     (epoch, loss_real_total[-1], loss_fake_total[-1], loss_g_total[-1], acc_real_total[-1],
                      acc_fake_total[-1], acc_g_total[-1], batch_score, batch_new_score)))
                # train_record.write(
                #     ('=============== test score = {:.5f} ==== new test score = {:.5f} ==saved model\n'.format(
                #         batch_score, batch_new_score)))
            # save the model
            torch.save({
                'model_g': model_g,
                'model_d': model_d,
                'args': args,
                'last_epoch': epoch},
                '%s/%s' % (args.log_dir, args.save_model_name))
            dict = {'acc_real_total': acc_real_total, 'acc_fake_total': acc_fake_total,
                    'acc_g_total': acc_g_total}
            df = pd.DataFrame(dict)
            df.to_csv('%s/train.csv' % args.log_dir)
            show_result(epoch_score, epoch_new_score)
            show_loss(loss_real_total, loss_fake_total, loss_g_total)
            for i in range(len(img_total)):  # len(img_total)
                for j in range(len(img_total[i])):
                    img_name = f'test_{epoch}_{i*len(img_total)+j}.png'
                    show_from_tensor(img_total[i][j], pic_path, img_name)  # save fig
            for i in range(len(img_new_total)):  # len(img_total)
                for j in range(len(img_new_total[i])):
                    img_name = f'new_test_{epoch}_{i*len(img_new_total)+j}.png'
                    show_from_tensor(img_new_total[i][j], pic_path_new, img_name)  # save fig
            # else:
            #     best_new_score = batch_new_score
            #     with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            #         train_record.write(
            #             ('[epoch: %02d] loss_real_total: %s | loss_fake_total: %s | loss_g_total: %.5f | acc_real_total: %s | acc_fake_total: %.5f | acc_g_total: %.5f | test score = %.5f | new test score = %.5f save model\n' %
            #              (epoch, loss_real_total[-1], loss_fake_total[-1], loss_g_total[-1], acc_real_total[-1],
            #               acc_fake_total[-1], acc_g_total[-1], batch_score, batch_new_score)))
            #         for i in range(len(img_total)):  # len(img_total)
            #             for j in range(len(img_total[i])):
            #                 img_name = f'test_{epoch}_{i * len(img_total) + j}.png'
            #                 show_from_tensor(img_total[i][j], pic_path, img_name)  # save fig
            #         for i in range(len(img_new_total)):  # len(img_total)
            #             for j in range(len(img_new_total[i])):
            #                 img_name = f'new_test_{epoch}_{i * len(img_new_total) + j}.png'
            #                 show_from_tensor(img_new_total[i][j], pic_path_new, img_name)  # save fig
            #     # save the model
            #     torch.save({
            #         'model_g': model_g,
            #         'model_d': model_d,
            #         'args': args,
            #         'last_epoch': epoch},
            #         '%s/new_%s' % (args.log_dir, args.save_model_name))
            #     dict = {'acc_real_total': acc_real_total, 'acc_fake_total': acc_fake_total,
            #             'acc_g_total': acc_g_total}
            #     df = pd.DataFrame(dict)
            #     df.to_csv('%s/train.csv' % args.log_dir)
            #     show_result(epoch_score, epoch_new_score)
            #     show_loss(loss_real_total, loss_fake_total, loss_g_total)
        else:
            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(
                    ('[epoch: %02d] loss_real_total: %s | loss_fake_total: %s | loss_g_total: %.5f | acc_real_total: %s | acc_fake_total: %.5f | acc_g_total: %.5f | test score = %.5f | new test score = %.5f\n' %
                                (epoch, loss_real_total[-1], loss_fake_total[-1], loss_g_total[-1], acc_real_total[-1],
                                 acc_fake_total[-1], acc_g_total[-1], batch_score, batch_new_score)))

        if epoch % args.save_step == 0:
            show_result(epoch_score, epoch_new_score)
            show_loss(loss_real_total, loss_fake_total, loss_g_total)
            for i in range(len(img_total)):  # len(img_total)
                for j in range(len(img_total[i])):
                    img_name = f'test_{epoch}_{i * len(img_total) + j}.png'
                    show_from_tensor(img_total[i][j], pic_path, img_name)  # save fig
            for i in range(len(img_new_total)):  # len(img_total)
                for j in range(len(img_new_total[i])):
                    img_name = f'new_test_{epoch}_{i * len(img_new_total) + j}.png'
                    show_from_tensor(img_new_total[i][j], pic_path_new, img_name)  # save fig

        d_lr_step.step()  # WGAN2: use RMS loss
        g_lr_step.step()
        progress.update(1)
        progress.set_description('best_epoch = %d | best score = %.5f | best_new_epoch = %d | best new score = %.5f'
                                 % (best_epoch, best_score, best_new_epoch, best_new_score))
    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write(
            ('best_epoch = %d | best score = %.5f | best_new_epoch = %d | best new score = %.5f'
             % (best_epoch, best_score, best_new_epoch, best_new_score)))
    print('\nbest_epoch = %d | best score = %.5f | best_new_epoch = %d | best new score = %.5f'
                                 % (best_epoch, best_score, best_new_epoch, best_new_score))
    show_result(epoch_score, epoch_new_score)
    show_loss(loss_real_total, loss_fake_total, loss_g_total)


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
    plt.close('all')


def tensor_to_np(tensor):
    # print('size(): ', tensor.size())
    img = tensor.add_(1.0).mul(127.5).byte()
    # print('size(): ', img.size())
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img


def show_loss(loss_real, loss_fake, loss_g):
    loss_real = torch.tensor(loss_real, device='cpu')
    loss_fake = torch.tensor(loss_fake, device='cpu')
    loss_g = torch.tensor(loss_g, device='cpu')
    plt.figure()
    plt.title('WGAN Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    lns1 = plt.plot(loss_real, 'r--', label='Loss_real')
    lns2 = plt.plot(loss_fake, 'g--', label='Loss_fake')
    lns3 = plt.plot(loss_g, 'b--', label='Loss_g')
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="upper right")
    plt.tight_layout()
    plt.savefig('%s/loss.png' % args.log_dir)
    plt.close('all')


def show_result(epoch_score, epoch_new_score):
    score = torch.tensor(epoch_score, device='cpu')
    new_score = torch.tensor(epoch_new_score, device='cpu')
    plt.figure()
    plt.title('WGAN Score', fontsize=18)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)

    lns1 = plt.plot(score, 'r+', label='score')
    lns2 = plt.plot(new_score, 'b+', label='new_score')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="lower right")
    plt.tight_layout()
    plt.savefig('%s/train_result.png' % args.log_dir)
    plt.close('all')


def parser_arg():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--save_model_name', default='dcgan.pth')
    parser.add_argument('--load_model_name', default='dcgan.pth')
    parser.add_argument('--log_dir', default='log/wgan')
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--root', default='./')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--title_score', default='WGAN Score')
    parser.add_argument('--label_score', default='WGAN Score')
    parser.add_argument('--title_loss', default='WGAN Loss')
    parser.add_argument('--label_loss', default='WGAN Loss')
    # train
    parser.add_argument('--epochs', default=2800, type=int)
    parser.add_argument('--save_step', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lrd', default=1e-4, type=float)
    parser.add_argument('--lrg', default=1e-4, type=float)  #
    parser.add_argument('--beta', default=0.4, type=float)
    parser.add_argument('--gamma', default=1.1, type=float)
    parser.add_argument('--gamma_step', default=400, type=int)
    parser.add_argument('--iter_d', default=1, type=int)
    parser.add_argument('--iter_g', default=2, type=int)
    parser.add_argument('--clp', default=.01, type=float)
    parser.add_argument('--train_count', default=3, type=int)
    # test
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--seed', default=203, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_arg()
    if not torch.cuda.is_available():
        args.device = 'cpu'
    dataset_train = ICLEVRLoader(args.root, mode=args.mode)
    data_loader_train = tud.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=10)
    args.mode = 'test'
    dataset_test = ICLEVRLoader(args.root, mode=args.mode)
    data_loader_test = tud.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=10)
    args.mode = 'new_test'
    dataset_new_test = ICLEVRLoader(args.root, mode=args.mode)
    data_loader_new_test = tud.DataLoader(dataset_new_test, batch_size=args.batch_size, shuffle=False, num_workers=10)
    train(DCGAN(), Discriminator(), args)

