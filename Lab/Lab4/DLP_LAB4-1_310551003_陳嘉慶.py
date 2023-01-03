from dataloader import read_bci_data  # move to the same directory
import torch
import torch.nn as nn
import torch.utils.data as tu
import matplotlib.pyplot as plt
import numpy as np

print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())
# torch.set_default_tensor_type('torch.cuda.FloatTensor')  # set gpu as default device
gpu = torch.device("cuda")


class EEGNet(nn.Module):
    def __init__(self, activation_function, dropout):
        super(EEGNet, self).__init__()  # run init at Module
        self.first_conv = nn.Sequential(  # By sequence to form module 1 after 1
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
            # padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv2d(1, 16, (1, 51), (1, 1), (0, 25), bias=False),
            # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,
            # affine=True, track_running_stats=True, device=None, dtype=None)
            nn.BatchNorm2d(16))  # out_channel = 16
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), (1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            activation_function,
            nn.AvgPool2d((1, 4), (1, 4)),  # take avg_sample of input, 750 -> 187
            # torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False,
            # count_include_pad=True, divisor_override=None)
            nn.Dropout(dropout))
        # torch.nn.Dropout(p=0.5, inplace=False)
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), (1, 1), (0, 7), bias=False),
            nn.BatchNorm2d(32),
            activation_function,
            nn.AvgPool2d((1, 8), (1, 8)),  # take avg_sample of input, 187 -> 23
            nn.Dropout(dropout))
        self.classify = nn.Sequential(
            nn.Linear(736, 2))  # 23 x 32 = 736
        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)

    def forward(self, x):
        x = self.first_conv(x)
        # print('x: ', x.shape)
        x = self.depthwise_conv(x)
        # print('x: ', x.shape)
        x = self.separable_conv(x)
        # print('x: ', x.shape)
        x = x.flatten(1)  # flatten the shape to 2D by 32*1*23 = 736 as linear input
        # print('x: ', x.shape)
        x = self.classify(x)
        return x


class DeepConvNet(nn.Module):
    def __init__(self, activation_function, dropout):
        super(DeepConvNet, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (2, 1)),
            nn.BatchNorm2d(25),
            activation_function,
            nn.MaxPool2d((1, 2)),
            # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            # default_stride = kernel_size
            nn.Dropout(p=dropout)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50),
            activation_function,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=dropout)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100),
            activation_function,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=dropout)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200),
            activation_function,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(p=dropout),
            nn.Flatten()
        )
        self.classify = nn.Sequential(
            nn.Linear(8600, 2)
        )

    def forward(self, x):
        x = self.Conv1(x)  # 750 -> 746 => 746/2=373
        # print('x: ', x.shape)  # (25, 50, (1, 5)) ([40, 25, 1, 373]) => (373-4)/2=184
        x = self.Conv2(x)
        # print('x: ', x.shape)  # (50, 100, (1, 5)) ([40, 50, 1, 184]) => (184-4)/2=90
        x = self.Conv3(x)
        # print('x: ', x.shape)  # (100, 200, (1, 5)) ([40, 100, 1, 90]) => (90-4)/2=43
        x = self.Conv4(x)
        # print('x: ', x.shape)  # [40, 8600])=> 200 x 43 = 8600
        x = x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        return x


epochs = 1001
batch_size = 40

# prepare data
train_data, train_label, test_data, test_label = read_bci_data()
# (batch size = 1080, Dimension = 1, Channel = 2, Size = 750) (prediction = 1080,) (1080, 1, 2, 750) (1080,)
train_data_tensor = torch.as_tensor(train_data)  # transfer to tensor
train_data_tensor = train_data_tensor.to(torch.float32)
train_label_tensor = torch.as_tensor(train_label)
train_label_tensor = train_label_tensor.to(torch.float32)
train_dataset = tu.TensorDataset(train_data_tensor, train_label_tensor)
batches_loader = tu.DataLoader(train_dataset, batch_size, shuffle=True)
# torch.Tensor(, device='cpu')
# combine data to dataset, shuffle=True, mess up data
print('train data size:{}, label:{} '.format(train_data_tensor.size(), train_label_tensor.size()))

test_data_tensor = torch.as_tensor(test_data)  # transfer to tensor
test_data_tensor = test_data_tensor.to(torch.float32)
test_label_tensor = torch.as_tensor(test_label)
test_label_tensor = test_label_tensor.to(torch.float32)
test_dataset = tu.TensorDataset(test_data_tensor, test_label_tensor)
test_loader = tu.DataLoader(test_dataset, batch_size)  # combine data to dataset
print('test data size:{}, label:{} '.format(test_data_tensor.size(), test_label_tensor.size()))


def train_and_test(model):
    batches_acc = []
    test_total_acc = []
    print_step = 50
    name = 'test'
    print('Running on GPU:', next(model.parameters()).is_cuda)  # check model on gpu or not
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, *, maximize=False)

    for epoch in range(epochs):
        model.train()  # train model
        for batches_x, batches_y in batches_loader:
            x = batches_x.to(gpu)  # put data to gpu to run
            y = batches_y.to(gpu)
            optimizer.zero_grad()  # clean the grad
            # print('batch: ', batches_x.shape, batches_y.shape)
            y_pred = model(x)
            loss = loss_function(y_pred, y.long())
            # print(loss.item())
            loss.backward()  # to calculate the BP 1by1, (loss_sum/y_pred_i)
            optimizer.step()  # update parameter
            batch_acc = y_pred.argmax(axis=1) == y  # argmax to take the index of max value every row(axis=1)
        batches_acc.append(sum(batch_acc) / len(batch_acc))

        model.eval()  # test model
        with torch.no_grad():  # no gradient calculation is need
            for test_x, test_y in test_loader:
                x = test_x.to(gpu)
                y = test_y.to(gpu)
                pred_y = model(x)
                test_acc = pred_y.argmax(axis=1) == y  # argmax to take the index of max value every row(axis=1)
            test_total_acc.append(sum(test_acc) / len(test_acc))
        if epoch % print_step == 0:
            print('epoch:{}, train_acurracy:{:.2%}, test_acurracy:{:.2%}'.format(epoch, batches_acc[-1], test_total_acc[-1]))
    print('train_epoch:{}, train_acurracy:{:.2%}, test_epoch:{}, test_acurracy:{:.2%}'.format(batches_acc.index(max(batches_acc)), max(batches_acc),
                                                                                              test_total_acc.index(max(test_total_acc)), max(test_total_acc)))
    return batches_acc, test_total_acc, test_total_acc.index(max(test_total_acc)), max(test_total_acc)


def show_result(train, test, name):
    train = torch.tensor(train, device='cpu')  # convert data to CPU for plot
    test = torch.tensor(test, device='cpu')  # convert data to CPU for plot
    plt.title('Activation Function Comparison', fontsize=18)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    # plt.yscale('log')
    plt.plot(train, label=name + '_train')
    plt.plot(test, label=name + '_test')
    plt.legend()
    print('----------', name, ' done!----------')


label_EEG = 'EEGNet'
label_deep = 'DeepConvNet'
label_elu = '_ELU'
label_relu = '_ReLU'
label_leaky = '_Leaky_ReLU'

elu = nn.ELU().to(gpu)
relu = nn.ReLU().to(gpu)
leaky = nn.LeakyReLU().to(gpu)
max_index = np.zeros(6)
max_acc = np.zeros(6)

dp = 0.5
# opt = torch.optim.Adam(EEGNet(leaky, dp).to(gpu).parameters())

cnn = EEGNet(elu, dp)
print('EEGNet Architecture')
print(cnn)  # print network structure

cnn = DeepConvNet(elu, dp)
print('DeepConvNet Architecture')
print(cnn)  # print network structure

train_acc, test_acc, max_index[0], max_acc[0] = train_and_test(EEGNet(elu, dp).to(gpu))  # put model to gpu and run
show_result(train_acc, test_acc, label_EEG+label_elu)
train_acc, test_acc, max_index[1], max_acc[1] = train_and_test(EEGNet(relu, dp).to(gpu))  # put model to gpu and run
show_result(train_acc, test_acc, label_EEG+label_relu)
train_acc, test_acc, max_index[2], max_acc[2] = train_and_test(EEGNet(leaky, dp).to(gpu))  # put model to gpu and run
show_result(train_acc, test_acc, label_EEG+label_leaky)
plt.show()
train_acc, test_acc, max_index[3], max_acc[3] = train_and_test(DeepConvNet(elu, dp).to(gpu))
show_result(train_acc, test_acc, label_deep+label_elu)
train_acc, test_acc, max_index[4], max_acc[4] = train_and_test(DeepConvNet(relu, dp).to(gpu))  # put model to gpu and run
show_result(train_acc, test_acc, label_deep+label_relu)
train_acc, test_acc, max_index[5], max_acc[5] = train_and_test(DeepConvNet(leaky, dp).to(gpu))  # put model to gpu and run
show_result(train_acc, test_acc, label_deep+label_leaky)

print(label_EEG+label_elu)
print('Max: test_epoch:{}, test_acurracy:{:.2%}'.format(max_index[0], max_acc[0]))
print(label_EEG+label_relu)
print('Max: test_epoch:{}, test_acurracy:{:.2%}'.format(max_index[1], max_acc[1]))
print(label_EEG+label_leaky)
print('Max: test_epoch:{}, test_acurracy:{:.2%}'.format(max_index[2], max_acc[2]))
print(label_deep+label_elu)
print('Max: test_epoch:{}, test_acurracy:{:.2%}'.format(max_index[3], max_acc[3]))
print(label_deep+label_relu)
print('Max: test_epoch:{}, test_acurracy:{:.2%}'.format(max_index[4], max_acc[4]))
print(label_deep+label_leaky)
print('Max: test_epoch:{}, test_acurracy:{:.2%}'.format(max_index[5], max_acc[5]))
plt.show()
