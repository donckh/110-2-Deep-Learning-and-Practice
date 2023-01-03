import dataloader
import torch
import torch.nn as nn
import torch.utils.data as tu
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import ConfusionMatrix


# import sys
# print(sys.getrecursionlimit())
# sys.setrecursionlimit(2000)
# print(sys.getrecursionlimit())
torch.cuda.empty_cache()
print(torch.cuda.get_device_name(), torch.cuda.device_count())
batches = 4
epochs = 11
root = './data/'
train = 'train'
test = 'test'
gpu = torch.device("cuda")


def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


fix_seed(2)

train_data = dataloader.RetinopathyLoader(root, train)
train_data_loader = tu.DataLoader(train_data, batch_size=batches, shuffle=True)

test_data = dataloader.RetinopathyLoader(root, test)
test_data_loader = tu.DataLoader(test_data, batch_size=batches)
print(train_data)

resnet18 = models.resnet18()  # pretrained: bool = False
resnet50 = models.resnet50()
resnet18.fc = nn.Linear(in_features=512, out_features=5)
resnet50.fc = nn.Linear(in_features=2048, out_features=5)

pretrain_resnet18 = models.resnet18(pretrained=True)
pretrain_resnet50 = models.resnet50(pretrained=True)
# print(pretrain_resnet50)
pretrain_resnet18.fc = nn.Linear(in_features=512, out_features=5)
pretrain_resnet50.fc = nn.Linear(in_features=2048, out_features=5)

# print(resnet18)
# print(pret
# rain_resnet50)


def train_and_test(model, lr, path, label, mode):
    model.to(gpu)
    print('Running on GPU: ', next(model.parameters()).is_cuda)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    # train_acc_total = torch.tensor([]).to(gpu, dtype=torch.float)
    # test_acc_total = torch.tensor([]).to(gpu, dtype=torch.float)
    prediction_batches = torch.tensor([]).to(gpu, dtype=torch.int)
    prediction_total = torch.tensor([]).to(gpu, dtype=torch.int)
    target_batches = torch.tensor([]).to(gpu, dtype=torch.int)
    target_total = torch.tensor([]).to(gpu, dtype=torch.int)
    train_ep_acc = torch.tensor([]).to(gpu, dtype=torch.float)
    test_ep_acc = torch.tensor([]).to(gpu, dtype=torch.float)
    for epoch in range(epochs):
        if mode == 'tt' or mode == 'train':
            model.train()
            train_acc = 0
            # train_ep_acc = []
            for x_batch, y_batch in train_data_loader:
                x = x_batch.to(gpu, dtype=torch.float)
                y = y_batch.to(gpu, dtype=torch.long)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_function(y_pred, y)

                loss.backward()
                optimizer.step()
                # train_acc = y_pred.argmax(axis=1) == y

                # print('test_pred: {}, y: {}, train_acc: {}

                # train_acc_total = data_reshape(train_acc_total, train_acc)
                # print('ypred: {}, y: {}'.format(torch.max(y_pred,1)[1], y))

                train_acc += (torch.max(y_pred, 1)[1] == y).sum().item()
            train_acc_total = torch.tensor([train_acc / len(train_data)]).to(gpu, dtype=torch.float)
            train_ep_acc = torch.cat((train_ep_acc, train_acc_total)).to(gpu)
            # y_pred_arr_total = np.array(y_pred_arr_total)
            # y_target_arr_total = np.array(y_target_arr_total)
            # print('train_acc: {}, ytrain_acc_total: {}'.format(train_acc, train_acc_total))
            # print(type(y_arr_total))
            # train_ep_acc.append((sum(train_acc_total)/len(train_acc_total)))
            # print('*********************************************************************************')
            # print('train_ep_acc: {}, \nsum: {}, len: {}'.format(train_ep_acc, train_acc_total, len(train_data)))

            torch.save(model.state_dict(), path+label+'_lr='+str(lr)+'_'+str(epoch))

        if mode == 'tt' or mode == 'test':
            model.load_state_dict(torch.load(path+label+'_lr='+str(lr)+'_'+str(epoch)))
            model.eval()
            with torch.no_grad():
                test_acc = 0
                # test_ep_acc = []
                # prediction_batches =[]
                # prediction_total = []
                # target_batches = []
                # target_total = []
                for x_batch, y_batch in test_data_loader:
                    x = x_batch.to(gpu, dtype=torch.float)
                    y = y_batch.to(gpu, dtype=torch.long)
                    y_pred = model(x)
                    prediction = torch.max(y_pred, 1)[1]
                    prediction_batches = torch.cat((prediction_batches, prediction)).to(gpu)
                    target_batches = torch.cat((target_batches, y)).to(gpu)
                    test_acc += (prediction == y).sum().item()
                    # print('y: {}, y-batch: {}.format'.format(y, target_batches))
                test_acc_total = torch.tensor([test_acc / len(test_data)]).to(gpu, dtype=torch.float)
                test_ep_acc = torch.cat((test_ep_acc, test_acc_total)).to(gpu)
                prediction_total = torch.cat((prediction_total, prediction_batches))
                target_total = torch.cat((target_total, target_batches)).to(gpu)
                # print('test: ', len(prediction_total), len(target_total))

                # y_test_pred_arr_total = np.array(y_test_pred_arr_total)
                # y_test_target_arr_total = np.array(y_test_target_arr_total)
                # test_ep_acc.append((sum(test_acc_total)/len(test_acc_total)))
                print('\repochs: {:.2%}'.format(epoch/(epochs-1)), end='')
    print('\n')
    return train_ep_acc, test_ep_acc, prediction_total, target_total  # y_pred_arr_total, y_target_arr_total,


def show_result(train_acc, test_acc, name):
    train = torch.tensor(train_acc, device='cpu')
    test = torch.tensor(test_acc, device='cpu')
    plt.title('CNN Accuracy Comparison ', fontsize=18)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('epochs', fontsize=12)
    plt.plot(train, label=name + '_train')
    plt.plot(test, label=name + '_test')
    # plt.plot([0.5,0.5], label='testing')
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")  # , borderaxespad=0
    plt.tight_layout()


def cm(y_pred, y_true, name):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true")  # , normalize="true"
    fig, ax = plt.subplots()
    mappable = ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    fig.colorbar(mappable, ax=ax)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(name + ' Confusion Matrix', fontsize=18)
    ax.xaxis.tick_bottom()

    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=format(confmat[i, j], '.2f'), va='center', ha='center')  # format(confmat[i, j], '.2f')



label_18 = 'Resnet18'
label_50 = 'Resnet50'
label_pre18 = 'Pretrain_Resnet18'
label_pre50 = 'Pretrain_Resnet50'
label_train = 'train'
label_test = 'test'
label_tt = 'tt'
path = './weight/'

mode = label_tt

lr = 1e-3
train_acc_18, test_acc_18, y_predict_18, y_target_18 = train_and_test(resnet18, lr, path, label_18, mode)  # y_train_pred, y_train_target,
show_result(train_acc_18, test_acc_18, label_18)
cm(y_predict_18, y_target_18, label_18)
print(train_acc_18, test_acc_18)
print(label_18, end=' ')
if len(train_acc_18) != 0 and len(test_acc_18) != 0:
    print('max train accuracy: {:.2%}'.format(max(train_acc_18)), end=' ')
    print('max test accuracy: {:.2%}'.format(max(test_acc_18)), end=' ')
elif len(train_acc_18) != 0:
    print('max train accuracy: {:.2%}'.format(max(train_acc_18)), end=' ')
else:
    print('max test accuracy: {:.2%}'.format(max(test_acc_18)), end=' ')
plt.show()

# train_acc_50, test_acc_50, y_predict_50, y_target_50 = train_and_test(resnet50, lr, path, label_50, mode)
# show_result(train_acc_50, test_acc_50, label_50)
# cm(y_predict_50, y_target_50, label_50)
# print(train_acc_50, test_acc_50)
# print(label_50, end=' ')
# if len(train_acc_50) != 0 and len(test_acc_50) != 0:
#     print('max train accuracy: {:.2%}'.format(max(train_acc_50)), end=' ')
#     print('max test accuracy: {:.2%}'.format(max(test_acc_50)), end=' ')
# elif len(train_acc_50) != 0:
#     print('max train accuracy: {:.2%}'.format(max(train_acc_50)), end=' ')
# else:
#     print('max test accuracy: {:.2%}'.format(max(test_acc_50)), end=' ')
# plt.show()

# train_acc_pre18, test_acc_pre18, y_predict_pre18, y_target_pre18 = train_and_test(pretrain_resnet18, lr, path, label_pre18, mode)
# show_result(train_acc_pre18, test_acc_pre18, label_pre18)
# cm(y_predict_pre18, y_target_pre18, label_pre18)
# print(train_acc_pre18, test_acc_pre18)
# print(label_pre18, end=' ')
# if len(train_acc_pre18) != 0 and len(test_acc_pre18) != 0:
#     print('max train accuracy: {:.2%}'.format(max(train_acc_pre18)), end=' ')
#     print('max test accuracy: {:.2%}'.format(max(test_acc_pre18)), end=' ')
# elif len(train_acc_pre18) != 0:
#     print('max train accuracy: {:.2%}'.format(max(train_acc_pre18)), end=' ')
# else:
#     print('max test accuracy: {:.2%}'.format(max(test_acc_pre18)), end=' ')
# plt.show()

# train_acc_pre50, test_acc_pre50, y_predict_pre50, y_target_pre50 = train_and_test(pretrain_resnet50, lr, path, label_pre50, mode)
# show_result(train_acc_pre50, test_acc_pre50, label_pre50)
# cm(y_predict_pre50, y_target_pre50, label_pre50)
# print(train_acc_pre50, test_acc_pre50)
# print(label_pre50, end=' ')
# if len(train_acc_pre50) != 0 and len(test_acc_pre50) != 0:
#     print('max train accuracy: {:.2%}'.format(max(train_acc_pre50)), end=' ')
#     print('max test accuracy: {:.2%}'.format(max(test_acc_pre50)), end=' ')
# elif len(train_acc_pre50) != 0:
#     print('max train accuracy: {:.2%}'.format(max(train_acc_pre50)), end=' ')
# else:
#     print('max test accuracy: {:.2%}'.format(max(test_acc_pre50)), end=' ')
# plt.show()

# plt.show()
print('Done!')
