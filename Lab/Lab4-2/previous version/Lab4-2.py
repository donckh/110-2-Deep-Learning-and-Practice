import dataloader
import torch
import torch.nn as nn
import torch.utils.data as tu
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import ConfusionMatrix


# import sys
# print(sys.getrecursionlimit())
# sys.setrecursionlimit(2000)
# print(sys.getrecursionlimit())
torch.cuda.empty_cache()
print(torch.cuda.get_device_name(), torch.cuda.device_count())
batches = 3
epochs = 11
root = './data/'
train = 'train'
test = 'test'
gpu = torch.device("cuda")

train_data = dataloader.RetinopathyLoader(root, train)
train_data_loader = tu.DataLoader(train_data, batch_size=batches)


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
# print(pretrain_resnet50)

def train_and_test(model, lr):
    model.to(gpu)
    train_ep_acc = []
    test_ep_acc = []
    print('Running on GPU: ', next(model.parameters()).is_cuda)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        train_acc = []
        train_acc_total = []
        for x_batch, y_batch in train_data_loader:
            x = x_batch.to(gpu)
            y = y_batch.to(gpu)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y)

            loss.backward()
            optimizer.step()
            train_acc = y_pred.argmax(axis=1) == y

            # print('test_pred: {}, y: {}, train_acc: {}'.format(y_pred, y, train_acc))
            train_acc_total = data_reshape(train_acc_total, train_acc)
        # y_pred_arr_total = np.array(y_pred_arr_total)
        # y_target_arr_total = np.array(y_target_arr_total)
        # print('train_acc_total: {}, y: {}'.format(train_acc_total, train_acc_total))
        # print(type(y_arr_total))
        train_ep_acc.append((sum(train_acc_total)/len(train_acc_total)))
        # print('*********************************************************************************')
        # print('train_ep_acc: {}, \nsum: {}, len: {}'.format(train_ep_acc, sum(train_ep_acc), len(train_ep_acc)))

        model.eval()
        with torch.no_grad():
            test_acc = []
            test_acc_total = []
            y_test_pred_arr_total = []
            y_test_target_arr_total = []
            for x_batch, y_batch in test_data_loader:
                x = x_batch.to(gpu)
                y = y_batch.to(gpu)
                y_pred = model(x)
                test_acc = y_pred.argmax(axis=1) == y  # take the highest probi one to compare as ground truth
                test_acc_total = data_reshape(test_acc_total, test_acc)
                y_test_pred_arr_total = data_reshape(y_test_pred_arr_total, y_pred.argmax(axis=1))
                y_test_target_arr_total = data_reshape(y_test_target_arr_total, y)
            y_test_pred_arr_total = np.array(y_test_pred_arr_total)
            y_test_target_arr_total = np.array(y_test_target_arr_total)
            test_ep_acc.append((sum(test_acc_total)/len(test_acc_total)))
            print('\repochs: {:.2%}'.format(epoch/(epochs-1)), end='')
    print('\n')
    return train_ep_acc, test_ep_acc, y_test_pred_arr_total, y_test_target_arr_total  # y_pred_arr_total, y_target_arr_total,


def show_result(train_acc, test_acc, name):
    train = torch.tensor(train_acc, device='cpu')
    test = torch.tensor(test_acc, device='cpu')
    plt.title('Resnet18', fontsize=18)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('epochs', fontsize=12)
    plt.plot(train, label=name + '_train')
    plt.plot(test, label=name + '_test')
    # plt.plot([0.5,0.5], label='testing')
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")  # , borderaxespad=0
    plt.tight_layout()

def data_reshape(y_store, y_tensor):
    y_tensor_arr = torch.tensor(y_tensor, device='cpu')
    # print('type: ', len(y_tensor_arr), len(y_tensor))
    # y_store = y_tensor_arr.flatten(0)
    # y_store = y_store.numpy()
    y_tensor_arr = y_tensor_arr.numpy()
    for batch in range(len(y_tensor)):
        y_store.append(y_tensor_arr[batch])
    # print('y: {}, y_total: {}'.format(y_tensor, y_store))
    # print('y_store: {}'.format(y_store))
    return y_store


def cm(y_pred, y_true, name):
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

    plt.show()


label_18 = 'Resnet18'
label_50 = 'Resnet50'
label_pre18 = 'Pretrain_Resnet18'
label_pre50 = 'Pretrain_Resnet50'

lr = 1e-3
train_acc_18, test_acc_18, y_predict_18, y_target_18 = train_and_test(resnet18, lr)  # y_train_pred, y_train_target,
show_result(train_acc_18, test_acc_18, label_18)
# print('\ntrain size: {}, test_size: {}'.format(len(train_acc), len(test_acc)))
# print(train_acc_18, test_acc_18)
# print(max(train_acc_18), len(train_acc_18), train_acc_18)
#
train_acc_50, test_acc_50, y_predict_50, y_target_50 = train_and_test(resnet50, lr)
show_result(train_acc_50, test_acc_50, label_50)
# print(train_acc_p18, test_acc_p18)

train_acc_pre18, test_acc_pre18, y_predict_pre18, y_target_pre18 = train_and_test(pretrain_resnet18, lr)
show_result(train_acc_pre18, test_acc_pre18, label_pre18)

train_acc_pre50, test_acc_pre50, y_predict_pre50, y_target_pre50 = train_and_test(pretrain_resnet50, lr)
show_result(train_acc_pre50, test_acc_pre50, label_pre50)

print(label_18, 'max train accuracy: {:.2%}, max test accuracy: {:.2%}'.format(max(train_acc_18).item(), max(test_acc_18).item()))
print(label_50, 'max train accuracy: {:.2%}, max test accuracy: {:.2%}'.format(max(train_acc_50).item(), max(test_acc_50).item()))
print(label_pre18, 'max train accuracy: {:.2%}, max test accuracy: {:.2%}'.format(max(train_acc_pre18).item(), max(test_acc_pre18).item()))
print(label_pre50, 'max train accuracy: {:.2%}, max test accuracy: {:.2%}'.format(max(train_acc_pre50).item(), max(test_acc_pre50).item()))
plt.show()

cm(y_predict_18, y_target_18, label_18)
cm(y_predict_50, y_target_50, label_50)
cm(y_predict_pre18, y_target_pre18, label_pre18)
cm(y_predict_pre50, y_target_pre50, label_pre50)
# plt.show()
print('Done!')
