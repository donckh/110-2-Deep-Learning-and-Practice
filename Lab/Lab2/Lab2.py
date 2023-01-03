import itertools

import numpy as np
import matplotlib.pyplot as plt
import copy


def generate_linear(n=100):  # 1 group of data is 100
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy(n=11):  # 1 group of data is 11
    import numpy as np
    inputs = []
    labels = []
    for i in range(n):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y, losses):
    plt.subplot(1, 3, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 3, 2)
    plt.title('Predicted result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 3, 3)
    plt.title('Learning curve', fontsize=18)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.plot(losses)
    plt.tight_layout()
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(sigmoid(x), 1.0 - sigmoid(x))


class SigmoidLayer:
    def __init__(self):
        pass

    def forward(self, x):
        return sigmoid(x)

    def backward(self, x, d_input):
        return derivative_sigmoid(x) * d_input  # output layer gradient dC/dz = dC/dy * dy/dz
    # dy/dz - fwd grad(sigmoid_grad), input_grad, dC/dy - (y_pred - y_hat) * sigmoid_grad


def relu(x):  # a = coefficient, a = 0 for normal relu
    return np.maximum(0, x)


def derivative_relu(x):
    d_relu = x > 0  # when x > 0, d_relu = true
    return d_relu


class ReluLayer:
    def __init__(self):
        pass

    def forward(self, x):
        return relu(x)

    def backward(self, x, d_input):
        # print('relu: {}, foward: {}'.format(x, d_input))
        # print('d_relu: {}, x: {}, d_input: {}'.format(len(derivative_relu(x)), x.shape, d_input.shape))
        return d_input * derivative_relu(x)


class NoActivationFunction:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, x, d_input):
        return d_input


def mean_square_error(pred_result, ground_truth):
    sum = 0
    for i in range(len(ground_truth)):
        sum += (ground_truth[i] - pred_result[i]) ** 2
    return sum / len(ground_truth)


def d_mse(pred_result, ground_truth, sigmoid_input):  # mean_square_error gradient
    return 2 * (pred_result - ground_truth) * derivative_sigmoid(sigmoid_input)


def accuracy_calculation(prediction, ground_truth):
    return np.sqrt(((prediction - ground_truth) ** 2) / len(ground_truth))


class Layer:
    def __init__(self, input_size, output_size, lr):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr  # learning rate
        self.w = np.random.rand(input_size, output_size)  # create random weight according to layer size
        self.b = np.zeros(output_size)  # bias

    def forward(self, x):
        # print('input: {}, weight: {}'.format(x.shape, self.w.shape))
        return np.add(np.dot(x, self.w), self.b)  # wx+b

    def backward(self, fwd_input, d_input):  # forward input and derivative input
        # print('w: {}, d_input: {}'.format(self.w.shape, d_input.shape))
        d_output = d_input @ self.w.T  # array_Y * w_Transpose as Y and w is (y_size, 1) array
        # print('fwd_input: {}, d_input: {}'.format(fwd_input.shape, d_input.shape))
        d_w = fwd_input.T @ d_input  # fwd_input is derivative sigmoid function
        d_b = d_input.mean(axis=0)
        # print('fwd_input: {}, d_input: {}'.format(fwd_input.shape, d_input.shape))
        # print('weight: {}, d_b: {}, w: {}'.format(d_w.shape, d_b.shape, self.w.shape))
        self.w += -self.lr * d_w
        self.b += -self.lr * d_b
        return d_output


class Network:
    def __init__(self, input_size, hidden_layer_size, output_size, learning_rate, activation_function):
        # input size = 2, output size = 1, total layer = 2, hidden layer size = any
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        if activation_function == 'Sigmoid':
            activation_layer = SigmoidLayer()
        elif activation_function == 'Relu':
            activation_layer = ReluLayer()
        else:
            activation_layer = NoActivationFunction()
        self.network = [Layer(input_size, hidden_layer_size[0], learning_rate), activation_layer,  # Input layer, sigmoid
                        Layer(hidden_layer_size[0], hidden_layer_size[1], learning_rate), activation_layer,  # hidden layer, sigmoid
                        Layer(hidden_layer_size[1], output_size, learning_rate), activation_layer]  # hidden layer, sigmoid(output)
        # network process: x -> 1st layer (wx -> z1=Wx -> a1=sigmoid(z1)) -> 2nd layer -> z2=Wa2 -> output=sigmoid(a2)

    def forward(self, x):
        layer_output = []
        data_input = x
        for each_layer in self.network:  # to load each layer inside network
            layer_output.append(each_layer.forward(data_input))
            data_input = layer_output[-1]
        layer_output = [x] + layer_output
        return layer_output

    def backward(self, fwd_input, d_input):
        for each_layer in reversed(range(len(self.network))):  # to load each layer inside network, backward
            d_input = self.network[each_layer].backward(fwd_input[each_layer], d_input)
            # fwd_input by recorded data from forward calculation on each layer
            # d_input = derivative input result


def run(data_type, epochs, hidden_layer_size, lr, act_func):
    if data_type == 'Linear':
        x, y_hat = generate_linear()  # x = input, y_hat = ground truth
        recognize = 'Linear'
    else:
        x, y_hat = generate_XOR_easy()
        recognize = 'XOR'

    accuracy = 0
    loss = []
    x_size = x.shape  # (data, dimension)
    y_size = y_hat.shape
    network = Network(x_size[-1], hidden_layer_size, y_size[-1], lr, act_func)
    for each_epoch in range(epochs):
        z = network.forward(x)
        y_pred = z[-1]
        network.backward(z, d_mse(y_pred, y_hat, z[-2]))
        loss.append(mean_square_error(y_pred, y_hat))
        print_step = epochs / 100
        if each_epoch % print_step == 0:
            print('epoch: {}, loss: {}'.format(each_epoch, float(loss[-1])))
    y_hat_len = len(y_hat)
    y_pred = np.around(y_pred)
    for y_i in range(y_hat_len):
        if y_pred[y_i] == y_hat[y_i]:
            accuracy += 1
    accuracy = accuracy / y_hat_len
    print(recognize, 'accuracy: {:.2%}'.format(accuracy))
    # print('y_hat: {}, y_pred: {}'.format(y_hat, y_pred))
    show_result(x, y_hat, y_pred, loss)


ln = 'Linear'
xor = 'XOR'
sigm = 'Sigmoid'  # 1000, (4, 4), 0.1
re = 'Relu'
none = 'None'

# run(ln, 10000, (4, 8), 0.1, sigm)  # run Linear data
# run(xor, 100000, (4, 8), 0.1, sigm)  # run XOR data

# run(ln, 10000, (4, 8), 0.003, re)  # run Linear data
# run(xor, 400, (4, 8), 0.005, re)  # run XOR data

# run(ln, 10000, (4, 8), 0.003, none)  # run Linear data
run(xor, 150, (4, 8), 0.001, none)  # run XOR data