import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot


def generate_linear(n=100, dataset_number=1):  # 1 group of data is 100
    np.random.seed(dataset_number)  # fix random dataset
    pts = np.random.uniform(0, 1, (n, 2))  # creating dataset btw 0 to 1, n is number of data, 2 is input data dimension
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
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predicted result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    # plt.subplot(1, 3, 3)
    # plt.title('Learning curve', fontsize=18)
    # plt.xlabel("Epoch", fontsize=12)
    # plt.ylabel("Loss", fontsize=12)
    # plt.plot(losses)
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

    def backward(self, x, d_input, optimizer):
        return derivative_sigmoid(x) * d_input  # derivative output layer dC/dz = dC/dy * dy/dz chain rule
    # dy/dz - d_fwd(d_sigmoid), d_input, dC/dy - (y_pred - y_hat)


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

    def backward(self, x, d_input, optimizer):
        # print('relu: {}, foward: {}'.format(x, d_input))
        # print('d_relu: {}, x: {}, d_input: {}'.format(len(derivative_relu(x)), x.shape, d_input.shape))
        return d_input * derivative_relu(x)


def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

def derivative_tanh(x):
    return 1 - tanh(x)**2


class tanhLayer:
    def __init__(self):
        pass

    def forward(self, x):
        return tanh(x)

    def backward(self, x, d_input, optimizer):
        # print('relu: {}, foward: {}'.format(x, d_input))
        # print('d_relu: {}, x: {}, d_input: {}'.format(len(derivative_relu(x)), x.shape, d_input.shape))
        return d_input * derivative_tanh(x)


class NoActivationFunction:
    def __init__(self):
        pass

    def forward(self, x):  #
        return x

    def backward(self, x, d_input, optimizer):
        return d_input


def mean_square_error(pred_result, ground_truth):
    sum = 0
    data_size = len(ground_truth)
    for i in range(data_size):
        sum += (ground_truth[i] - pred_result[i]) ** 2
    return sum / (2 * data_size)


def d_mse(pred_result, ground_truth, sigmoid_input):  # derivative mean_square_error
    data_size = len(ground_truth)
    return 2 * (pred_result - ground_truth) * derivative_sigmoid(sigmoid_input) / data_size


class Layer:
    def __init__(self, input_size, output_size, lr, optimizer, dataset_number=1):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr  # learning rate
        np.random.seed(dataset_number)
        self.w = np.random.rand(input_size, output_size)  # create random weight according to layer size
        self.b = np.zeros(output_size)  # bias
        self.lr_w = 0
        self.lr_b = 0
        self.e = 1e-7
        self.lamda = 0.9

    def forward(self, x):
        # print('input: {}, weight: {}'.format(x.shape, self.w.shape))
        return np.add(np.dot(x, self.w), self.b)  # wx+b

    def backward(self, fwd_input, d_input, optimizer):  # forward input and derivative input
        # print('w: {}, d_input: {}'.format(self.w.shape, d_input.shape))
        d_output = d_input @ self.w.T  # array_Y * w_Transpose as Y and w is (y_size, 1) array
        # print('fwd_input: {}, d_input: {}'.format(fwd_input.shape, d_input.shape))
        d_w = fwd_input.T @ d_input  # fwd_input is derivative sigmoid function
        d_b = d_input.mean(axis=0)  # take mean value at axis = 0
        # print('fwd_input: {}, d_input: {}'.format(fwd_input.shape, d_input.shape))
        # print('weight: {}, d_b: {}, w: {}'.format(d_w.shape, d_b.shape, self.w.shape))
        if optimizer == 'Momentum':
            self.lr_w = self.lamda * self.lr_w - self.lr * d_w
            self.lr_b = self.lamda * self.lr_b - self.lr * d_b
            self.w += self.lr_w
            self.b += self.lr_b

        elif optimizer == 'Adagrad':
            self.lr_w = self.lr_w + d_w ** 2
            self.lr_b = self.lr_b + d_b ** 2
            # print('self.lr_w: ', self.lr_w)
            self.w += -self.lr/(np.sqrt(self.lr_w) + self.e) * d_w
            self.b += -self.lr/(np.sqrt(self.lr_b) + self.e) * d_b

        else:  # SGD
            self.w += -self.lr * d_w
            self.b += -self.lr * d_b


        return d_output


class Network:
    def __init__(self, input_size, hidden_layer_size, output_size, learning_rate, optimizer, activation_function):
        # input size = 2, output size = 1, total layer = 2, hidden layer size = any
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        if activation_function == 'Sigmoid':
            activation_layer = SigmoidLayer()
        elif activation_function == 'Relu':
            activation_layer = ReluLayer()
        elif activation_function == 'tanh':
            activation_layer = tanhLayer()
        else:
            activation_layer = NoActivationFunction()
        self.network = [Layer(input_size, hidden_layer_size[0], learning_rate, optimizer), activation_layer,  # Input layer, sigmoid
                        Layer(hidden_layer_size[0], hidden_layer_size[1], learning_rate, optimizer), activation_layer,  # hidden layer, sigmoid
                        Layer(hidden_layer_size[1], output_size, learning_rate, optimizer), activation_layer]  # hidden layer, sigmoid(output)
        # network process: x -> 1st layer (wx -> z1=Wx -> a1=sigmoid(z1)) -> 2nd layer -> z2=Wa2 -> output=sigmoid(a2)

    def forward(self, x):
        layer_output = []
        data_input = x
        for each_layer in self.network:  # to load each layer of network
            layer_output.append(each_layer.forward(data_input))  # append output data to layer_output
            data_input = layer_output[-1]  # last column of output data is next input data
        layer_output = [x] + layer_output  # add input data to output data for later use
        return layer_output

    def backward(self, fwd_input, d_input):
        for each_layer in reversed(range(len(self.network))):  # to load each layer inside network, backward
            d_input = self.network[each_layer].backward(fwd_input[each_layer], d_input, self.optimizer)
            # fwd_input by recorded data from forward calculation on each layer
            # d_input = derivative input result


def run(data_type, epochs, hidden_layer_size, lr, optimizer, act_func):
    if data_type == 'Linear':
        x, y_hat = generate_linear()  # x = input, y_hat = ground truth
        recognize_data = 'Linear'
    else:
        x, y_hat = generate_XOR_easy()
        recognize_data = 'XOR'

    accuracy = 0
    loss = []
    x_size = x.shape  # (data, dimension)
    y_size = y_hat.shape
    network = Network(x_size[-1], hidden_layer_size, y_size[-1], lr, optimizer, act_func)
    for each_epoch in range(epochs):
        z = network.forward(x)  # run network forward part
        y_pred = z[-1]   # last column of data is our prediction value
        network.backward(z, d_mse(y_pred, y_hat, z[-2]))  # run network backward part, z is previous input, d_input is d_mse
        loss.append(mean_square_error(y_pred, y_hat))  # calculate loss by ground truth and prediction value
        print_step = epochs / 20  # creating print step 100
        if each_epoch % print_step == 0:
            print('epoch: {}, loss: {}'.format(each_epoch, float(loss[-1])))
    y_hat_len = y_size[0]  # get the output length
    print('prediction data output: \n', y_pred)
    y_pred = np.around(y_pred)  # round it to the nearest whole number
    for y_i in range(y_hat_len):  # compare every y_data 1 by 1 and count accuracy
        if y_pred[y_i] == y_hat[y_i]:
            accuracy += 1
    accuracy = accuracy / y_hat_len

    print(recognize_data, 'accuracy: {:.2%}'.format(accuracy))
    # print('y_hat: {}, y_pred: {}'.format(y_hat, y_pred))
    show_result(x, y_hat, y_pred, loss)
    return loss


ln = 'Linear'
xor = 'XOR'
sigm = 'Sigmoid'
re = 'Relu'
tan = 'tanh'
none = 'None'
sgd = 'SGD'
adagrad = 'Adagrad'
momentum = 'Momentum'

linear_loss = run(ln, 100000, (4, 8), 1, sgd, sigm)  # run Linear data
xor_loss = run(xor, 100000, (4, 8), 1, sgd, sigm)  # run XOR data

plt.title('Learning curve comparison', fontsize=18)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
pyplot.xscale('log')
# pyplot.yscale('log')
plt.plot(linear_loss, label='Linear Loss')
plt.plot(xor_loss, label='XOR Loss')
plt.legend()
plt.show()