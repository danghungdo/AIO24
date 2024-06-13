import math
import random


# Ex1
def calculate_f1_score(tp, fp, fn):
    assert isinstance(tp, int), 'tp must be int'
    assert isinstance(fp, int), 'fp must be int'
    assert isinstance(fn, int), 'fn must be int'
    assert tp > 0 and fp > 0 and fn > 0, 'tp and fp and fn must be greater than zero'
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f'Precision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}')
    return f1_score


# Ex2
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return max(0, x)


def elu(x, alpha=0.01):
    return x if x > 0 else alpha * (math.exp(x) - 1)


# Given function
def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def calculate_activation():
    x = input('x = ')
    activation_func = input(
        'Activation function (sigmoid|relu|elu) = ').lower()
    supported_funcs = {'sigmoid': sigmoid, 'relu': relu, 'elu': elu}
    if activation_func not in supported_funcs.keys():
        print(f'{activation_func} is not supported')
        return
    if not is_number(x):
        print('x must be a number')
        return
    x = float(x)
    result = supported_funcs[activation_func](x)
    print(f'{activation_func}: f({x}) = {result}')
    return result


# Ex3
def mae(y_true, y_pred):
    return sum([abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]) / len(y_true)


def mse(y_true, y_pred):
    return sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))]) / len(y_true)


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def calculate_loss():
    num_samples = input('Number of samples which are generated: ')
    if not num_samples.isnumeric() or int(num_samples) <= 0:
        print('Number of samples must be a positive integer')
        return
    loss_name = input('Loss function (mae|mse|rmse): ').lower()
    supported_funcs = {'mae': mae, 'mse': mse, 'rmse': rmse}
    if loss_name not in supported_funcs.keys():
        print(f'{loss_name} is not supported')
        return
    logs = [f'Loss name: {loss_name}']
    y_true, y_predict = [], []
    for i in range(int(num_samples)):
        pred = random.uniform(0.0, 10.0)
        target = random.uniform(0.0, 10.0)
        y_true.append(target)
        y_predict.append(pred)
        log = f'sample: {i+1}, pred: {pred}, true: {target}'
        logs.append(log)
    print(', '.join(logs))
    loss = supported_funcs[loss_name](y_true, y_predict)
    print(f'final {loss_name} = {loss}')


# Ex4
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)


N_MUST_BE_GREATER_THAN_ZERO = 'n must be greater than zero'


def approximate_sin(x, n):
    assert n > 0, N_MUST_BE_GREATER_THAN_ZERO
    approx = 0
    for i in range(n):
        approx += (-1) ** i * x ** (2 * i + 1) / factorial(2 * i + 1)
    return approx


def approximate_cos(x, n):
    assert n > 0, N_MUST_BE_GREATER_THAN_ZERO
    approx = 0
    for i in range(n):
        approx += (-1) ** i * x ** (2 * i) / factorial(2 * i)
    return approx


def approximate_sinh(x, n):
    assert n > 0, N_MUST_BE_GREATER_THAN_ZERO
    approx = 0
    for i in range(n):
        approx += x ** (2 * i + 1) / factorial(2 * i + 1)
    return approx


def approximate_cosh(x, n):
    assert n > 0, N_MUST_BE_GREATER_THAN_ZERO
    approx = 0
    for i in range(n):
        approx += x ** (2 * i) / factorial(2 * i)
    return approx


# Ex5
def md_nre_single_sample(y_inner, y_hat_inner, n, p):
    return (y_inner ** (1 / n) - y_hat_inner ** (1 / n)) ** p


if __name__ == '__main__':
    calculate_loss()
    # Q1
    print('Question 1: ')
    assert round(calculate_f1_score(tp=2, fp=3, fn=5), 2) - 0.33 < 0.01
    print(round(calculate_f1_score(tp=2, fp=4, fn=5), 2))
    print('-------------------')
    # Q2
    print('Question 2: ')
    assert is_number(3) - 1.0 < 0.01
    assert is_number('-2a') < 0.01
    print(is_number(1))
    print(is_number('n'))
    print('-------------------')
    # Q4
    print('Question 4: ')
    assert round(sigmoid(3), 2) - 0.95 < 0.01
    print(round(sigmoid(2), 2))
    print('-------------------')
    # Q5
    print('Question 5: ')
    assert round(elu(1, 0.01)) == 1
    print(round(elu(-1, 0.01), 2))
    print('-------------------')
    # Q6
    print('Question 6: ')
    print(round(calculate_activation(), 2))
    print('-------------------')
    # Q7
    print('Question 7: ')
    y = [1]
    y_hat = [6]
    assert mae(y, y_hat) == 5
    y = [2]
    y_hat = [9]
    print(mae(y, y_hat))
    print('-------------------')
    # Q8
    print('Question 8: ')
    y = [4]
    y_hat = [2]
    assert mse(y, y_hat) == 4
    print(mse([2], [1]))
    print('-------------------')
    # Q9
    print('Question 9: ')
    assert round(approximate_cos(1, 10), 2) - 0.54 < 0.01
    print(round(approximate_cos(3.14, 10), 2))
    print('-------------------')
    # Q10
    print('Question 10: ')
    assert round(approximate_sin(1, 10), 4) - 0.8415 < 0.01
    print(round(approximate_sin(3.14, 10), 4))
    print('-------------------')
    # Q11
    print('Question 11: ')
    assert round(approximate_sinh(1, 10), 2) - 1.18 < 0.01
    print(round(approximate_sinh(3.14, 10), 2))
    print('-------------------')
    # Q12
    print('Question 12: ')
    assert round(approximate_cosh(1, 10), 2) - 1.54 < 0.01
    print(round(approximate_cosh(3.14, 10), 2))
    print('-------------------')
