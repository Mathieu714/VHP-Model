import os
import random
import math
from numpy import linalg
import numpy as np

# This version rewrote empower logic


# y = a_1x^n + a_2x^n + a_3x^n+ ... + a_nx^n
# info structure: [ [a1, a2, a3, ...], [n, n, n, ...] ]: [ [weight_array], [power_array] ]
# data structure: [ x1, x2, x3, ..., xn, y]

AlterPowerError = Exception()
EmpowerError = Exception()
EvaluationError = Exception()

global seed
seed = None


def calc(data_array, weight_array, power_array):
    # returns calculated y
    y = 0
    for i in range(0, len(data_array)):
        y += (data_array[i] ** power_array[i]) * weight_array[i]

    return y


def percent_acc(obj, center):
    error = abs(obj - center)
    mod_obj = center - error

    # if mod_obj < 0:
    #     raise EvaluationError

    percent = mod_obj / center
    return percent

# is ref actual? why compare obj to actual
# compare obj to other obj: raise/lower
# write choice mechanism for fit, not only raise/lower

#
# def evaluation(obj, ref, pred, target):
#     delta = abs(percent_error(obj, pred) - percent_error(ref, pred))
#     print(f'delta is {delta}')
#     return delta > target
#


def alter_power(power_array, add: bool):
    print('-----altering power...')
    print(f'power is adding {add}')
    print(f'initial power_array is {power_array}')
    for i in range(0, len(power_array)):
        print(f'initial power is {power_array[i]}')
        if add:
            power_array[i] += 1
        elif not add:
            power_array[i] -= 1
        else:
            raise AlterPowerError

        print(f'final power is {power_array[i]}')

    print(f'final power_array is {power_array}')
    print('-----...altering power complete')
    return power_array


def apply_power(power_array, data_array):
    print('-----applying power to data...')
    for j in data_array:
        for i in range(0, len(j)):
            print(f'initial parameter is {data_array[data_array.index(j)][i]}')
            data_array[data_array.index(j)][i] = data_array[data_array.index(j)][i] ** power_array[i]
            print(f'final parameter is {data_array[data_array.index(j)][i]}')

    print('-----...power applying complete')
    return data_array


def true(data, d, itr, power_array):
    print('-----solving true parameters...')
    coeff_mat = []
    const_mat = []
    for i in range(0, d):
        coeff_mat.append(data[(itr + i)%len(data)][0:d])
        const_mat.append(data[(itr + i)%len(data)][-1])
        print(f'data line {i} used')
    coeff_mat = noise(apply_power(power_array, coeff_mat))
    print(f'the powered coeff mat is {coeff_mat}')
    print(f'the const mat is {const_mat}')
    var_mat = linalg.solve(coeff_mat, const_mat)
    print(f'the solved mat is {var_mat}')
    print('-----...true parameters solved.')
    return var_mat


def noise(data_array):
    print('-----noise adding...')
    random.seed = seed
    print(f'seed is {seed}')
    bound = 0.00000000001
    print(f'noise addition is between {-bound} and {bound}')
    for i in data_array:
        for j in range(0, len(i)):
            print(f'initial parameter is {data_array[data_array.index(i)][j]}')
            data_array[data_array.index(i)][j] += random.uniform(-bound, bound)
            print(f'final parameter is {data_array[data_array.index(i)][j]}')

    print('-----...noise adding complete')
    return data_array


def train_test_split(data, r):
    print(f'-----train test split function called. result will be {r} train and {1-r} test...')
    length = len(data)
    print(f'total length of data is {length}')
    train_len = int(np.ceil(length*r))
    print(f'there will be {train_len} train data')
    test_len = length - train_len
    print(f'there will be {test_len} test data')
    train_data = data[:train_len]
    test_data = data[-test_len:]

    print('-----...train test split function executed successfully')
    return train_data, test_data


class Model:
    def __init__(self, dim, lr):
        self.itr = 0
        self.augmented_array = []
        self.prev_true = []
        self.dim = dim
        self.lr = lr
        self.history = []
        self.best_history = []
        self.best_weight = []
        self.best_acc = 0

    def init_model(self):
        print('-----initializing model...')
        d = self.dim
        weight_array = []
        power_array = []
        for i in range(0, d):
            x = random.random()
            print(f'the init value of parameter {i} is {x}')
            weight_array.append(x)
            power_array.append(1)

        self.augmented_array = [weight_array, power_array]
        self.best_weight = [weight_array, power_array]
        print(f'the init aug array is {self.augmented_array}')
        print('-----...initialization complete')

    def fit(self, replace=True):
        print(f'-----fitting starts, replacement is set to be {replace}...')
        lr = self.lr
        print(f'learning rate is {lr}')
        new_pred = []
        weight_array = self.augmented_array[0]
        print(f'old aug array is {self.augmented_array}')
        dim = self.dim
        prev_true = self.prev_true
        print(f'the previous true is {prev_true}')

        length = len(weight_array)
        for i in range(0, length):
            print(f'fitting parameter number {i}')
            print(f'range upper bound is {weight_array[i]}')
            print(f'range lower bound is {weight_array[i]+lr*(prev_true[i]-weight_array[i])}')
            print(f'the guess is between {weight_array[i]} and {weight_array[i]+lr*(prev_true[i]-weight_array[i])}')
            pred = random.uniform(weight_array[i], weight_array[i]+lr*(prev_true[i]-weight_array[i]))
            print(f'the guess is {pred}')
            new_pred.append(pred)

        if replace:
            self.augmented_array[0] = new_pred
            print('successfully replaced')
        else:
            print('not replaced')
            return new_pred

        print(f'new aug array is {self.augmented_array}')
        print(f'-----...fitting complete. replaced {replace}')

    def update_true(self, data):
        print('-----updating true...')
        weight_array, power_array = self.augmented_array
        dim = self.dim
        self.prev_true = true(data, dim, self.itr, power_array)
        print('-----...finished updating true')

    def empower(self, data_array, y, test):
        print('-----empowering...')
        y = y
        weight_array, power_array = self.augmented_array
        print(f'old aug array is {self.augmented_array}')

        current_acc = calc(data_array, weight_array, power_array)
        print(f'the current output is {current_acc}')

        temp_array = self.fit(replace=False)
        temp_powered = calc(data_array, temp_array, power_array)
        print(f'the output after fit is {temp_powered}')

        raise_powered = calc(data_array, weight_array, alter_power(power_array, add=True))
        print(f'the output after raising power is {raise_powered}')
        lower_powered = calc(data_array, weight_array, alter_power(power_array, add=False))
        print(f'the output after lowering power is {lower_powered}')

        temp_evaluation = percent_acc(temp_powered, y)
        raise_evaluation = percent_acc(raise_powered, y)
        lower_evaluation = percent_acc(lower_powered, y)

        choices = [temp_evaluation, raise_evaluation, lower_evaluation]
        print(f'the percent acc after fit, raise, and lower is {choices}')
        print(f'the best acc is {max(choices)}------------------------------------------------------------------------')
        choice = choices.index(max(choices))
        print(f'choice is {choice}')

        if choice == 0:
            self.augmented_array = [temp_array, power_array]
        elif choice == 1:
            self.augmented_array = [weight_array, alter_power(power_array, add=True)]
        elif choice == 2:
            self.augmented_array = [weight_array, alter_power(power_array, add=False)]
        else:
            raise EmpowerError

        total = 0
        for i in range(0, len(test)):
            print(test[i][:-1])
            p = self.predict(test[i][:-1])
            print(p)
            acc = percent_acc(p, test[i][-1])
            print(f'accuracy is {acc}')
            total += acc

        val_acc = total/len(test)

        best = 0
        for i in range(0, len(test)):
            print(test[i][:-1])
            p = self.predict(test[i][:-1], best=True)
            print(p)
            acc = percent_acc(p, test[i][-1])
            print(f'accuracy is {acc}')
            best += acc

        best_val_acc = best / len(test)

        print(f'val acc is {val_acc}')
        print(f'best val acc is {best_val_acc}')
        # if val_acc > 0:
        #     self.history.append(val_acc)
        # if best_val_acc > 0:
        #     self.best_history.append(best_val_acc)
        self.history.append(val_acc)
        self.best_history.append(best_val_acc)

        if val_acc > self.best_acc:
            print('acc is best -> replacement')
            self.best_acc = val_acc
            self.best_weight = self.augmented_array

        print(f'new aug array is {self.augmented_array}')
        print('-----...finished empowering')

    def train(self, data, test, max):
        print('---------------training starts...')
        length = len(data)
        print(f'the length of the data is {length}')
        i = 1
        self.init_model()
        print(f'the initialized aug_array is {self.augmented_array}')
        self.update_true(data)
        print(f'the initialized true data is {self.prev_true}')
        while i < max:
            self.empower(data[i%length][:-1], data[i%length][-1], test)
            print(f'model empowered {i} times--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
            i += 1

        print('---------------...training complete-----------------------------------------------------------------------------------------------------------------------------------------')

    def predict(self, data_array, best=False):
        print('-----predicting starts...')
        print(self.augmented_array)
        print(self.best_weight)
        weight_array, power_array = self.augmented_array
        weight_array_best, power_array_best = self.best_weight
        print(f'given aug array is {self.augmented_array}')
        prediction = calc(data_array, weight_array, power_array)
        best_prediction = calc(data_array, weight_array_best, power_array_best)
        print(f'prediction is {prediction}')

        print('-----...prediction complete')
        if not best:
            return prediction
        if best:
            return best_prediction
