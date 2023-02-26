import argparse
import random
from sympy import *
from math import *
from prettytable import PrettyTable
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Neural network for calculating the value of a boolean function', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('function', type=str, help="Boolean function. Example 'not(x1 and x2) and x3 and x4'")
parser.add_argument('--act_func', type=str, default='', metavar='string', help="Your second activation function. Example = '1/(1+exp(-net))'")
parser.add_argument('--n', type=float, metavar='float', default = 0.3, help='Learning rate')
parser.add_argument('--size', type=int, metavar='int', default = 16, help='Minimum test set size')
args = parser.parse_args()


def generate_truth_table(func) -> list:
    table = []
    for x1 in range(2):
        for x2 in range(2):
            for x3 in range(2):
                for x4 in range(2):
                    value = int(eval(func))
                    table.append([x1, x2, x3, x4, value])
    return table


class Neuron:
    def __init__(self, table: list, n, c_func: str):
        self.w_array = [0, 0, 0, 0, 0]
        self.truth_table = table
        self.n = n
        self.custom_func = c_func
        self.print_table = PrettyTable()
        self.print_table.field_names = ["Эпоха", "Вектор весов", "Выходной вектор У", "Число ошибок Е"]

    def __calculate_net(self, x_array) -> float:
        net = self.w_array[0]
        for i in range(1, len(self.w_array)):
            net += x_array[i-1]*self.w_array[i]
        return net

    def __calculate_delta_w(self, x, d, net_value, custom_function: bool):
        if not custom_function:
            return self.n*x*d

        if self.custom_func == '':
            raise KeyError('Empty activation function')
        net = symbols('net')
        dif = str(diff(self.custom_func))
        net = net_value
        return self.n*eval(dif)*x*d

    def __calculate_y(self, net, custom_function: bool):
        if not custom_function:
            if net >= 0:
                return 1
            return 0

        if self.custom_func == '':
            raise KeyError('Empty activation function')
        if eval(self.custom_func) >= 0.5:
            return 1
        return 0

    def learn(self, set_size, custom_function = False) -> list:
        self.w_array = [0, 0, 0, 0, 0]
        self.print_table = PrettyTable()
        self.print_table.field_names = ["Эпоха", "Вектор весов", "Выходной вектор У", "Число ошибок Е"]
        errors_num = 1
        errors_array = []

        test_set = set()
        if set_size < 16:
            while len(test_set) != set_size:
                test_set.add(random.randint(0, 15))
            print(test_set)

        k = 0
        while errors_num != 0:
            k+=1
            if k == 200:
                return [None, None]
            y_array = []

            # counting errors
            errors_num = 0
            for i in range(len(self.truth_table)):

                net = self.__calculate_net(self.truth_table[i])
                y_array.append(self.__calculate_y(net, custom_function))

                if self.truth_table[i][-1] - y_array[-1] != 0:
                    errors_num += 1  

            print_w_array = []
            for w in self.w_array:
                print_w_array.append(round(w, 3))
            self.print_table.add_row([k, print_w_array, y_array, errors_num])
            errors_array.append(errors_num)

            # correcting weights
            if errors_num > 0:
                for i in range(len(self.truth_table)):
                    if set_size < 16:
                        if not i in test_set:
                            continue
                    net = self.__calculate_net(self.truth_table[i])
                    y = self.__calculate_y(net, False)
                    delta = self.truth_table[i][-1] - y

                    if delta != 0:  
                        self.w_array[0] += self.__calculate_delta_w(1, delta, net, custom_function) 
                        for j in range(1, len(self.w_array)):
                            self.w_array[j] += self.__calculate_delta_w(self.truth_table[i][j-1], delta, net, custom_function)

        return self.print_table, errors_array


if __name__ == "__main__":
    neuron = Neuron(generate_truth_table(args.function), args.n, args.act_func)
    result = [None, None]
    while result[0] == None:
        if args.act_func == '':
            result = neuron.learn(args.size)
        else:    
            result = neuron.learn(args.size, True)

    print(result[0])
    plt.plot([i+1 for i in range(len(result[1]))], result[1])
    plt.title("Errors graph")   
    plt.ylabel('Errors (E)')   
    plt.xlabel('Epochs (k)')
    plt.grid(True)
    plt.show()
