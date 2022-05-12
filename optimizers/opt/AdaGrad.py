from opt.generalOpt import generalOpt
import math
from sympy import *
import numpy as np
from opt.functions import getFunction

"""
paper: Adaptive subgradient methods for online learning and stochastic optimization

First generation adaptive learning rate method.
Auto-adjust learning rate for each parameter basing on the accumulation of previous gradient.

It adapts the learning rate to the parameters, performing larger updates for infrequent and 
smaller updates for frequent parameters

Advantage:
a) Auto-adjust learning rate
b) improve the robustness of SGD
Shortcoming:
a) Learning rate is monotonically decreasing, which makes learning rate too small to learn sufficient knowledge along with the training.
b) The converging speed increasingly slow down along with the training, especially with a small learning rate.

Actually, in the perspective of improving SGD's robustness, this method can automatic decrease learning rate to approach the optima as the training goes. (annealing)
However, the reducing process is uncontrollable, which actually is controlled by gradient. Specifically, once the beginning gradient is large, 
the learning rate shrinks dramatically and never becomes large any longer, which will impede the algorithm converge to optima and stuck in local optima.
Also, Adagrad can avoid diverge due to the accumulation of gradient in the denominator.
"""


class AdaGrad(generalOpt):
    def __init__(self,
                 objectFunction,
                 lr,
                 gamma,
                 X,
                 Y,
                 epsilon=1e-06
                 ):
        """
        :param lr: learning rate
        :param objectFunction: loss function (self-define)
        :param epsilon: smooth term (avoid division by zero)
        :param X
        :param Y
        """
        self.lr = lr
        self.objectFunction = objectFunction
        self.X = X
        self.Y = Y
        self.name = 'AdaGrad'
        self.G = np.zeros((2,))
        self.epsilon = epsilon

    def evaluateGrid(self, param):
        """
        compute gradient.
        :param param: current values.
        :return: partial derivative.
        """
        return np.array([diff(self.objectFunction, self.X).evalf(subs={self.X: param[0], self.Y: param[1]}),
                         diff(self.objectFunction, self.Y).evalf(subs={self.X: param[0], self.Y: param[1]})], dtype='float')

    def step(self, param):
        """
        θ_t = θ_{t-1} - (lr/sqrt(G + ε))▽J(θ)
        :param param: parameters
        :return: updated parameter
        """
        current_grad = self.evaluateGrid(param)
        self.G = self.G + current_grad ** 2
        param = param - np.multiply(self.lr / np.sqrt(self.G + self.epsilon), current_grad)

        return param

    def getName(self):
        return self.name


if __name__ == '__main__':
    X, Y = symbols('X Y', real=True)
    J = getFunction(X, Y, 'Rosenbrock', a=1, b=1)  # object function
    lr = 0.5
    gamma = 0.9
    optimizer = AdaGrad(J, lr, gamma, X, Y)
    theta = np.array([-2, 3])
    lossBefore, lossUpdated = math.inf, 0  # used to define stop rule
    step = 1
    while abs(lossUpdated - lossBefore) > 1e-08:  # threshold
        lossBefore = J.evalf(subs={X: theta[0], Y: theta[1]})
        theta = optimizer.step(theta)
        lossUpdated = J.evalf(subs={X: theta[0], Y: theta[1]})
        if step % 10 == 0:
            print('step: {}; loss: {}'.format(step, lossUpdated))
        step += 1
        if lossUpdated == math.inf:  # check whether diverge
            raise Exception('Diverge occupy! Try smallest learning rate!')
    print('total update step: {}'.format(step))
