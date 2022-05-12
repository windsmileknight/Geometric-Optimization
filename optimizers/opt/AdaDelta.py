from opt.generalOpt import generalOpt
import math
from sympy import *
import numpy as np
from opt.functions import getFunction

"""
paper: ADADELTA: AN ADAPTIVE LEARNING RATE METHOD

Adadelta is evolution vision of Adagrad, which solves two problem: 
1) introducing exponentially decay window parameter to handle gradient accumulation problem in Adagrad;
2) introducing Hessian matrix approximation to solve mismatch between the units of parameter and of gradient. Also, remove
the requirement of learning rate.
"""


class AdaDelta(generalOpt):
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
        :param gamma: decay gamma
        :param epsilon: smooth term (avoid division by zero)
        :param X
        :param Y
        """
        self.lr = lr
        self.objectFunction = objectFunction
        self.gamma = gamma
        self.X = X
        self.Y = Y
        self.name = 'AdaDelta'
        self.epsilon = epsilon
        self.expected_grad = np.zeros((2,))
        self.expected_theta = np.zeros((2,))

    def evaluateGrid(self, param):
        """
        compute gradient.
        :param param: current values.
        :return: partial derivative.
        """
        return np.array([diff(self.objectFunction, self.X).evalf(subs={self.X: param[0], self.Y: param[1]}),
                         diff(self.objectFunction, self.Y).evalf(subs={self.X: param[0], self.Y: param[1]})],
                        dtype='float')

    def step(self, param):
        """
        E[g²]_t = γE[g²]_{t-1} + (1 - γ)g²_t
        Δθ_t = -(RMS[Δθ]_{t-1} / RMS[g]_t) g_t
        E[Δθ²]_t = γE[Δθ²]_{t-1} + (1 - γ)Δθ²_t
        θ_{t+1} = θ_t + Δθ_t
        :param param: parameters
        :return: updated parameter
        """
        current_grad = self.evaluateGrid(param)
        self.expected_grad = self.gamma * self.expected_grad + (1 - self.gamma) * current_grad ** 2
        RMS_grad = np.sqrt(self.expected_grad + self.epsilon)
        RMS_theta = np.sqrt(self.expected_theta + self.epsilon)
        delta_theta = np.multiply(-RMS_theta / RMS_grad, current_grad)
        self.expected_theta = self.gamma * self.expected_theta + (1 - self.gamma) * delta_theta ** 2
        return param + delta_theta

    def getName(self):
        return self.name


if __name__ == '__main__':
    X, Y = symbols('X Y', real=True)
    J = getFunction(X, Y, 'Rosenbrock', a=1, b=1)  # object function
    lr = 0.5
    gamma = 0.9
    optimizer = AdaDelta(J, lr, gamma, X, Y)
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
