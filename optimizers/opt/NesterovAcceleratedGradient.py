from opt.generalOpt import generalOpt
import math
from sympy import *
import numpy as np
from opt.functions import getFunction


"""
paper: A method for unconstrained convex minimization problem with the rate of convergence o(1/k2)

Due to the potential effect of large momentum, especially in the side of valley, Momentum
method could be unstable and easily diverge with sensitive learning rate.

Nesterov Accelerated Gradient (NAG) was proposed to solve this problem, which use the estimation 
of the gradient at the next step parameter (θ + γv_{t-1}) to counterbalance the previous (may be 
significant) momentum, namely "peeking ahead".

While Momentum first computes the current gradient and then takes a big jump in the direction of the
updated accumulated gradient, NAG first makes a big jump in the direction of the
previous accumulated gradient, measures the gradient and then makes a correction.

It is easily to comprehend that the gamma will effect learning rate range.

"""


class NAG(generalOpt):
    def __init__(self,
                 objectFunction,
                 lr,
                 gamma,
                 X,
                 Y
                 ):
        """
        :param lr: learning rate
        :param gamma: momentum ratio
        :param objectFunction: loss function (self-define)
        :param X
        :param Y
        """
        self.lr = lr
        self.gamma = gamma
        self.objectFunction = objectFunction
        self.X = X
        self.Y = Y
        self.name = 'NAG'
        self.v_previous = np.zeros((2,))

    def evaluateGrid(self, param):
        """
        compute gradient.
        :param param: current values.
        :return: partial derivative.
        """
        return np.array([diff(self.objectFunction, self.X).evalf(subs={self.X: param[0], self.Y: param[1]}),
                         diff(self.objectFunction, self.Y).evalf(subs={self.X: param[0], self.Y: param[1]})])

    def step(self, param):
        """
        v_t = γ * v_{t-1} + lr * ▽J(θ_{t-1} - γ * v_{t-1})
        θ = θ - v_t
        :param param: parameters
        :return: updated parameter
        """
        v_current = self.gamma * self.v_previous + self.lr * self.evaluateGrid(param - self.gamma * self.v_previous)
        self.v_previous = v_current

        return param - v_current

    def getName(self):
        return self.name


if __name__ == '__main__':
    X, Y = symbols('X Y', real=True)
    J = getFunction(X, Y, 'Rosenbrock', a=1, b=1)  # object function
    lr = 0.1
    gamma = 0.6
    optimizer = NAG(J, lr, gamma, X, Y)
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
