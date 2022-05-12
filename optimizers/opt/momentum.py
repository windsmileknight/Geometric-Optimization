from opt.generalOpt import generalOpt
import math
from sympy import *
import numpy as np
from opt.functions import getFunction


"""
paper: On the momentum term in gradient descent learning algorithms
Momentum was propose to handle the oscillation when long and narrow valley occupying.

Straightforward comprehension: 
The momentum term helps average out the oscillation along the short 
axis (oscillating direction) while at the same time adds up contributions 
along the long axis (steepest direction).

Advantages:
a) enlarge the bound of learning
b) accelerate gradient speed under valley situation
Shortcoming:
a) due to affect by large momentum (especially in valley side), the optimizer may fall into local optima and easily diverge

Choice of gamma (recommended γ = 0.9): too large will cause oscillation near optima, too small will not accelerate algorithm.
The loss curve of momentum is not stable as vanilla as the descent inertia from previous gradient will effect the following, 
which increase the opportunity of falling in local optima. 
"""


class momentum(generalOpt):
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
        self.name = 'momentum'
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
        v_t = γ * v_{t-1} + lr * ▽J(θ_{t-1})
        θ_t = θ_{t-1} - v_t
        :param param: parameters
        :return: updated parameter
        """
        v_current = self.gamma * self.v_previous + self.lr * self.evaluateGrid(param)
        self.v_previous = v_current

        return param - v_current

    def getName(self):
        return self.name


if __name__ == '__main__':
    X, Y = symbols('X Y', real=True)
    J = getFunction(X, Y, 'Rosenbrock', a=1, b=1)  # object function
    lr = 0.05
    gamma = 0.9
    optimizer = momentum(J, lr, gamma, X, Y)
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
