from opt.generalOpt import generalOpt
import math
from sympy import *
import numpy as np
from opt.functions import getFunction


"""
Shortcomings and advantages of vanilla gradient descent
-------------------------------------------------------
Advantage:
a) good convergent guarantee
Shortcoming:
a) easily fall into local minima
b) same learning rate for each parameter
c) hard to escape saddle point
"""


class vanillaGradientDescent(generalOpt):
    def __init__(self,
                 objectFunction,
                 lr,
                 gamma,
                 X,
                 Y
                 ):
        """
        :param lr: learning rate
        :param objectFunction: loss function (self-define)
        :param X
        :param Y
        """
        self.lr = lr
        self.objectFunction = objectFunction
        self.X = X
        self.Y = Y
        self.name = 'vanilla'

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
        θ_{t} = θ_{t-1} - lr * ▽J(θ_{t-1})
        :param param: parameters
        :return: updated parameter
        """
        return param - self.lr * self.evaluateGrid(param)

    def getName(self):
        return self.name


if __name__ == '__main__':
    X, Y = symbols('X Y', real=True)
    J = getFunction(X, Y, 'Rosenbrock', a=1, b=1)  # object function

    optimizer = vanillaGradientDescent(J, 0.5, X, Y)
    theta = np.array([-2, -2])
    lossBefore, lossUpdated = math.inf, 0  # used to define stop rule
    while abs(lossUpdated - lossBefore) > 1e-08:  # threshold
        """
        Note the .evalf() and .subs():
        1) .evalf() evaluates a given numerical expression upto a given floating point precision upto 100 digits. 
        The function also takes subs parameter a dictionary object of numerical values for symbols.
        2) .subs() replaces all occurrences of first parameter with second.
        
        Note that if you use .subs() in your expression to compute a value, but some of your component is the instance 
        in sympy, etc. sin(1), then the result will not be a value, but a expression with sin(1). In using .evalf() can
        avoid this unexpected conflict. .evalf() will transfer the symbols to corresponding number first and compute.
        """
        lossBefore = J.evalf(subs={X: theta[0], Y: theta[1]})
        theta = optimizer.step(theta)
        lossUpdated = J.evalf(subs={X: theta[0], Y: theta[1]})
        print(lossUpdated)


