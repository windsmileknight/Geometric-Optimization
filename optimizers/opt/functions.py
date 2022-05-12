from sympy import *
import math

"""
provide various binary function
reference: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def Rosenbrock(X, Y, a=1, b=100):
    return (a - X) ** 2 + b * (Y - X ** 2) ** 2


def Rastrigin(X, Y, A=10):
    return A * 2 + X ** 2 - A * cos(2 * math.pi * X) + Y ** 2 - A * cos(2 * math.pi * Y)


def Ackley(X, Y):
    return -20 * math.e ** (-0.2 * sqrt(0.5 * (X ** 2 + Y ** 2))) - math.e ** (
            0.5 * (cos(2 * math.pi * X) + cos(2 * math.pi * Y))) + math.e + 20


def Sphere(X, Y):
    return X ** 2 + Y ** 2


def Beale(X, Y):
    return (1.5 - X + X * Y) ** 2 + (2.25 - X + X * Y ** 2) ** 2 + (2.625 - X + X * Y ** 3) ** 2


def Goldstein_Price(X, Y):
    return (1 + (X + Y + 1) ** 2 * (19 - 14 * X + 3 * X ** 2 - 14 * Y + 6 * X * Y + 3 * Y ** 2)) * (
                30 + (2 * X - 3 * Y) ** 2 * (18 - 32 * X + 12 * X ** 2 + 48 * Y - 36 * X * Y + 27 * Y ** 2))


def Booth(X, Y):
    return (X + 2 * Y - 7) ** 2 + (2 * X + Y - 5) ** 2


def BukinN6(X, Y):
    return 100 * sqrt(abs(Y - 0.01 * X ** 2)) + 0.01 * abs(X + 10)


def Matyas(X, Y):
    return 0.26 * (X ** 2 + Y ** 2) - 0.48 * X * Y


def LeviN13(X, Y):
    return sin(3 * math.pi * X) ** 2 + (X - 1) ** 2 * (1 + sin(3 * math.pi * Y) ** 2) + (Y - 1) ** 2 * (1 + sin(2 * math.pi * Y) ** 2)


def Himmelblau(X, Y):
    return (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2


def ThreeHumpCamel(X, Y):
    return 2 * X ** 2 - 1.05 * X ** 4 + (X ** 6) / 6 + X * Y + Y ** 2


def Easom(X, Y):
    return -cos(X) * cos(Y) * exp(-((X - math.pi) ** 2 + (Y - math.pi) ** 2))


def CrossInTray(X, Y):
    return -0.0001 * (abs(sin(X) * sin(Y) * exp(abs(100 - sqrt(X ** 2 + Y ** 2) / math.pi))) + 1) ** 0.1


def EggHolder(X, Y):
    return -(Y + 47) * sin(sqrt(abs(X / 2 + (Y + 47)))) - X * sin(sqrt(abs(X - (Y + 47))))


def HolderTable(X, Y):
    return -abs(sin(X) * cos(Y) * exp(abs(1 - sqrt(X ** 2 + Y ** 2) / math.pi)))


def McCormick(X, Y):
    return sin(X + Y) + (X - Y) ** 2 - 1.5 * X + 2.5 * Y + 1


def StyblinskiTang(X, Y):
    return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X + Y ** 4 - 16 * Y ** 2 + 5 * Y)


def getFunction(X, Y, name=None, **kwargs):
    """
    .. rubric:: Optimization test functions\n
    **Rosenbrock** (default: a=1, b=100)\n
    .. math:: Z = (a - X)^{2} + b(Y - X^{2})^{2}\n
    Recommend X and Y range in showing the 3D plot: X: (-2, 2); Y: (-1, 3.5)\n
    **Rastrigin** (default: A = 10)\n
    .. math:: Z = An + \sum_{i=1}^{n} [X_i^2 - A cos(2\pi X_i)]\n
    Recommend X and Y range in showing the 3D plot: X: (-4, 4); Y: (-4, 4)\n
    **Ackley**\n
    .. math:: Z = -20\exp[-0.2\sqrt{0.5(X^2 + Y^2)}] - \exp[0.5(\cos 2\pi X + \cos 2\pi Y)] + e + 20\n
    Recommend X and Y range in showing the 3D plot: X: (-4, 4); Y: (-4, 4)\n
    **Sphere**\n
    .. math:: Z = X^2 + Y^2\n
    Recommend X and Y range in showing the 3D plot: X: (-2, 2); Y: (-2, 2)\n
    **Beale**\n
    .. math:: Z = (1.5 - X + XY)^2 + (2.25 - X + XY^2)^2 + (2.625 - X + XY^3)^2\n
    Recommend X and Y range in showing the 3D plot: X: (-4, 4); Y: (-4, 4)\n
    **Goldstein_Price**\n
    .. math:: Z = [1 + (X + Y + 1)^2(19 - 14X + 3X^2 - 14Y + 6XY + 3Y^2)][30 + (2X - 3Y)^2(18 - 32X + 12X^2 + 48Y - 36XY + 27Y^2)]\n
    Recommend X and Y range in showing the 3D plot: X: (-2, 2); Y: (-3, 1)\n
    **Booth**\n
    .. math:: Z = (X + 2Y - 7)^2 + (2X + Y - 5)^2\n
    Recommend X and Y range in showing the 3D plot: X: (-10, 10); Y: (-10, 10)\n
    **BukinN6**\n
    .. math:: Z = 100 * \sqrt{|Y - 0.01X^2|} + 0.01|X + 10|\n
    Recommend X and Y range in showing the 3D plot: X: (-14, -6); Y: (-4, 6)\n
    **Matyas**\n
    .. math:: Z = 0.26(X^2 + Y^2) - 0.48XY\n
    Recommend X and Y range in showing the 3D plot: X: (-10, 10); Y: (-10, 10)\n
    **LeviN13**\n
    .. math:: Z = \sin^2(3\pi X) + (X - 1)^2(1 + sin^2(3\pi Y)) + (Y - 1)^2(1 + sin^2(2\pi Y))\n
    Recommend X and Y range in showing the 3D plot: X: (-4, 6); Y: (-4, 6)\n
    **Himmelblau**\n
    .. math:: Z = (X^2 + Y - 11)^2 + (X + Y^2 - 7)^2\n
    Recommend X and Y range in showing the 3D plot: X: (-4, 4); Y: (-4, 4)\n
    **ThreeHumpCamel**\n
    .. math:: Z = 2X^2 - 1.05X^4 + X^6/6 + XY + Y^2\n
    Recommend X and Y range in showing the 3D plot: X: (-4, 4); Y: (-4, 4)\n
    **Easom**\n
    .. math:: Z = -\cos{X}\cos{Y}\exp(-((X - \pi)^2 + (Y - \pi)^2))\n
    Recommend X and Y range in showing the 3D plot: X: (0, 6); Y: (-1, 7)\n
    **CrossInTray**\n
    .. math:: Z = -0.0001[|sinX sinY exp(|100 - (\sqrt{X^2 + Y^2})/(\pi)|)| + 1]^{0.1}\n
    Recommend X and Y range in showing the 3D plot: X: (-10, 10); Y: (-10, 10)\n
    **Eggholder**\n
    .. math:: Z = -(y + 47)\sin(\sqrt{|X/2 + (Y + 47)|}) - X\sin(\sqrt{|X - (Y + 47)|})\n
    Recommend X and Y range in showing the 3D plot: X: (-1000, 1000); Y: (-1000, 1000)\n
    **HolderTable**\n
    .. math:: Z = -|sinXcosYexp(|(1 - sqrt(X² + Y²))/\pi|)|\n
    Recommend X and Y range in showing the 3D plot: X: (-10, 10); Y: (-10, 10)\n
    **McCormick**\n
    .. math:: Z = \sin(X + Y) + (X - Y)^2 - 1.5X + 2.5Y + 1\n
    Recommend X and Y range in showing the 3D plot: X: (-3, 4); Y: (-3, 4)\n
    **StyblinskiTang**\n
    .. math:: Z = 0.5(\sum_{i=1}^{n} X_i^4 - 16X_i^2 + 5X) \n
    Recommend X and Y range in showing the 3D plot: X: (-5, 5); Y: (-5, 5)\n
    :param X
    :param Y
    :param name: function name
    :param kwargs: extensive configurations
    :return: function (sympy instance)
    """
    if name == 'Rosenbrock':
        return Rosenbrock(X, Y, **kwargs)
    elif name == 'Rastrigin':
        return Rastrigin(X, Y, **kwargs)
    elif name == 'Ackley':
        return Ackley(X, Y)
    elif name == 'Sphere':
        return Sphere(X, Y)
    elif name == 'Beale':
        return Beale(X, Y)
    elif name == 'Goldstein-Price':
        return Goldstein_Price(X, Y)
    elif name == 'Booth':
        return Booth(X, Y)
    elif name == 'BukinN6':
        return BukinN6(X, Y)
    elif name == 'Matyas':
        return Matyas(X, Y)
    elif name == 'LeviN13':
        return LeviN13(X, Y)
    elif name == 'Himmelblau':
        return Himmelblau(X, Y)
    elif name == 'ThreeHumpCamel':
        return ThreeHumpCamel(X, Y)
    elif name == 'Easom':
        return Easom(X, Y)
    elif name == 'CrossInTray':
        return CrossInTray(X, Y)
    elif name == 'EggHolder':
        return EggHolder(X, Y)
    elif name == 'HolderTable':
        return HolderTable(X, Y)
    elif name == 'McCormick':
        return McCormick(X, Y)
    elif name == 'StyblinskiTang':
        return StyblinskiTang(X, Y)
    else:
        raise Exception("Function does not support!")
