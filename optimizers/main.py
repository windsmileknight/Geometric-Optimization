import math
from sympy import *
from painter import Painter
from opt import OPT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from opt.functions import getFunction

# change output backend
mpl.use('Qt5Agg')


def lossEvaluate(J, currentPoint):
    """
    evaluate loss in current point
    :param J: objective function
    :param currentPoint: current parameter
    :return:
    """
    return float(J.evalf(subs={X: currentPoint[0], Y: currentPoint[1]}))


def optimize(J, optimizer, initPoint, threshold=1e-8):
    """
    optimizing and return updating trace
    :param J: objective function
    :param optimizer: optimizer
    :param initPoint: initial point
    :param threshold: stopping standard
    :return: trace ---> point updating trace; lossTrace ---> loss updating trace; step: updating total step
    """
    trace = [initPoint]  # parameter updating trace
    lossTrace = [lossEvaluate(J, initPoint)]  # loss trace
    step = 1  # updating step
    previousLoss, updatedLoss = lossTrace[0], math.inf
    param = initPoint
    while abs(updatedLoss - previousLoss) > threshold:
        previousLoss = lossTrace[-1]
        param = optimizer.step(param)  # update parameter
        updatedLoss = lossEvaluate(J, param)
        trace.append(param)  # update trace
        lossTrace.append(updatedLoss)  # update loss trace
        if step % 10 == 0:
            print('step: {}; loss: {}'.format(step, updatedLoss))
        step += 1
        if updatedLoss == math.inf:  # check whether diverge
            raise Exception('Diverge occupy! Try smallest learning rate!')
    print('total update step: {}'.format(step))

    return np.array(trace).T, np.array(lossTrace), step


def animationUpdate(num, gradientTrace, gradientTrace2D, bottom, pointTrace, lossTrace):
    """
    update gradient trace data. (this function called by animation.FuncAnimation in Painter)
    :param bottom: the bottom of z of gradientTrace2D
    :param gradientTrace2D: gradient trace in 2D
    :param lossTrace: updating trace of objective function loss
    :param pointTrace: updating trace of updating point
    :param num: frame
    :param gradientTrace: gradient trace plot
    """
    gradientTrace.set_data(pointTrace[:, : num])
    gradientTrace.set_3d_properties(lossTrace[: num])
    gradientTrace2D.set_data(pointTrace[:, : num])
    gradientTrace2D.set_3d_properties([bottom] * num)


if __name__ == '__main__':
    X, Y = symbols('X Y', real=True)
    functionName = 'Rosenbrock'
    J = getFunction(X, Y, functionName, a=1, b=1)  # object function
    opt = 'AdaDelta'
    lr = 0.5  # learning rate
    gamma = 0.95  # momentum ratio
    threshold = 1e-08  # optimization threshold
    optimizer = OPT[opt](J, lr, gamma, X, Y)  # optimizer
    painter = Painter(J, optimizer)  # figure painter
    initPoint = np.array([-2, 3])
    # this link, https://ggb123.cn/3d, is recommended to tune you 3D parameter
    XMin, XMax, YMin, YMax = -2, 2, -1, 3.5

    # obtain updating trace
    trace, lossTrace, step = optimize(J, optimizer, initPoint, threshold=threshold)  # get updating process data

    # gif save path
    saveGifPath = r'fig/' + functionName + '_' + opt + '_gamma_' + str(gamma) + '_lr_' + str(lr) + '_step_' + str(step) + '.gif'

    # draw figure
    painter.draw3DPlot(XMin, XMax, YMin, YMax, interval=100)  # draw 3D figure first

    # draw start point
    painter.drawTextLabel(initPoint, lossEvaluate(J, initPoint))

    # draw animation figure
    painter.drawAnimationPlot(animationUpdate, trace, lossTrace, step, saveGifPath)

    # plt.show()
