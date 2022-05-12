import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sympy import *
import math
from opt import OPT
import matplotlib.animation as animation
from opt.functions import getFunction

# change output backend
mpl.use('Qt5Agg')


class Painter:
    def __init__(self, J, optimizer, **kwargs):
        """
        define axis and figure.
        :param nRow: number of subplot in a row
        :param nColumn: number of subplot in a column
        :param J: function (sympy instance)
        :param optimizer: optimization algorithm
        """
        self.fig = plt.figure(**kwargs)
        self.ax = self.fig.add_subplot(projection='3d')
        self.J = J
        self.optimizer = optimizer
        self.bottom = None

    def draw3DPlot(self,
                   XMin, XMax,
                   YMin, YMax,
                   interval=100
                   ):
        # data for 3D plotting
        xData = np.linspace(XMin, XMax, interval)
        yData = np.linspace(YMin, YMax, interval)
        xData, yData = np.meshgrid(xData, yData)  # transform to grid data (2D) (plot_surface requirement)
        zData = np.zeros_like(xData)
        for i in range(xData.shape[0]):
            for j in range(xData.shape[1]):
                zData[i, j] = float(self.J.evalf(subs={self.optimizer.X: xData[i, j], self.optimizer.Y: yData[i, j]}))
        self.bottom = np.min(zData)
        self.ax.set_zlim3d(self.bottom, np.max(zData))

        # draw figure
        self.ax.set_xlabel('$X$', fontsize=14)
        self.ax.set_ylabel('$Y$', fontsize=14)
        self.ax.set_zlabel('$Z$', fontsize=14)
        self.ax.plot_surface(xData, yData, zData, cmap=plt.cm.CMRmap, alpha=0.5)  # 3D figure
        self.ax.contour(xData, yData, zData, 50, zdir='z', offset=np.min(zData),
                        cmap=plt.cm.CMRmap)  # contour figure of 3D figure

    def drawAnimationPlot(self, updateFunction, trace, lossTrace, steps, saveGifPath):
        """
        draw gradient descent animation figure
        :param saveGifPath: .gif save path
        :param updateFunction: function used to renew animation
        :param trace: updating point trace
        :param lossTrace: updating loss trace
        :param steps: updating steps
        :return: ax: axis
        """
        gradientTrace, = self.ax.plot(trace[0, :1], trace[1, :1], lossTrace[0], color='blue')
        gradientTrace2D, = self.ax.plot(trace[0, :1], trace[1, :1], self.bottom, color='blue')
        ani = animation.FuncAnimation(self.fig, updateFunction, steps, fargs=(gradientTrace, gradientTrace2D, self.bottom, trace, lossTrace),
                                      interval=int(1000 / 60), blit=False)
        ani.save(saveGifPath, writer='pillow', fps=60, dpi=200)
        print('animation saved in ' + saveGifPath)
        # plt.show()

    def drawTextLabel(self, initPoint, initLoss):
        """
        draw start point label in 3D figure
        :param initLoss: loss in initial point
        :param initPoint: initial point
        """
        self.ax.scatter(initPoint[0], initPoint[1], self.bottom, color='blue')
        self.ax.scatter(initPoint[0], initPoint[1], initLoss, color='blue')
        self.ax.plot([initPoint[0]] * 2, [initPoint[1]] * 2, [self.bottom, initLoss], color='blue', linestyle='dashed')
        self.ax.text(initPoint[0] - 1, initPoint[1] - 1.8, self.bottom, '$start point: [' + str(initPoint[0]) + ', ' + str(initPoint[1]) + ']$')

    def setXLim3D(self, XMin, XMax):
        self.ax.set_xlim3d(XMin, XMax)

    def setYLim3D(self, YMin, YMax):
        self.ax.set_ylim3d(YMin, YMax)

    def setZLim3D(self, ZMin, ZMax):
        self.ax.set_zlim3d(ZMin, ZMax)

    def setXLabelName(self, xName):
        self.ax.set_xlabel(xName)

    def setYLabelName(self, yName):
        self.ax.set_ylabel(yName)

    def setZLabelName(self, zName):
        self.ax.set_zlabel(zName)


if __name__ == '__main__':
    X, Y = symbols('X Y', real=True)
    J = getFunction(X, Y, 'StyblinskiTang')
    opt = 'vanilla'
    lr = 0.01
    optimizer = OPT[opt](J, lr, None, X, Y)
    painter = Painter(J, optimizer)
    XMin, XMax, YMin, YMax = -5, 5, -5, 5
    painter.draw3DPlot(XMin, XMax, YMin, YMax)
    plt.show()
