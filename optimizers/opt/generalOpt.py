import abc


class generalOpt(metaclass=abc.ABCMeta):
    """
    define abstract class for opts.
    """
    @abc.abstractmethod
    def evaluateGrid(self, param):
        """
        compute gradient.
        """
        pass

    @abc.abstractmethod
    def step(self, param):
        """
        update.
        :param param: parameters (X and Y)
        :return: updated parameters
        """
        pass

    @abc.abstractmethod
    def getName(self):
        pass
