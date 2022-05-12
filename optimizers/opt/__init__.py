from .vanillaGradientDescent import vanillaGradientDescent
from .momentum import momentum
from .NesterovAcceleratedGradient import NAG
from .AdaGrad import AdaGrad
from .AdaDelta import AdaDelta
from .RMSprop import RMSprop


OPT = {
    'vanilla': vanillaGradientDescent,
    'momentum': momentum,
    'NAG': NAG,
    'AdaGrad': AdaGrad,
    'AdaDelta': AdaDelta,
    'RMSprop': RMSprop
}
