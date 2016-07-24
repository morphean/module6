# vendor libs
from accelerate import cuda
from os import path

from matplotlib import pyplot as plt
from scipy.stats import t

import Copula
import rng

gpuEnabled = cuda.cuda_compatible()

print 'GPUGP libraries detected, enabling GPU for rng' if gpuEnabled else 'not using GPU'


class ReferenceName(object):
    def __init__(self, name=str, intensity=float, mean=float, var=float):
        self.name = name
        self.intensity = intensity
        self.mean = mean
        self.var = var


def generateRandomSampleMatrix(sims=int, basketSize=1):
    """
    generate a matrix of two independent standard normal variables,
    $Z_1, Z_2$

    nb: it will source these via GPU is CUDA drivers are installed and configured correctly.
    :param sims:
    :return:
    """
    shape = (basketSize, sims)
    if gpuEnabled:
        return rng.getPseudoRandomNumbers_Standard_cuda(shape)

    return rng.getPseudoRandomNumbers_Standard(shape)


def generateRandomTSampleMatrix(sims=int, basketSize=1, dof=2.74335149908):
    """
    generate a matrix of n independent variables from t distribution,
    $Z_1, Z_2$

    nb: not available on GPU
    :param sims:
    :return:
    """

    return t.rvs(df=dof, size=(sims, basketSize))


def plotCreditSpreads(data=list):
    # plot spread curves
    for item in data:
        plt.plot(item['Date'], item['CDS_spreads'], lw=0.5)
    plt.grid = True
    plt.show()


# let setup some data
srcDataPath = 'file://localhost/' + path.realpath('../data/db/sovereign_pd_data.xls')

# define basket (sheetnames in xl file)
basket = ['uk', 'aus', 'de', 'fr', 'us']

RHO = 0.1
reference = ReferenceName('UK', 0.01, 100, 10000)
reference2 = ReferenceName('US', 0.05, 20, 400)
reference3 = ReferenceName('DE', 0.05, 20, 400)
reference4 = ReferenceName('FR', 0.05, 20, 400)
reference5 = ReferenceName('AU', 0.05, 20, 400)

gpuEnabled = False
numberOfSimulations = 1000
z = generateRandomSampleMatrix(numberOfSimulations)
tSim = generateRandomTSampleMatrix(numberOfSimulations)

# simulating Copula functions
# g = Copula.simulateCopula(numberOfSimulations, rho=RHO, type='g', intensity=(reference.intensity, reference2.intensity))
t = Copula.simulateCopula(numberOfSimulations, rho=RHO, type='t', lamda=(reference.intensity, reference2.intensity))
t1 = Copula.simulateCopula(numberOfSimulations, rho=RHO * 4, type='t',
                           lamda=(reference.intensity, reference2.intensity))
t2 = Copula.simulateCopula(numberOfSimulations, rho=RHO * 6, type='t',
                           lamda=(reference.intensity, reference2.intensity))
t3 = Copula.simulateCopula(numberOfSimulations, rho=RHO * 8, type='t',
                           lamda=(reference.intensity, reference2.intensity))
t4 = Copula.simulateCopula(numberOfSimulations, rho=RHO * 9, type='t',
                           lamda=(reference.intensity, reference2.intensity))

# plot copula simulations, showing change as rho increases
# plt.plot(g['u1'], g['u2'], 's')
plt.plot(t['u1'], t['u2'], 's')
plt.plot(t1['u1'], t1['u2'], 's')
plt.plot(t2['u1'], t2['u2'], 's')
plt.plot(t3['u1'], t3['u2'], 's')
plt.plot(t4['u1'], t4['u2'], 's')
plt.show()

# plot credit spreads
plt.plot()

# import bootstrapped pd data (dbresearch)

#
