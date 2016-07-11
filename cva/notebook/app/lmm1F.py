import time

from numpy import ndarray, zeros, sqrt, power, put, exp
from scipy.optimize import fsolve

import CreditDefaultSwap as cds
import rng
from utils import printTime

gpuEnabled = True
# pushing the size of rng generated above 100000 causes GPU to run out of space
# possible optization is to load it into a 3d vector shape instead of flat structure.
randN = 100000

randSourceGPU = rng.getPseudoRandomNumbers_Uniform_cuda(randN)
randSourceCPU = rng.getPseudoRandomNumbers_Uniform(randN)

randSource = randSourceGPU if gpuEnabled else randSourceCPU

debug = False


# print 'Checking for compatible GPU ...'
# if getGpuCount() < 1:
#     print 'no compatible GPU installed, will revert to using numpy libraries for RNG'
# else:
#     print 'nVidia GPU(s) found: ', getGpuCount()
#     print 'Enabling GPU for RNG'
#     gpuEnabled = True

class lSimulation:
    def __init__(self, liborTable=ndarray, dfTable=ndarray):
        """

        :type dfTable: ndarray
        :type liborTable: ndarray
        """
        self.__liborTable = liborTable
        self.__dfTable = dfTable

    @property
    def liborTable(self):
        return self.__liborTable

    @property
    def dfTable(self):
        return self.__dfTable

    def liborTableAtT(self, t=int):
        return self.liborTable[t]

    def dfTableAtT(self, t=int):
        return self.dfTable[t]


class LMM1F:
    def __init__(self, strike=0.05, alpha=0.5, sigma=0.15, dT=0.5, nFwdRates=4, nSims=10, initialSpotRates=ndarray):
        """

        :param strike:  caplet
        :param alpha: daycount factor
        :param sigma: fwd rates volatility
        :param dT:
        :param nFwdRates: no of fwd rates supplied
        :param nSims: no of simulations to run
        """
        self.K = strike
        self.alpha = alpha
        self.sigma = sigma
        self.dT = dT
        self.N = nFwdRates
        self.M = nSims
        self.initialSpotRates = initialSpotRates

    def generateLiborSim(self):

        l = zeros(shape=(self.N + 1, self.N + 1), dtype=float)
        d = zeros(shape=(self.N + 2, self.N + 2), dtype=float)

        # init zero based tables

        # init spot rates
        for index, s in enumerate(self.initialSpotRates):
            l[index][0] = s

        simulations = []

        for i in xrange(self.M):
            # setup brownian motion multipliers
            gbm_multipliers = self.initDiscountFactors(self.N + 1)

            # computeFwdRatesTableau
            self.computeFwdRatesTableau(self.N, self.alpha, self.sigma, l, self.dT, gbm_multipliers)

            # computeDiscountRatesTableau
            self.computeDiscountRatesTableau(self.N, l, d, self.alpha)
            storeValue = lSimulation(l, d)
            simulations.append(storeValue)
        return simulations

    def getSimulationData(self):
        libor = self.generateLiborSim()
        return libor

    def initDiscountFactors(self, length=int):
        seq = zeros(self.N + 1)
        for dWi in xrange(1, length):
            dW = sqrt(self.dT) * rng.getBoxMullerSample()
            put(seq, dWi, dW)

        if debug:
            print 'Discount Factors', seq
        return seq

    def computeFwdRatesTableau(self, N=int, alpha=float, sigma=float, l=ndarray, dT=float, dW=ndarray):
        for n in range(0, N):

            for i in range(n + 1, N + 1):  # (int i = n + 1; i < N + 1; ++i)
                drift_sum = 0.0

                for k in range(i + 1, N + 1):  # (int k = i + 1; k < N + 1; ++k)
                    drift_sum += (alpha * sigma * l[k][n]) / (1 + alpha * l[k][n])

                newVal = l[i][n] * exp((-drift_sum * sigma - 0.5 * power(sigma, 2)) * dT + sigma * dW[n + 1])
                put(l[i], n + 1, newVal)
                # l[i][n + 1] = l[i][n] * np.math.exp((-drift_sum * sigma - 0.5 * sigma * sigma) * dT + sigma * dW[n + 1])

                if debug:
                    print 'L: i = ', i, ', n+1 = ', n + 1, ', = ', l[i][n + 1]
        return l

    def computeDiscountRatesTableau(self, N=int, L=ndarray, D=ndarray, alpha=float):
        for n in xrange(0, N + 1):  # (int n = 0; n < N + 1; ++n)
            for i in xrange(n + 1, N + 2):  # (int i = n + 1; i < N + 2; ++i)
                df_prod = 1.0
                for k in xrange(n, i):  # (int k = n; k < i; k++)
                    df_prod *= 1 / (1 + alpha * L[k][n])
                put(D[i], n, df_prod)
                if debug:
                    print 'D: i = ', i, ',n = ', n, ', D[i][n] = ', D[i][n]


n = 100
irEx = LMM1F(nSims=n, initialSpotRates=[0.01, 0.03, 0.04, 0.05, 0.07])
start_time = time.clock()
a = irEx.getSimulationData()
printTime('GPU: generating simulartion data took: ', start_time)

n = 1000
irEx = LMM1F(nSims=n)
start_time = time.clock()
a = irEx.getSimulationData()
printTime('GPU: generating simulartion data took: ', start_time)

gpuEnabled = False

n = 100
irEx = LMM1F(nSims=n)
start_time = time.clock()
a = irEx.getSimulationData()
printTime('CPU: generating simulartion data took: ', start_time)

n = 1000
irEx = LMM1F(nSims=n)
start_time = time.clock()
a = irEx.getSimulationData()
printTime('CPU: generating simulartion data ', start_time)

df = [0, 0.9975, 0.9900, 0.9777, 0.9607]
seed = 0.01
payments = 4
notional = 1000000


def calibrateCDS(lamda=float, seed=0.01):
    c = cds.CreditDefaultSwap(N=notional, timesteps=payments, discountFactors=df, lamda=lamda, seed=seed)
    return c.markToMarket


#
# p = optimize.minimize(calibrateCDS, [0.03,0.02,0.01, 0.0128], options={'gtol':1e-6, 'disp': True})

calibratedLambda = fsolve(calibrateCDS, 0.03)
print calibratedLambda
c = cds.CreditDefaultSwap(N=notional, timesteps=payments, discountFactors=df, lamda=calibratedLambda, seed=seed)

print c.markToMarket