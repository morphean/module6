import time

from numpy import zeros, sqrt, put, exp, array, square

import CreditDefaultSwap as cds
import rng
from utils import printTime

debug = False


# @jitclass
class LMM1F(object):
    def __init__(self,
                 nSims=int,
                 initialSpotRates=array,
                 notional=float,
                 strike=0.05,
                 alpha=0.5,
                 sigma=0.15,
                 dT=0.5,
                 randSource=list):
        """

        :param strike:  caplet
        :param alpha: daycount factor
        :param sigma: fwd rates volatility - assuming 15%
        :param dT: 6M (year fraction of 0.5 is default)
        :param nSims: no of simulations to run
        """
        self.K = strike
        self.alpha = alpha
        self.sigma = sigma
        self.dT = dT
        self.N = len(initialSpotRates) - 1
        self.M = nSims
        self.initialSpotRates = initialSpotRates
        self.notional = notional
        self.simulations = []
        self.randSource = randSource

    # @jit(target='cpu')
    def simulateLMMviaMC(self):

        l = zeros(shape=(self.N + 1, self.N + 1), dtype=float)
        d = zeros(shape=(self.N + 2, self.N + 2), dtype=float)

        # init zero based tables

        # init spot rates
        for index, s in enumerate(self.initialSpotRates):
            l[index][0] = s

            for i in xrange(self.M):
                # setup brownian motion multipliers
                gbm_multipliers = self.initWeinerProcess(self.N + 1)

                # compute Fwd Rates And DiscountFactors Tableau
                l, d = self.computeTableaus(self.N, self.alpha, self.sigma, l, self.dT, gbm_multipliers, d)

                sim = cds.Simulation(l, d, self.notional, self.dT)
                self.simulations.append(sim)
                # print 'Completed simulation: ', i
        return self.simulations

    # @jit(target='cpu')
    def getSimulationData(self):
        # implied singleton here, so that dataset is not regenerated on each call to this method
        if len(self.simulations) == 0:
            self.simulations = self.simulateLMMviaMC()

        return array(self.simulations)

    # @jit(target='cpu')
    def initWeinerProcess(self, length=int):
        seq = zeros(self.N + 1)
        for dWi in xrange(length):
            dW = sqrt(self.dT) * rng.getBoxMullerSample(self.randSource)
            put(seq, dWi, dW)

        if debug:
            print 'Discount Factors', seq
        return seq

    # @jit(target='cpu')
    def computeTableaus(self, N=int, alpha=float, sigma=float, l=array, dT=float, dW=array, d=array):
        """
        Using Pessler Tableau to compute LIBOR rates and accompanying discount factors
        :param N:
        :param alpha:
        :param sigma:
        :param l:
        :param dT:
        :param dW:
        :param d:
        :return:
        """
        for n in range(0, N):

            for i in range(n + 1, N + 1):  # (int i = n + 1; i < N + 1; ++i)
                drift_sum = 0.0

                for k in range(i + 1, N + 1):  # (int k = i + 1; k < N + 1; ++k)
                    drift_sum += (alpha * sigma * l[k][n]) / (1 + alpha * l[k][n])

                newVal = l[i][n] * exp((-drift_sum * sigma - 0.5 * square(sigma)) * dT + sigma * dW[n + 1])
                put(l[i], n + 1, newVal)

                if debug:
                    print 'L: i = ', i, ', n = ', n + 1, ', = ', l[i][n + 1]

        for n in xrange(0, N + 1):  # (int n = 0; n < N + 1; ++n)
            for i in xrange(n + 1, N + 2):  # (int i = n + 1; i < N + 2; ++i)
                df_prod = 1.0
                for k in xrange(n, i):  # (int k = n; k < i; k++)
                    df_prod *= 1 / (1 + alpha * l[k][n])
                put(d[i], n, df_prod)
                if debug:
                    print 'D: i = ', i, ',n = ', n, ', D[', i, '][', n, '] = ', d[i][n]

        return l, d


def benchmark():
    n = 100
    initRates = array([0.01, 0.03, 0.04, 0.05, 0.07])
    irEx = LMM1F(nSims=n, initialSpotRates=initRates)
    start_time = time.clock()
    a = irEx.getSimulationData()
    printTime('GPU: generating simulation data', start_time)

    n = 10000
    irEx = LMM1F(nSims=n, initialSpotRates=initRates)
    start_time = time.clock()
    a = irEx.getSimulationData()
    printTime('GPU: generating simulation data', start_time)

    gpuEnabled = False

    n = 100
    irEx = LMM1F(nSims=n, initialSpotRates=initRates)
    start_time = time.clock()
    a = irEx.getSimulationData()
    printTime('CPU: generating simulation data', start_time)

    n = 10000
    irEx = LMM1F(nSims=n, initialSpotRates=initRates)
    start_time = time.clock()
    a = irEx.getSimulationData()
    printTime('CPU: generating simulation data', start_time)


if __name__ == '__main__':
    print 'LMM 1F Class. it is for importing.'
    benchmark()
