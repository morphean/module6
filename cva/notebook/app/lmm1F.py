import time
from itertools import cycle

from matplotlib import pyplot
from numba import autojit
from numpy import zeros, sqrt, put, exp, linspace, array, append, square, mean, percentile, insert
from scipy.optimize import fsolve

import CreditDefaultSwap as cds
import rng
from utils import printTime

gpuEnabled = False


# pushing the size of rng generated above 100000 causes GPU to run out of space
# possible optization is to load it into a 3d vector shape instead of flat structure.
@autojit
def initRandSource():
    randN = 100000
    randSourceGPU = rng.getPseudoRandomNumbers_Uniform_cuda(randN) if gpuEnabled else []
    randSourceCPU = rng.getPseudoRandomNumbers_Uniform(randN)
    return randSourceGPU if gpuEnabled else randSourceCPU


randSource = initRandSource()

debug = False

# 5Y tenor
noOfYears = 5.
# 6M payments
paymentFrequency = 0.5
yearFraction = paymentFrequency / noOfYears
noOfPayments = noOfYears / paymentFrequency

# no of timesteps
timesteps = linspace(0, 1, noOfPayments + 1)


# @jitclass
class LMM1F(object):
    def __init__(self, nSims=int, initialSpotRates=array, notional=float, strike=0.05, alpha=0.5, sigma=0.15, dT=0.5, ):
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
            dW = sqrt(self.dT) * rng.getBoxMullerSample(randSource)
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


n = 100
# initRates = array([0.01, 0.03, 0.04, 0.05, 0.07])
initRates = array([0.01, 0.00037, 0.00232, 0.00306, 0.00437])
irEx = LMM1F(nSims=n, initialSpotRates=initRates, notional=1000000)
start_time = time.clock()
a = irEx.getSimulationData()
printTime('GPU: generating simulation data', start_time)

# df = [0.9975, 0.9900, 0.9779, 0.9607]
df = [0.9999, 0.9988, 0.9776, 0.9662]
seed = 0.01
payments = 4
notional = 1000000
# fixed hazard rates
lamda = 0.03


def calibrateCDS(lamda=float, seed=0.01):
    # calibration method
    c = cds.CreditDefaultSwap(N=notional, timesteps=payments, discountFactors=df, lamda=lamda, seed=seed)
    return c.markToMarket


def getExpectedExposure(simData=array):
    expectedExposure = mean(array([sim.mtm for sim in simData]), axis=0)
    return expectedExposure


def getPercentile(simData=array, p=float):
    ptile = percentile(array([sim.mtm for sim in simData]), q=p, axis=0)
    return ptile


def plotExpectedExposure(simData=array):
    # plot simulationsz
    cycol = cycle('cmy').next

    for simulation in a:
        x, y = timesteps, append(simulation.mtm, 0)
        # print y[0]
        pyplot.plot(x, y, 's', c=cycol(), lw=0.5, alpha=0.6)

    # get expected Exposure
    expectedExposure = getExpectedExposure(simData=simData)
    ninetySevenP5 = getPercentile(simData=simData, p=97.5)
    twoP5 = getPercentile(simData=simData, p=2.5)

    # rate from B91 on ukois16_mdaily.xls
    initRates_BOE_6m_plot = insert(initRates_BOE_6m, 0, 0.463126310164261)
    # add to plot
    # pyplot.plot(x, twoP5,c='#000000', lw=2, alpha=0.8)
    pyplot.plot(x, append(ninetySevenP5, 0.0), c='#00CC00', lw=2, alpha=0.8)
    pyplot.plot(x, append(twoP5, 0.0), c='#0000CC', lw=2, alpha=0.8)
    pyplot.plot(x, append(expectedExposure, 0.0), c='#FF0000', lw=2, alpha=0.8)
    # pyplot.plot(x, append(initRates_BOE_6m,0.0), lw=2, alpha=0.8)
    # pyplot.plot(x, initRates, lw=2, alpha=0.8)
    pyplot.xlabel('$\Delta t$')
    pyplot.xticks(timesteps)
    pyplot.title('Simulated $L_{6M}$')
    pyplot.ylabel('Payment')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()

#
# p = optimize.minimize(calibrateCDS, [0.03,0.02,0.01, 0.0128], options={'gtol':1e-6, 'disp': True})

calibratedLambda = fsolve(calibrateCDS, lamda)

print calibratedLambda
c = cds.CreditDefaultSwap(N=notional, timesteps=payments, discountFactors=df, lamda=lamda, seed=seed)

print c.markToMarket

n = 10
# from
# initRates = [0.004625384831,0.006812859244,0.009606329010,0.012040577151,0.013898638373,0.015251633803,0.016212775988,0.016899521932,0.017420140777,0.017841154790]
initRates_BOE_SPT = [0.461370334, 0.452898877, 0.443905273, 0.435584918, 0.428713527, 0.423546874, 0.420158416,
                     0.418516458, 0.41849405, 0.419889508, 0.42246272, 0.425980925, 0.430265513, 0.43519681,
                     0.440673564, 0.44660498, 0.452907646, 0.45950348, 0.466318901, 0.473289989, 0.480366388,
                     0.487508789, 0.494686524, 0.501875772, 0.509058198, 0.516220029, 0.523351338, 0.530445369,
                     0.537497985, 0.544507238, 0.551473011, 0.558396735, 0.565281153, 0.572130127, 0.578948479,
                     0.585741857, 0.592516573, 0.599278976, 0.606034922, 0.612789787, 0.619548528, 0.626315731,
                     0.633095655, 0.63989227, 0.646709286, 0.653550185, 0.660418244, 0.667316554, 0.67424803,
                     0.681215261, 0.688220353, 0.695264957, 0.702350311, 0.709477279, 0.716646384, 0.723857839,
                     0.731111575, 0.738407263, 0.745744337, 0.753122018]

initRates_BOE_6m = [0.423546874, 0.425980925, 0.45950348, 0.501875772, 0.551473011, 0.585741857, 0.626315731,
                    0.667316554, 0.709477279, 0.753122018]

initRates_scaled = [x * 1 / 100. for x in initRates_BOE_6m]
BOE_6m_discountFactors = cds.genDiscountFactors(timesteps, initRates_scaled)
BOE_6m_fwdCurve = cds.genFwdCurve(timesteps, initRates_scaled)
# payments = 10
# d = cds.CreditDefaultSwap(N=notional, timesteps=payments, discountFactors=BOE_6m_discountFactors, lamda=lamda, seed=seed, recoveryRate=0.4, Dt=0.5)
# c = cds.CreditDefaultSwap(N=notional, timesteps=payments, discountFactors=df, lamda=lamda, seed=seed, recoveryRate=0.4, Dt=0.25)
# simulate L6M
irEx = LMM1F(nSims=n, initialSpotRates=initRates_scaled, dT=yearFraction, notional=notional)
start_time = time.clock()
a = irEx.getSimulationData()

# exposure = c.getExpectedExposureA(a)
exposure = array([0., 803.47, 729.78, 486.66])
cva = c.calcCVA(exposure)
printTime('CPU: generating simulation data', start_time)

# bva = cva - c.calcCVA(c.getExpectedExposureB(a))

print '====='
print 'cva: ', cva
# print 'bva: ', bva
