import math
import time

import numpy as np

from notebook.app import utils

"""
please note lambda has a special meaning in python
so when it is used here to represent the greek symbol
i have written is as 'lamda' not to be confused with the
built in method 'lambda'
"""

""" spot rate data from:
https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/Historic-Yield-Data-Visualization.aspx"""

# constants
PROBABILITY_OF_DEFAULT = 'probabilityOfDefault'
PROBABILITY_OF_SURVIVAL = 'probabilityOfSurvival'
EXPOSURE = 'exposure'
DISCOUNT_FACTOR = 'discountFactor'
RECOVERY_RATE = 'lamda'

# setup vars
notional = 1000000
tenor = 12  # aka 5Y
dt = 0.25  # payment freq 6M
fixedRate = 0.0039
recoveryRate = 0.4  # rr

# timesteps
# timeSteps = [1/60.,3/60.,6/60.,12/60.,36/60.,1.]
# add spot rates
# spotRatesN = [0,0.0024,0.0027,0.0038,0.0048,0.0064,0.0076,0.0108]
# timeSteps = [0, 0.25, 0.5, 0.75, 1]
# spotRatesN = [0, 0.0004, 0.0025, 0.0032, 0.0040]
"""
OIS spot rates taken from BoE with today being 31/05/16
"""
timeSteps = [0.08, 0.17, 0.25, 0.33, 0.42, 0.50, 0.58, 0.67, 0.75, 0.83, 0.92, 1.00, 1.08, 1.17, 1.25, 1.33, 1.42, 1.50,
             1.58, 1.67, 1.75, 1.83, 1.92, 2.00, 2.08, 2.17, 2.25, 2.33, 2.42, 2.50, 2.58, 2.67, 2.75, 2.83, 2.92, 3.00,
             3.08, 3.17, 3.25, 3.33, 3.42, 3.50, 3.58, 3.67, 3.75, 3.83, 3.92, 4.00, 4.08, 4.17, 4.25, 4.33, 4.42, 4.50,
             4.58, 4.67, 4.75, 4.83, 4.92, 5.00]
spotRatesN = [0.46, 0.45, 0.44, 0.44, 0.43, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.43, 0.43, 0.44, 0.44, 0.45, 0.45,
              0.46, 0.47, 0.47, 0.48, 0.49, 0.49, 0.50, 0.51, 0.52, 0.52, 0.53, 0.54, 0.54, 0.55, 0.56, 0.57, 0.57,
              0.58, 0.59, 0.59, 0.60, 0.61, 0.61, 0.62, 0.63, 0.63, 0.64, 0.65, 0.65, 0.66, 0.67, 0.67, 0.68, 0.69,
              0.70, 0.70, 0.71, 0.72, 0.72, 0.73, 0.74, 0.75, 0.75]



def genFwdCurve(t=list, s=list):
    curve = []
    for i, (ti, si) in enumerate(zip(t, s)):
        if (i > 0):
            sim1 = s[i - 1]
            tim1 = t[i - 1]
            Li = ((si * ti) - (sim1 * tim1)) / (ti - tim1)
            curve.append(Li)
    return curve


def genDiscountFactors(t=list, s=list):
    discount = []
    for i, (ti, si) in enumerate(zip(t, s)):
        if i > 0:
            a = si * ti * -1
            DFi = math.exp(a)
            discount.append(DFi)
    return discount


def genDefaultProbabilities(lamda=float, t=list):
    defaultProb = []
    for i, ti in enumerate(t):
        if (i > 0):
            a = lamda * t[i - 1]
            b = lamda * ti
            PDi = math.exp(-a) - math.exp(-b)
            defaultProb.append(PDi)
    return defaultProb


# refactor PD to use this functon
def survivalProbability(lamda=float, time=float):
    """
    to calculate survival probability
    -1 * lamda * t

    :param lamda:
    :param time:
    :return: survival probability
    :rtype: float
    """
    multiplier = -1 * lamda * time
    return math.exp(multiplier)


def generateForwardRateCurve(spotRateData=list):
    curve = []
    for index, entry in enumerate(spotRateData):
        Ti = entry[0]
        Si = entry[1]
        SiTi = Si * Ti
        if index > 0:
            prevEntry = spotRateData[index - 1]
            tm1 = prevEntry[0]
            sm1 = prevEntry[1]
            SiM1 = sm1 * tm1
            TiM1 = Ti - tm1
            if TiM1 > 0:
                curve.append((SiTi - (SiM1)) / TiM1)
    return curve


def calculateSwapPayment(notional=int, dt=float, discountFactor=float, L=float, K=float):
    result = notional * dt * discountFactor * (L - K)
    return result


# start_time = time.clock()
# ABC = math.exp(123)
# print(time.clock() - start_time)


def timeFunctions():
    start_time = time.clock()
    FWD2 = genFwdCurve(timeSteps, spotRatesN)
    utils.printTime('Generating fwd curve', start_time)
    print FWD2

    start_time = time.clock()
    DP = genDefaultProbabilities(0.03, timeSteps)
    utils.printTime('Generating default probability', start_time)
    print DP

    start_time = time.clock()
    DF = genDiscountFactors(timeSteps, spotRatesN)
    print DF
    utils.printTime('Generating discount factors', start_time)


# final cleanup below
class CVAIRS:
    def __init__(self, notional, tenor, paymentFreq, fixedRate, floatingRate):
        self.notional = notional
        self.tenor = tenor
        self.paymentFreq = paymentFreq
        self.fixedRate = fixedRate
        self.floatingRate = floatingRate
        tenorFraction = self.paymentFreq / self.tenor
        self.timeSteps = np.arange(0, 1 + tenorFraction, tenorFraction)


d = CVAIRS(100000, 60., 6, 0.05, 0.05)


# timeFunctions()

def swaps(t, f, d):
    swapPayments = []
    for time, fwd, df in zip(t[1:], f, d):
        payment = calculateSwapPayment(notional, dt, df, fwd, fixedRate)
        swapPayments.append(payment)
    return swapPayments


def generateMtM(swapPrices=list):
    mtm = []
    for index, item in enumerate(sw):
        mtm.append(sum(sw[index:]))
    # append 0 to end since by definition price at maturity = 0
    mtm.append(0);
    return mtm


def calculateExposure(markToMarketValues=list):
    exposure = []
    for mtm in markToMarketValues:
        exposure.append(max(mtm, 0))
    return exposure
    # return np.asarray(exposure, float)


def getAverageValuesForTimeStepGivenData(data=dict):
    # get data from supplied dictionary / hashtable
    exp = data[EXPOSURE]
    discFactor = data[DISCOUNT_FACTOR]
    pd = data[PROBABILITY_OF_DEFAULT]
    ps = data[PROBABILITY_OF_SURVIVAL]

    # discount factor at t0 = 1.0
    discFactor.insert(0, 1.0)
    # pd.insert(0, 0)

    if len(exp) != len(discFactor) != pd:
        print 'ERROR: Data sources not equal. The length of each data source must the same.'
        return
    else:
        result = []
        print 'exposure: ', exp, len(exp)
        print 'disc: ', discFactor, len(discFactor)
        print 'pd: ', pd, len(pd)
        print 'ps: ', ps, len(ps)
        for index, (e, d, p, s) in enumerate(zip(exp, discFactor, pd, ps)):
            """
            object vs tuple here, tuple chosen since it is most probably quicker to enumerate
            than a dictionary with keys. A further enhancement to consider would be to use np.array instead of list / dict

            tStep = {exposure: utils.getTPlusFromList(exp, index, True),
                     discountFactor: utils.getTPlusFromList(discFactor, index, True),
                      ...}
            """

            tStep = (utils.getTPlusFromList(exp, index, True),
                     utils.getTPlusFromList(discFactor, index, True),
                     p,
                     s
                     )

            result.append(tStep)

    return result


def mc_sim_exposure(noOfSims=int):
    pass

fwdRates = genFwdCurve(timeSteps, spotRatesN)
probDef = genDefaultProbabilities(0.03, timeSteps)
discFac = genDiscountFactors(timeSteps, spotRatesN)
sw = swaps(timeSteps, fwdRates, discFac)

# discretizing the integral into time steps
# value taken form the middle of teh curves. ti-1i1/2 -> average taken from mid point

exposure = calculateExposure(generateMtM(sw))
ps = [1 - recoveryRate] * len(exposure)

noOfSimulations = 10
# TODO - plugin ee from LMM_sim
expectedExposure = mc_sim_exposure(10)

# smoothCurve = np.linspace(exposure.min(), exposure.max(), 300)
# smoothedCurve = interpolate.spline(exposure)



# discrete integration
cvaData = getAverageValuesForTimeStepGivenData(
    {EXPOSURE: exposure,
     DISCOUNT_FACTOR: discFac,
     PROBABILITY_OF_DEFAULT: probDef,
     PROBABILITY_OF_SURVIVAL: ps})


# cva = reduce(lambda x, y: x*y, item)

def sumCVA(data=list):
    cva = 0.
    for datum in data:
        cva += np.prod(datum)
    return cva


print cvaData, len(cvaData)

print 'CVA: ' + sumCVA(cvaData)


# print mergeLists(avgDiscFactor, avgExposure)

def getCvaForTimeStep(e=float, df=float, pd=float, rr=float):
    return e * df * pd * (1 - rr)


# cva for each time step
# cva = DFi*PD*(1-r)
# plot.plot(timeSteps,smoothCurve,marker='o', linestyle='--', color='r')
# plot.xlabel("Time (t)")
# plot.ylabel("$")
# plot.title('Exposure of Interest Rate Swap')
# plot.legend()
# plot.show()
# def calculateCVA(exposure=list)
