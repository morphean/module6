from numpy import array, ones, zeros, linspace, exp, put, sum, mean
from pandas import DataFrame

from utils import getTPlusFromList

debug = False


class CreditDefaultSwap(object):
    def __init__(self, N=float, timesteps=int, discountFactors=list, lamda=float, seed=float, Dt=0.25,
                 recoveryRate=0.4):
        self.N = N  # notional
        self.__lamda = lamda
        self.__seed = seed
        self.recoveryRate = recoveryRate
        self.Dt = Dt
        self.timesteps = linspace(0, 1, timesteps + 1)
        self.discountFactors = discountFactors
        self.pT = self.generateProbSurvival()
        self.premiumLegsSpread = self.calculatePremiumLegSpreads()
        self.__premiumLegSum = sum(self.premiumLegsSpread)
        self.pD = self.generateProbDefault()
        self.defaultLegsSpread = self.calculateDefaultLegSpreads()
        self.__defaultLegSum = sum(self.defaultLegsSpread)
        self.fairSpread = self.premiumLegsSpread / self.defaultLegsSpread
        self.__expectedExposure = []

    @property
    def premiumLegSum(self):
        return self.__premiumLegSum

    @property
    def defaultLegSum(self):
        return self.__defaultLegSum

    @property
    def markToMarket(self):
        return self.__premiumLegSum - self.__defaultLegSum

    @property
    def lamda(self):
        return self.__lamda

    @property
    def seed(self):
        return self.__seed

    def generateDiscountFactors(self):
        pass

    def generateProbSurvival(self):
        """
        using $\exp^{-\lambda*T_i}$
        :return:
        """
        pt = ones(len(self.timesteps))
        for index, t in enumerate(self.timesteps):
            if t > 0:
                ps = exp(self.lamda * -1 * t)
                put(pt, index, ps)
        return pt

    def generateProbDefault(self):
        """
        using $P(T,0) = P_{i-1} - P_i$
        :return:
        """
        pd = zeros(len(self.timesteps))
        for index, pt in enumerate(self.pT):
            if index > 0:
                pdi = self.pT[index - 1] - pt
                put(pd, index, pdi)

        return pd

    def calculatePremiumLegSpreads(self):
        """
        returns the list of the premium leg values
        :return: array
        """
        #          assume 1%
        spreads = zeros(len(self.timesteps))
        for index, (df, pt) in enumerate(zip(self.discountFactors, self.pT)):
            if index > 0:
                spread = self.N * self.Dt * self.seed * df * pt
                put(spreads, index, spread)
        return spreads

    def calculateDefaultLegSpreads(self):
        """
        returns the list of the default leg values
        :return: array
        """
        #          assume 1%
        spreads = zeros(len(self.timesteps))
        for index, (df, pd) in enumerate(zip(self.discountFactors, self.pD)):
            if index > 0:
                spread = self.N * (1 - self.recoveryRate) * df * pd
                put(spreads, index, spread)
        return spreads

    def calcCVA(self, expectedExposure=array):
        cvaData = DataFrame()
        cvaData['t'] = self.timesteps[1:]
        cvaData['discountFactor'] = self.discountFactors
        cvaData['pd'] = self.pD[1:]
        cvaData['1-R'] = [1 - self.recoveryRate] * len(self.pD[1:])
        cvaData['exposure'] = [getTPlusFromList(expectedExposure, i, True) for i in range(len(expectedExposure))]
        cvaData['cvaPerTimeStep'] = cvaData['discountFactor'] * cvaData['pd'] * cvaData['1-R'] * cvaData['exposure']
        cva = cvaData['cvaPerTimeStep'].sum()
        cvaData.describe()
        print cvaData
        print 'CVA = ', cva
        return cva

        return True

    def calcBVA(self, eeA=array, eeB=array):
        # LAST BIT TO DO
        return True

    def getExpectedExposureA(self, simData=array):
        expectedExposure = mean(array([sim.expA for sim in simData]), axis=0)
        return expectedExposure

    def getExpectedExposureB(self, simData=array):
        expectedExposure = mean(array([sim.expB for sim in simData]), axis=0)
        return expectedExposure


class Simulation(object):
    """
    this class is used a container for a single simulation
    providing convenenience methods to retrieve
    markToMarket values
    eeA = expected exposure from the simulation for Counterparty A
    eeB = expected exposure from the simulation for Counterparty B
    """

    def __init__(self, liborTable=array, dfTable=array, notional=1000000, dt=0.25, k=0.04):
        """

        :type k: float
        :type dt: float
        :type notional: float
        :type dfTable: ndarray
        :type liborTable: ndarray
        """
        self.__liborTable = liborTable
        self.__dfTable = dfTable

        # calculate payments for each timestep using the given notional, tenor, fixed rate,
        # floating(simulated) and discount factors (simulated)
        self.payments = self.calcPayments(notional, dt, k)

        self.mtm = array([flt - fxd for flt, fxd in self.payments])

        # exposure for counterParty A (using positive mtm)
        self.expA = array([max(L - K, 0) for L, K in self.payments])

        # exposure for counterParty B (using negative mtm)
        self.expB = array([min(L - K, 0) for L, K in self.payments])

    def liborTable(self):
        return self.__liborTable

    def dfTable(self):
        return self.__dfTable

    def calcPayments(self, notional=float, dt=float, fixed=-1.0):
        """
        calculate payments for the simulation of the Fwd rates and discount factors
        given notional and tenor

        if fixed is set it will use a fixed rate
        there is the possibility here of a negative interest rate but that is outside the
        scope of this exercise
        :param notional:
        :param dt:
        :param fixed:
        :return: float
        """
        payments = []

        for index in range(0, len(self.__liborTable)):
            fwdCurve = self.__liborTable[:, index]
            df = self.__dfTable[1:, index]

            floatingLeg = [fwd * dfi * notional * dt for fwd, dfi in zip(fwdCurve, df)]
            fixedLeg = [fixed * dfi * notional * dt for dfi in df]
            # fixedLeg[len(self.__liborTable)] = 0
            payments.append([sum(floatingLeg), sum(fixedLeg)])

            if debug:
                print 'from t-', index, '--- fixed - ', sum(fixedLeg), '--- floating -', sum(floatingLeg)
                print fixedLeg
                print '--'
                print floatingLeg
                print '--'

        return payments


def genDiscountFactors(t=list, s=list):
    discount = []

    for i, (ti, si) in enumerate(zip(t[1:], s)):
        #
        # DFi = 0.0
        #
        # if i > 0:
        a = si * ti * -1
        DFi = exp(a)

        discount.append(DFi)

    return discount


# plot tbis against reference curve
def genFwdCurve(t=list, s=list):
    curve = []

    for i, (ti, si) in enumerate(zip(t, s)):
        if i >= 0:
            sim1 = s[i - 1] if i > 0 else 0.0
            tim1 = t[i - 1] if i > 0 else 0.0
            Li = ((si * ti) - (sim1 * tim1)) / (ti - tim1)
            curve.append(Li)
    return curve


if __name__ == "__main__":
    print 'Testing Basket Default Swap'

    # df = [0.9803, 0.9514, 0.9159, 0.8756, 0.8328]
    # seed = 0.01
    # payments = 4
    # notional = 1000000
    # # fixed hazard rates
    # lamda = 0.03
    #
    # b = BasketCDS(N=notional, timesteps=5, discountFactors=df, lamda=lamda, dt=1.0, seed=seed,)
    # b.recoveryRate = 0.5
    # b.buildFlatTermStructure([57]*5)
