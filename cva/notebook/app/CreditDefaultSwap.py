from numpy import array, ones, zeros, linspace, exp, put, sum, log


class CreditDefaultSwap(object):
    def __init__(self, N=float, timesteps=int, discountFactors=list, lamda=float, seed=float):
        self.N = N  # notional
        self.__lamda = lamda
        self.__seed = seed
        self.recoveryRate = 0.4
        self.Dt = 0.25
        self.timesteps = linspace(0, 1, timesteps + 1)
        self.discountFactors = discountFactors
        self.pT = self.generateProbSurvival()
        self.premiumLegsSpread = self.calculatePremiumLegSpreads()
        self.__premiumLegSum = sum(self.premiumLegsSpread)
        self.pD = self.generateProbDefault()
        self.defaultLegsSpread = self.calculateDefaultLegSpreads()
        self.__defaultLegSum = sum(self.defaultLegsSpread)
        self.fairSpread = self.premiumLegsSpread / self.defaultLegsSpread

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

    def generateProbSurvival(self):
        pt = ones(len(self.timesteps))
        for index, t in enumerate(self.timesteps):
            if t > 0:
                ps = exp(self.lamda * -1 * t)
                put(pt, index, ps)
        return pt

    def generateProbDefault(self):
        pd = zeros(len(self.timesteps))
        for index, pt in enumerate(self.pT):
            if index > 0:
                pdi = self.pT[index - 1] - pt
                put(pd, index, pdi)

        return pd

    def calculatePremiumLegSpreads(self):
        #          assume 1%
        spreads = zeros(len(self.timesteps))
        for index, (df, pt) in enumerate(zip(self.discountFactors, self.pT)):
            if index > 0:
                spread = self.N * self.Dt * self.seed * df * pt
                put(spreads, index, spread)
        return spreads

    def calculateDefaultLegSpreads(self):
        #          assume 1%
        spreads = zeros(len(self.timesteps))
        for index, (df, pd) in enumerate(zip(self.discountFactors, self.pD)):
            if index > 0:
                spread = self.N * (1 - self.recoveryRate) * df * pd
                put(spreads, index, spread)
        return spreads

    def calcCVA(self, expectedExposure=array):
        # LAST BIT to DO
        return True

    def calcBVA(self, eeA=array, eeB=array):
        # LAST BIT TO DO
        return True


class BasketCDS(CreditDefaultSwap):
    def __init__(self, N=float, timesteps=int, discountFactors=list, lamda=float, seed=float, dt=float):
        super(BasketCDS, self).__init__(N, timesteps, discountFactors, lamda, seed)
        self.hazardRates = self.generateHazardRates()
        self.dt = dt

    def bootstrapCurve(self, spreads=list):
        ts = []
        impPs = []

        for index, spread in enumerate(spreads):
            # calc implied survival prob
            x = dict()
            if index == 0:
                impPs.append(1.0)
            else:
                # $\frac{(1-R)}{(1-R)+\deltaT*s}$
                isp = (1 - self.recoveryRate) / ((1 - self.recoveryRate) + self.dt * spread / 10000)
                impPs.append(isp)
        return ts

    def buildTermStructure(self, spreads=list):
        """
        returns terms structure matrix
        spreads should be supplied unscaled (ie as quoted from market data)
        :type spreads: list
        """
        t0 = [0.0]
        oneMr = 1 - self.recoveryRate

        # spreads are input as basis points so scale them for calculations
        scaledSpreads = [spreads] / 10000

        impliedProbSurvival = [1.0]
        impliedProbSurvival[1] = (oneMr) / ((oneMr) + self.dt * scaledSpreads[0])

        t1 = 0.0
        t2 = self.discountFactors[0] * ((1 - self.recoveryRate) * impliedProbSurvival) - (
                                                                                             1 - self.recoveryRate + self.dt *
                                                                                             scaledSpreads[1]) *
        t3 = self.discountFactors[0] * ((1 - self.recoveryRate) * impliedProbSurvival) - (
            1 - self.recoveryRate + self.dt * scaledSpreads[2])

        return t0

    def getDefaultPeriod(self):
        """
        if ProbOfSurv >= hazard rate -> default
        return tau

        :return: float
        """
        tau
        return 0.0

    def getExactDefaultTIme(self):
        """
        $ \deltat = - \frac{}{} log \frac{1-u}{P(0,t_{n-1}} $
        :return:
        """
        return dt

    def generateHazardRates(self):
        """
        generate hazard rates using:
        $\Lambda_m = -\frac({1}{\delta t} log \frac{P(0,t_m)}{P(0,t_{m-1})} $
        :return:
        """
        hz = zeros(len.self.timesteps)
        for index, pt in enumerate(self.pT):
            if index > 0:
                hz_m = -(1 / self.Dt) * log(pt / self.pT[index - 1])
                put(hz_m, index, hz)

        return hz
