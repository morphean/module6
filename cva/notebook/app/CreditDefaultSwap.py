from numpy import array, ones, zeros, linspace, exp, put, sum


class CreditDefaultSwap:
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

        return True

    def calcBVA(self, eeA=array, eeB=array):
        return True
