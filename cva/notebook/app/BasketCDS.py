import CreditDefaultSwap


class BasketCDS(CreditDefaultSwap):
    def __init__(self, N=float, timesteps=int, discountFactors=list, lamda=float, seed=float, dt=float):
        super(BasketCDS, self).__init__(N, timesteps, discountFactors, lamda, seed)
        # self.hazardRates = self.generateHazardRates()
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

    # noinspection PyTypeChecker
    def scaleFromBasisPoints(self, toScale=any):
        """
        Takes an input either list of floats or a single float
        and scales it to normal number for calculation purposes
        ( divide by 10^4 )
        :param toScale:
        :return:
        """
        if type(toScale) is list:
            return [item / 10000.0 for item in toScale]
        if type(toScale) is float:
            return toScale / 10000

    # noinspection PyTypeChecker
    def scaleToBasisPoints(self, toScale=any):
        """
        Takes an input either list of floats or a single float
        and scales it to normal number for calculation purposes
        ( multiply by 10^4 )
        :param toScale:
        :return:
        """
        if type(toScale) == list:
            return [item * 10000 for item in toScale]
        if type(toScale) is float:
            return toScale * 10000

    def getFirstImpliedSurvivalProbability(self, spread=float):
        return (1 - self.recoveryRate) / (1 - self.recoveryRate) + self.dt * spread[0]

    def buildFlatTermStructure(self, spreads=list):
        """
        returns flat term structure for a 5Y tenor


        :param spreads:
        :return:
        """

        # TODO USE A DATAFRAME
        scaledSpreads = self.scaleFromBasisPoints(spreads)

        # implSP = [1.0]
        #
        # implSP[1] = self.getFirstImpliedSurvivalProbability(scaledSpreads)
        #
        oneMr = 1 - self.recoveryRate
        #
        # quotients = [df * (oneMr+self.dt*spread) for df, spread in zip(scaledSpreads, self.discountFactors)]
        #
        # firstTerms = [0]*len(spreads-1)
        #
        # lastTerms = [0]*len(spreads-1)
        #
        # lastTerms.insert(0, implSP[1] * (oneMr/(oneMr+self.dt*scaledSpreads[1])))
        #
        # implSP[2] = firstTerms[0] + lastTerms[0]
        #
        # lastTerms.insert(1, implSP[2] * (oneMr/(oneMr+self.dt*scaledSpreads[2])))
        #
        # implSP[3] = firstTerms[1] + lastTerms[1]
        #
        # lastTerms.insert(2, implSP[3] * (oneMr/(oneMr+self.dt*scaledSpreads[3])))
        #
        # implSP[4] = firstTerms[2] + lastTerms[2]
        #
        # lastTerms.insert(3, implSP[4] * (oneMr/(oneMr+self.dt*scaledSpreads[4])))
        # implSP[5] = firstTerms[3] + lastTerms[3]
        # # TODO - refactor above in to loop
        #
        # ips, lt, ft = [1.0], [],[]
        # qt = [df * (oneMr+self.dt*spread) for df, spread in zip(scaledSpreads, self.discountFactors)]

        # for index, spread in enumerate(scaledSpreads):
        #     if index == 0:
        #         x = self.getFirstImpliedSurvivalProbability(spread)
        #         ips.append(x)
        #         ft.append(0.)
        #     else:
        #         x = ips[index] * (oneMr/(oneMr+self.dt*spread))
        #         ft.append(0.)
        #         lt.insert(x)
        #         ips.append(ft[index-2]+lt[index-2])

        qt = []
        pd = DataFrame()
        pd['year'] = range(1, len(scaledSpreads) + 1)
        pd['dt'] = [1] * len(scaledSpreads)
        pd['spread'] = scaledSpreads
        pd['df'] = self.discountFactors

        for r in pd.itertuples():
            # sample structure
            # Pandas(Index=0, year=1, dt=1, spread=0.0057000000000000002, df=0.98029999999999995)
            # Pandas(Index=1, year=2, dt=1, spread=0.0057000000000000002, df=0.95140000000000002)
            # Pandas(Index=2, year=3, dt=1, spread=0.0057000000000000002, df=0.91590000000000005)
            # Pandas(Index=3, year=4, dt=1, spread=0.0057000000000000002, df=0.87560000000000004)
            # Pandas(Index=4, year=5, dt=1, spread=0.0057000000000000002, df=0.83279999999999998)
            if r[0] == 0:
                qt.append(0.0)
            else:
                qt.append(r[4] * oneMr + r[2] * r[3])

        pd['qt'] = qt
        pd.head()
        print pd
        ips = [1.0]
        for i in xrange(len(spreads)):
            if i == 0:
                x = (1.0)
            if i == 1:
                x = self.getFirstImpliedSurvivalProbability(spreads[0])
            if i > 1:
                x = oneMr / ()

            ips.append(x)
        return pd

    def buildTermStructure(self, spreads=list):
        """
        returns terms structure matrix for a 5Y tenor
        spreads should be supplied unscaled (ie as quoted from market data)
        :type spreads: list
        """
        ts0 = [0.0]
        oneMr = 1 - self.recoveryRate

        # spreads are input as basis points so scale them for calculations
        scaledSpreads = [spread / 10000 for spread in spreads]

        impliedProbSurvival = [1.0]
        impliedProbSurvival[1] = (oneMr) / ((oneMr) + self.dt * scaledSpreads[0])

        lastTerms = []
        lastTerms[0] = impliedProbSurvival[1] * (oneMr) / (oneMr + self.dt)

        # now we have $T_1$ we can work out the rest of the implied P(0,T)

        ts1 = [0.0]
        t2 = self.discountFactors[0] * ((1 - self.recoveryRate) * impliedProbSurvival[0]) - (
            1 - self.recoveryRate + self.dt *
            scaledSpreads[1])
        t3 = self.discountFactors[0] * ((1 - self.recoveryRate) * impliedProbSurvival[1]) - (
            1 - self.recoveryRate + self.dt * scaledSpreads[2])

        return ts0

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
