from numpy import array


class CDSSimulationModel(object):
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

        # expected exposure for counterParty A (using positive mtm)
        self.eeA = [max(L - K, 0) for L, K in self.payments]

        # expected exposure for counterParty B (using negative mtm)
        self.eeB = [min(L - K, 0) for L, K in self.payments]

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


if __name__ == "__main__":
    print 'Testing CDS Simulation Model'
    debug = True
