from numpy import array, log, sqrt, square, random
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr, norm, t

import rng


def getPearsonRank(data=array):
    """

    method to provide Pearson Rank calculation given a data structure of
    two normally distributed variables

    :param data:
    :return: float;
    """

    return pearsonr(data[0], data[1])


def getSpearmanRank(data=array):
    """

    :param data:
    :return tuple: rank co-efficient, p value
    """
    return spearmanr(data[0], data[1])


def simulateCopula(simulations=10, type=str('g'), rho=float, lamda=tuple, tDof=4, basketSize=5, useGPU=False):
    result = []
    """

    $\tau = F^{-1}(u) = -\frac{log(1-u)}{\lambda}$

    """
    print 'simulating t distribution' if type == 't' else 'simulating gaussian dist'

    for z in xrange(0, simulations):
        # for the t distribution we use the same method but
        # sample from the chisquared distribution
        # if GPU is enabled, hand over to GPU to provide random number sample
        if useGPU and type == 'g':
            z1, z2, z3, z4, z5 = rng.getPseudoRandomNumbers_Standard_cuda(basketSize)
        else:
            z1, z2, z3, z4, z5 = random.chisquare(tDof, size=basketSize) if type == 't' else random.normal(size=5)
        # z1, z2, z3, z4, z5 = chi2.rvs(1, size=5) if type == 't' else random.normal(size=5)

        x1 = z1

        # using factorised copula procedure
        # $A_i = w_iZ + \sqrt{1-w{^2}{_i}\Epsilon_i $
        x2, x3, x4, x5 = [z1 * rho + sqrt(1 - square(rho)) * zn for zn in [z2, z3, z4, z5]]

        # converting to normal variables from t or normal distribution successfully
        # via cdf of relevant distribution
        if type == 't':
            u1, u2, u3, u4, u5 = [t.cdf(x, 1) for x in [x1, x2, x3, x4, x5]]
        else:
            u1, u2, u3, u4, u5 = [norm.cdf(x) for x in [x1, x2, x3, x4, x5]]
        u = [u1, u2, u3, u4, u5]
        # $\tau_i = -\frac{-log(1-u)}{\lambda_i} $
        tau1, tau2, tau3, tau4, tau5 = [-log(1 - u) / lamda[index] for index, u in enumerate(u)]
        result.append({'z1': z1,
                       'z2': z2,
                       'z3': z3,
                       'z4': z4,
                       'z5': z5,
                       'x1': x1,
                       'x2': x2,
                       'x3': x3,
                       'x4': x4,
                       'x5': x5,
                       'u1': u1,
                       'u2': u2,
                       'u3': u3,
                       'u4': u4,
                       'u5': u5,
                       'tau1': tau1,
                       'tau2': tau2,
                       'tau3': tau3,
                       'tau4': tau4,
                       'tau5': tau5,
                       })

    return DataFrame(result)
