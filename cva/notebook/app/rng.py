from accelerate.cuda.rand import PRNG, QRNG
from numpy import ndarray, empty, random, square, log, sqrt, put


def getPseudoRandomNumbers_Uniform(length):
    """

    generates a an array of psuedo random numbers from uniform distribution using numpy

    :param length:
    :return:
    """
    rand = empty(length)
    for i in range(length):
        put(rand, i, random.uniform())
    return rand


def getPseudoRandomNumbers_Uniform_cuda(length):
    """

    generates a an array of psuedo random numbers from uniform distribution using CUDA

    :param length:
    :return:
    """
    prng = PRNG(rndtype=PRNG.XORWOW)
    rand = empty(length)
    prng.uniform(rand)

    return rand


def getSOBOLseq_cuda(length=int):
    """

    returns an nd array of supplied length containing a SOBOL sequence of int64

    only for use on systems with CUDA libraries installed

    :param length:
    :return ndarray:
    """
    qrng = QRNG(rndtype=QRNG.SOBOL64)
    rand = ndarray(shape=100000, dtype=int)
    qrng.generate(rand)

    return rand[:length]


def getBoxMullerSample(randSource=ndarray):
    t = 1.0
    while t >= 1.0 or t == 0.0:
        randValues = random.choice(randSource, 2)
        x = randValues[0]
        y = randValues[1]
        t = square(x) + square(y)

    result = x * sqrt(-2 * log(t) / t)
    return result
