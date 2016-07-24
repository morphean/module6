import ghalton
import sobol_seq
from accelerate.cuda.rand import PRNG, QRNG
from numpy import array, empty, random, square, log, sqrt


def getPseudoRandomNumbers_Uniform(length=int):
    """

    generates a an array of psuedo random numbers from uniform distribution using numpy

    :param length:
    :return:
    """
    return random.uniform(size=length)


def getPseudoRandomNumbers_Uniform_cuda(length=int):
    # type: (object) -> object
    """

    generates a an array of psuedo random numbers from uniform distribution using CUDA

    :rtype: ndarray
    :param length:
    :return:
    """
    prng = PRNG(rndtype=PRNG.XORWOW)
    rand = empty(length)
    prng.uniform(rand)

    return rand


def getPseudoRandomNumbers_Standard(shape=tuple):
    """

    generates a an array of psuedo random numbers from uniform distribution using numpy

    :param length:
    :return:
    """
    return random.normal(size=shape)


def getPseudoRandomNumbers_Standard_cuda(shape=tuple):
    # type: (object) -> object
    """

    generates a an array of psuedo random numbers from standard normal distribution using CUDA

    :rtype: ndarray
    :param length:
    :return:
    """
    prng = PRNG(rndtype=PRNG.XORWOW)
    rand = empty(shape)
    prng.normal(rand, 0, 1)

    return rand


def getSOBOLseq_standard(shape=tuple):
    """
    generate a SOBOL sequence
    
    :param shape: tuple of row, column of the desired return matrix
    :return:
    """
    return sobol_seq.i4_sobol_generate_std_normal(shape)


# def getSOBOLseq_uniform(shape=tuple):
#     return sobol_seq.i4_uniform(shape)

def getSOBOLseq_cuda(length=int):
    """

    returns an nd array of supplied length containing a SOBOL sequence of int64

    only for use on systems with CUDA libraries installed

    :param length:
    :return ndarray:
    """
    qrng = QRNG(rndtype=QRNG.SOBOL64)
    rand = empty(shape=length, dtype=int)
    qrng.generate(rand)

    return rand


def getHaltonSeq(dimensions=int, length=int):
    """

    returns an array Halton sequence of quasi random number

    :param dimensions: number of dimensions of matrix (columsn)
    :param length: number of sequences returned (rows)
    :return:
    """
    sequencer = ghalton.Halton(dimensions)
    return sequencer.get(length)


def getBoxMullerSample(randSource=array):
    t = 1.0
    while t >= 1.0 or t == 0.0:
        randValues = random.choice(randSource, 2)
        x = randValues[0]
        y = randValues[1]
        t = square(x) + square(y)

    result = x * sqrt(-2 * log(t) / t)
    return result
