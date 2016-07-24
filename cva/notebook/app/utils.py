import time

from pandas import read_excel


def getTPlusFromList(source=list, index=int, average=bool):
    """
    utility to get t+1 from supplied list

    :rtype: any
    :param source: list
    :param index: int
    :param average: bool (optional) if selected will return the average of i and i+1
    :rtype any
    """
    if index + 1 == len(source):
        return 0
    else:
        return source[index + 1] if not average else (source[index] + source[index + 1]) * 0.5


def getTMinusFromList(source=list, index=int):
    """
    utility to get t-1 from supplied list

    :rtype any
    :param source: list
    :param index: int
    """
    if index - 1 < 0:
        return 0
    else:
        return source[index - 1]


def printTime(message=str, start=int):
    """
    utility method to print a time it took to run a function given its start time

    :param message:
    :param start:
    :return:
    """
    print (message + ' took: ' + repr(time.clock() - start) + ' seconds')


def loadBasketData(basket=list, srcDataPath=str):
    """
    loads basket data given list of names in an excel sheet
    could be adapted to use JSON, CSV etc. using the relevant method from pandas library
    :param basket:
    :return: array
    """
    basketData = []
    # read data
    for item in basket:
        itemData = read_excel(srcDataPath, sheetname=item, header=1, skiprows=0)
        basketData.append(itemData)

    return basketData
