import os


class CreditCurve(object):
    def __init__(self, timeSeries=DataFrame, discountFactor=DataFrame):
        pass


class YieldCurve(object):
    def __init__(self, filename=str):
        directory = os.path.dirname(__file__)
        filePath = os.path.join(directory, '../data/' + filename)
        pass
