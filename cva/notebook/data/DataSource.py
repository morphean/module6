import json
from urllib2 import Request, urlopen, quote

from pandas import DataFrame

BB_5YEAR = '5_YEAR'
# BB_3YEAR = '3_YEAR'
# BB_1YEAR = '1_YEAR'
BB_1MONTH = '1_MONTH'


class DataSource(object):
    def __init__(self, tickerName=str, timePeriod=str):
        """

        retrieves historical data for the ticker provided, input is url encoded before request
        so inputs like 'BRIT:IND' are perfectly fine.

        :param tickerName: reference name
        :param timePeriod:
        """
        self.tickerName = quote(tickerName)

        # time period constants
        self.BB_3YEAR = '3_YEAR'
        self.BB_1YEAR = '1_YEAR'
        self.BB_10YEAR = '10_YEAR'

    @property
    def BB_5YEAR(self):
        return '5_YEAR'

    def getBloombergData(self):
        """
        gets data for ticker from Bloomberg
        :return:
        """
        request = Request('http://www.bloomberg.com/markets/api/bulk-time-series/price/' +
                          self.tickerName + '?timeFrame=' + self.BB_1YEAR)
        response = urlopen(request)
        ts = response.read()
        data = json.loads(ts)
        if data[0]:
            datFrame = DataFrame(data[0]['price'])
        else:
            print 'Error processing request, returning empty data'
            DataFrame([])

        return datFrame

    def getCNBCData(self):
        """
        get cds data from cnbc:

        possible values
        ATCD5|
        BECD5
        CNCD5
        DKCD5
        DUCD5
        EGCD5
        FICD5
        FRCD5
        DECD5
        GRCD5
        HUCD5
        INOCD5
        IECD5
        ITCD5
        JPCD5
        KRCD5
        NLCD5
        PNCD5
        PTCD5
        SKCD5
        ESCD5
        SECD5
        CHCD5
        GBCD5
        USCD5
        :return:
        """
        request = Request('http://quote.cnbc.com/quote-html-webservice/quote.htm?&symbols=' + self.tickerName +
                          '&requestMethod=quick&noform=1&realtime=1&client=flexQuote&output=json&random=1320681808606')
        response = urlopen(request)
        ts = response.read()
        data = json.loads(ts)
        if data[0]:
            datFrame = DataFrame(data[0]['price'])
        else:
            print 'Error processing request, returning empty data'
            DataFrame([])

        return datFrame


uk = DataSource('BRIT:IND', DataSource.BB_5YEAR).getBloombergData()
us = DataSource('BUSY:IND', DataSource.BB_5YEAR).getBloombergData()

de = DataSource('BGER:IND', DataSource.BB_5YEAR).getBloombergData()
jp = DataSource('BJPN:IND', DataSource.BB_5YEAR).getBloombergData()
aus = DataSource('BAUS:IND', DataSource.BB_5YEAR).getBloombergData()
