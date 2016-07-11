from unittest import TestCase

from notebook.app import utils


class TestUtils(TestCase):
    def test_getTMinusFromList(self):
        source = [1, 2, 3, 4]
        expectedResult = 2

        assert expectedResult == utils.getTMinusFromList(source, 2)
        assert 0 == utils.getTMinusFromList(source, 0)

    def test_getTPlusFromList(self):
        source = [1, 2, 3, 4]
        expectedResult = 4
        assert expectedResult == utils.getTPlusFromList(source, 2)
        assert 0 == utils.getTPlusFromList(source, 3)
