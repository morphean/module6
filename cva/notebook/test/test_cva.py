from unittest import TestCase

from notebook.app import cva


class TestGenerateForwardRateCurve(TestCase):
    # def setUp(self):
    #     spotRates = [(0, 0),
    #                  (0.25,    0.04),
    #                  (0.5,    0.25),
    #                  (0.75,    0.32),
    #                  (1, 0.40)]
    #     expectedOutput = [0.04, 0.46, 0.45999999999999996, 0.6400000000000001]
    def test_generateForwardRateCurve(self):
        spotRates = [(0, 0),
                     (0.25, 0.04),
                     (0.5, 0.25),
                     (0.75, 0.32),
                     (1, 0.40)]
        expectedOutput = [0.04, 0.46, 0.45999999999999996, 0.6400000000000001]
        result = cva.generateForwardRateCurve(spotRates)
        assert expectedOutput == result

    def test_genFwdCurve(self):
        t = [0, 0.25, 0.5, 0.75, 1]
        s = [0, 0.04, 0.25, 0.32, 0.40]
        expectedOutput = [0.04, 0.46, 0.45999999999999996, 0.6400000000000001]
        result = cva.genFwdCurve(t, s)
        assert expectedOutput == result

    def test_getCvaForTimeStep(self):
        expectedOutput = 3.0
        result = cva.getCvaForTimeStep(1, 2, 3, 0.5)
        assert expectedOutput == result

# class TestGenFwdCurve(TestCase):
#
#     def test_genFwdCurve(self):
#         spotRates = [(0, 0),
#                      (0.25,    0.04),
#                      (0.5,    0.25),
#                      (0.75,    0.32),
#                      (1, 0.40)]
#         expectedOutput = [0.04, 0.46, 0.45999999999999996, 0.6400000000000001]
#         result = cva.genFwdCurve(spotRates)
#         assert expectedOutput == result
