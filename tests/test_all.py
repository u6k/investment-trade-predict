from .context import investment_stocks_predict_trend

import unittest


class TestApp(unittest.TestCase):
    def test_hello(self):
        self.assertEqual("hello", investment_stocks_predict_trend.hello())


if __name__ == "__main__":
    unittest.main()
