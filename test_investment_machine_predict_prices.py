import unittest

import investment_machine_predict_prices


class TestApp(unittest.TestCase):
    def test_hello(self):
        self.assertEqual("hello", investment_machine_predict_prices.hello())


if __name__ == "__main__":
    unittest.main()
