from logging import Formatter, getLogger, StreamHandler, DEBUG
from uuid import uuid4


def get_app_logger():
    logger = getLogger(f"investment_stock_predict_trend.{str(uuid4())}")
    formatter = Formatter("%(asctime)-15s - %(levelname)-8s - %(message)s")
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(formatter)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    return logger
