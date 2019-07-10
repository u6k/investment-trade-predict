from logging import Formatter, getLogger, StreamHandler, DEBUG


def get_app_logger(name=None):
    if name is not None:
        logger = getLogger(f"investment_stock_predict_trend.{name}")
    else:
        logger = getLogger("investment_stock_predict_trend")
    formatter = Formatter("%(asctime)-15s - %(levelname)-8s - %(message)s")
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(formatter)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    return logger
