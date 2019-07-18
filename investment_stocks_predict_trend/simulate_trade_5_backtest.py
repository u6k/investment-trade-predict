from simulate_trade_5 import SimulateTrade5


if __name__ == "__main__":
    s3_bucket = "u6k"
    input_prices_base_path = "ml-data/stocks/preprocess_1.20190717"
    input_preprocess_base_path = "ml-data/stocks/preprocess_6.20190717"
    input_model_base_path = "ml-data/stocks/predict_3_preprocess_6.20190717"
    output_base_path = "ml-data/stocks/simulate_trade_5_backtest.20190717"

    start_date = "2018-01-01"
    end_date = "2018-12-31"

    SimulateTrade5().backtest_singles(
        start_date,
        end_date,
        s3_bucket,
        input_prices_base_path,
        input_preprocess_base_path,
        input_model_base_path,
        output_base_path
    )

    SimulateTrade5().report_singles(
        s3_bucket,
        output_base_path
    )
