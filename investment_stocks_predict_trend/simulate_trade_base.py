from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np

from app_logging import get_app_logger
import app_s3


class SimulateTradeBase():
    def __init__(self, job_name):
        self._job_name = job_name

    def simulate(self, s3_bucket, input_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.simulate")
        L.info(f"{self._job_name}.simulate: start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.simulate_impl)(ticker_symbol, s3_bucket, input_base_path, output_base_path) for ticker_symbol in df_companies.index])

        for result in results:
            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        app_s3.write_dataframe(df_result, s3_bucket, f"{output_base_path}/companies.csv")

        L.info(f"{self._job_name}.simulate: finish")

    def simulate_impl(self, ticker_symbol, s3_bucket, input_base_path, output_base_path):
        raise Exception("Not implemented.")

    def simulate_report(self, start_date, end_date, s3_bucket, base_path):
        L = get_app_logger(f"{self._job_name}.simulate_report")
        L.info(f"{self._job_name}.simulate_report: start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.report_impl)(ticker_symbol, start_date, end_date, s3_bucket, base_path) for ticker_symbol in df_companies.index])

        for result in results:
            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

            for k in result.keys():
                if k != "ticker_symbol" and k != "exception":
                    df_result.at[ticker_symbol, k] = result[k]

        L.info(df_result[["trade_count", "win_rate", "expected_value", "expected_value_win_only", "risk", "profit_factor"]].describe())

        app_s3.write_dataframe(df_result, s3_bucket, f"{base_path}/report.{start_date}_{end_date}.csv")
        L.info("finish")

    def forward_test(self, predictor, s3_bucket, input_preprocess_base_path, input_simulate_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.forward_test")
        L.info(f"{self._job_name}.forward_test: start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{input_simulate_base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        for ticker_symbol in df_companies.index:
            result = self.forward_test_impl(ticker_symbol, predictor, s3_bucket, input_preprocess_base_path, input_simulate_base_path, output_base_path)

            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

        app_s3.write_dataframe(df_result, s3_bucket, f"{output_base_path}/companies.csv")
        L.info("finish")

    def forward_test_impl(self, ticker_symbol, predictor, s3_bucket, input_preprocess_base_path, input_simulate_base_path, output_base_path):
        L = get_app_logger(f"{self._job_name}.forward_test_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.forward_test_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        try:
            # Load data
            df_preprocess = app_s3.read_dataframe(s3_bucket, f"{input_preprocess_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
            df = app_s3.read_dataframe(s3_bucket, f"{input_simulate_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            # Predict
            df_data = df_preprocess.drop("date", axis=1).dropna()
            df_data = predictor.model_predict(ticker_symbol, df_data)

            df["predict"] = df_data["predict"]

            # Test
            for id in df.query("predict!=1").index:
                df.at[id, "buy_date"] = None
                df.at[id, "buy_price"] = None
                df.at[id, "sell_date"] = None
                df.at[id, "sell_price"] = None
                df.at[id, "profit"] = None
                df.at[id, "profit_rate"] = None

            # Save data
            app_s3.write_dataframe(df, s3_bucket, f"{output_base_path}/stock_prices.{ticker_symbol}.csv")
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def forward_test_report(self, start_date, end_date, s3_bucket, base_path):
        L = get_app_logger(f"{self._job_name}.forward_test_report")
        L.info(f"{self._job_name}.forward_test_report: start")

        df_companies = app_s3.read_dataframe(s3_bucket, f"{base_path}/companies.csv", index_col=0)
        df_result = pd.DataFrame(columns=df_companies.columns)

        results = joblib.Parallel(n_jobs=-1)([joblib.delayed(self.report_impl)(ticker_symbol, start_date, end_date, s3_bucket, base_path) for ticker_symbol in df_companies.index])

        for result in results:
            if result["exception"] is not None:
                continue

            ticker_symbol = result["ticker_symbol"]
            df_result.loc[ticker_symbol] = df_companies.loc[ticker_symbol]

            for k in result.keys():
                if k != "ticker_symbol" and k != "exception":
                    df_result.at[ticker_symbol, k] = result[k]

        L.info(df_result[["trade_count", "win_rate", "expected_value", "expected_value_win_only", "risk", "profit_factor"]].describe())

        app_s3.write_dataframe(df_result, s3_bucket, f"{base_path}/report.{start_date}_{end_date}.csv")
        L.info("finish")

    def report_impl(self, ticker_symbol, start_date, end_date, s3_bucket, base_path):
        L = get_app_logger(f"{self._job_name}.report_impl.{ticker_symbol}")
        L.info(f"{self._job_name}.report_impl: {ticker_symbol}")

        result = {
            "ticker_symbol": ticker_symbol,
            "exception": None
        }

        try:
            df = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0) \
                .query(f"'{start_date}' <= date < '{end_date}'")

            if len(df) == 0 or "profit" not in df.columns:
                raise Exception("no trade")

            result["trade_count"] = len(df.query("not profit.isnull()"))
            result["win_count"] = len(df.query("profit>0"))
            result["win_rate"] = result["win_count"] / result["trade_count"] if result["trade_count"] != 0 else None
            result["lose_count"] = len(df.query("profit<=0"))
            result["lose_rate"] = result["lose_count"] / result["trade_count"] if result["trade_count"] != 0 else None
            result["open_price_latest"] = df["open_price"].values[-1]
            result["high_price_latest"] = df["high_price"].values[-1]
            result["low_price_latest"] = df["low_price"].values[-1]
            result["close_price_latest"] = df["close_price"].values[-1]
            result["volume_average"] = df["volume"].mean()
            result["expected_value"] = df["profit_rate"].mean()
            result["expected_value_win_only"] = df.query("profit>0")["profit_rate"].mean()
            result["risk"] = df["profit_rate"].std()
            result["profit_total"] = df.query("profit>0")["profit"].sum()
            result["loss_total"] = df.query("profit<=0")["profit"].sum()
            if result["profit_total"] == 0 and result["loss_total"] == 0:
                result["profit_factor"] = None
            elif result["loss_total"] == 0:
                result["profit_factor"] = np.inf
            else:
                result["profit_factor"] = result["profit_total"] / abs(result["loss_total"])
            result["profit_average"] = df.query("profit>0")["profit"].mean()
            result["loss_average"] = df.query("profit<=0")["profit"].mean()
            if result["profit_average"] == 0 and result["loss_average"] == 0:
                result["payoff_ratio"] = None
            elif result["loss_average"] == 0:
                result["payoff_ratio"] = np.inf
            else:
                result["payoff_ratio"] = result["profit_average"] / abs(result["loss_average"])
            result["sharpe_ratio"] = result["expected_value"] / result["risk"] if result["risk"] != 0 else None
            result["profit_rate"] = (df["sell_price"].sum() - df["buy_price"].sum()) / df["sell_price"].sum()
            result["profit_rate_win_only"] = (df.query("profit>0")["sell_price"].sum() - df.query("profit>0")["buy_price"].sum()) / df.query("profit>0")["sell_price"].sum()
        except Exception as err:
            L.exception(f"ticker_symbol={ticker_symbol}, {err}")
            result["exception"] = err

        return result

    def forward_test_all(self, report_start_date, report_end_date, test_start_date, test_end_date, s3_bucket, base_path):
        L = get_app_logger(f"{self._job_name}.forward_test_all")
        L.info(f"{self._job_name}.forward_test_all: start")

        # Load data
        df_report = app_s3.read_dataframe(s3_bucket, f"{base_path}/report.{report_start_date}_{report_end_date}.csv", index_col=0)

        df_action = pd.DataFrame(columns=["ticker_symbol", "buy_date", "buy_price", "sell_date", "sell_price", "profit", "profit_rate"])

        for ticker_symbol in df_report.query("profit_factor>2.0").sort_values("expected_value", ascending=False).head(100).index:
            L.info(f"load data: {ticker_symbol}")
            df = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
            df = df[["ticker_symbol", "buy_date", "buy_price", "sell_date", "sell_price", "profit", "profit_rate"]].dropna()
            df_action = pd.concat([df_action, df])

        if len(df_action) == 0:
            for ticker_symbol in df_report.sort_values("expected_value", ascending=False).head(100).index:
                L.info(f"load data: {ticker_symbol}")
                df = app_s3.read_dataframe(s3_bucket, f"{base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)
                df = df[["ticker_symbol", "buy_date", "buy_price", "sell_date", "sell_price", "profit", "profit_rate"]].dropna()
                df_action = pd.concat([df_action, df])

        # Collect trade data
        df_action = df_action.sort_values("buy_date") \
            .query(f"'{test_start_date}'<=buy_date<'{test_end_date}' and '{test_start_date}'<=sell_date<'{test_end_date}'")
        df_action["id"] = range(len(df_action))
        df_action = df_action.set_index("id")

        for ticker_symbol in df_action["ticker_symbol"].unique():
            expected_value = df_report.at[ticker_symbol, "expected_value"]

            for id in df_action.query(f"ticker_symbol=={ticker_symbol}").index:
                df_action.at[id, "expected_value"] = expected_value

        app_s3.write_dataframe(df_action, s3_bucket, f"{base_path}/test_all.action.csv")

        # Init
        df_action_single = pd.DataFrame(columns=["date", "ticker_symbol", "action", "price", "stocks", "profit", "profit_rate", "fee", "tax"])
        df_action_compound = pd.DataFrame(columns=["date", "ticker_symbol", "action", "price", "stocks", "profit", "profit_rate", "fee", "tax"])
        df_stocks_single = pd.DataFrame(columns=["ticker_symbol", "buy_price", "buy_stocks", "sell_date", "sell_price", "profit", "profit_rate"])
        df_stocks_compound = pd.DataFrame(columns=["ticker_symbol", "buy_price", "buy_stocks", "sell_date", "sell_price", "profit", "profit_rate"])
        df_result = pd.DataFrame(columns=["fund_single", "asset_single", "fund_single_cost", "asset_single_cost", "fund_compound", "asset_compound"])

        init_asset = 1000000
        fund_single = init_asset
        fund_single_cost = init_asset
        fund_compound = init_asset
        asset_single = init_asset
        asset_single_cost = init_asset
        asset_compound = init_asset
        available_rate = 0.05
        fee_rate = 0.003
        tax_rate = 0.21

        for date in self.date_range(test_start_date, test_end_date):
            L.info(f"{self._job_name}.forward_test_all: {date}")

            # Buy
            for action_id in df_action.query(f"buy_date=='{date}'").sort_values("expected_value", ascending=False).index:
                ticker_symbol = df_action.at[action_id, "ticker_symbol"]
                buy_price = df_action.at[action_id, "buy_price"]
                sell_date = df_action.at[action_id, "sell_date"]
                sell_price = df_action.at[action_id, "sell_price"]
                profit = df_action.at[action_id, "profit"]
                profit_rate = df_action.at[action_id, "profit_rate"]

                buy_stocks_single = init_asset * available_rate // buy_price
                fee_single = buy_price * buy_stocks_single * fee_rate
                if buy_stocks_single > 0 and (fund_single_cost - buy_price * buy_stocks_single - fee_single) > 0:
                    df_action_single.loc[len(df_action_single)] = [date, ticker_symbol, "buy", buy_price,  buy_stocks_single, None, None, fee_single, None]
                    df_stocks_single.loc[len(df_stocks_single)] = [ticker_symbol, buy_price, buy_stocks_single, sell_date, sell_price, profit, profit_rate]

                    fund_single = fund_single - buy_price * buy_stocks_single
                    fund_single_cost = fund_single_cost - buy_price * buy_stocks_single - fee_single

                buy_stocks_compound = asset_compound * available_rate // buy_price
                fee_compound = buy_price * buy_stocks_compound * fee_rate
                if buy_stocks_compound > 0 and (fund_compound - buy_price * buy_stocks_compound - fee_compound) > 0:
                    df_action_compound.loc[len(df_action_compound)] = [date, ticker_symbol, "buy", buy_price, buy_stocks_compound, None, None, fee_compound, None]
                    df_stocks_compound.loc[len(df_stocks_compound)] = [ticker_symbol, buy_price, buy_stocks_compound, sell_date, sell_price, profit, profit_rate]

                    fund_compound = fund_compound - buy_price * buy_stocks_compound - fee_compound

            # Sell
            for stocks_id in df_stocks_single.index:
                if df_stocks_single.at[stocks_id, "sell_date"] != date:
                    continue

                ticker_symbol = df_stocks_single.at[stocks_id, "ticker_symbol"]
                buy_stocks = df_stocks_single.at[stocks_id, "buy_stocks"]
                sell_price = df_stocks_single.at[stocks_id, "sell_price"]
                profit = df_stocks_single.at[stocks_id, "profit"]
                profit_rate = df_stocks_single.at[stocks_id, "profit_rate"]

                fee = sell_price * buy_stocks * fee_rate
                tax = profit * buy_stocks * tax_rate if profit > 0 else 0

                df_action_single.loc[len(df_action_single)] = [date, ticker_symbol, "sell", sell_price, buy_stocks, profit * buy_stocks, profit_rate, fee, tax]
                df_stocks_single = df_stocks_single.drop(stocks_id)

                fund_single = fund_single + sell_price * buy_stocks
                fund_single_cost = fund_single_cost + sell_price * buy_stocks - fee - tax

            df_stocks_single["id"] = range(len(df_stocks_single))
            df_stocks_single = df_stocks_single.set_index("id")

            for stocks_id in df_stocks_compound.index:
                if df_stocks_compound.at[stocks_id, "sell_date"] != date:
                    continue

                ticker_symbol = df_stocks_compound.at[stocks_id, "ticker_symbol"]
                buy_stocks = df_stocks_compound.at[stocks_id, "buy_stocks"]
                sell_price = df_stocks_compound.at[stocks_id, "sell_price"]
                profit = df_stocks_compound.at[stocks_id, "profit"]
                profit_rate = df_stocks_compound.at[stocks_id, "profit_rate"]

                fee = sell_price * buy_stocks * fee_rate
                tax = profit * buy_stocks * tax_rate if profit > 0 else 0

                df_action_compound.loc[len(df_action_compound)] = [date, ticker_symbol, "sell", sell_price, buy_stocks, profit * buy_stocks, profit_rate, fee, tax]
                df_stocks_compound = df_stocks_compound.drop(stocks_id)

                fund_compound = fund_compound + sell_price * buy_stocks - fee - tax

            df_stocks_compound["id"] = range(len(df_stocks_compound))
            df_stocks_compound = df_stocks_compound.set_index("id")

            # Turn end
            asset_single = fund_single + (df_stocks_single["buy_price"] * df_stocks_single["buy_stocks"]).sum()
            asset_single_cost = fund_single_cost + (df_stocks_single["buy_price"] * df_stocks_single["buy_stocks"]).sum()
            asset_compound = fund_compound + (df_stocks_compound["buy_price"] * df_stocks_compound["buy_stocks"]).sum()

            df_result.loc[date] = [fund_single, asset_single, fund_single_cost, asset_single_cost, fund_compound, asset_compound]

            L.info(df_result.loc[date])

        app_s3.write_dataframe(df_action_single, s3_bucket, f"{base_path}/test_all.action.single.csv")
        app_s3.write_dataframe(df_action_compound, s3_bucket, f"{base_path}/test_all.action.compound.csv")
        app_s3.write_dataframe(df_result, s3_bucket, f"{base_path}/test_all.result.csv")

        L.info("finish")

    def date_range(self, start, end):
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")

        for n in range((e - s).days):
            yield (s + timedelta(n)).strftime("%Y-%m-%d")
