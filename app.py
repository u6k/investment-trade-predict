import argparse
import os
from comet_ml import Experiment
from investment_stocks_predict_trend import select_company
from investment_stocks_predict_trend import agent_1
from investment_stocks_predict_trend import agent_2
from investment_stocks_predict_trend import agent_3
from investment_stocks_predict_trend import agent_4
from investment_stocks_predict_trend import agent_5
from investment_stocks_predict_trend import agent_6
from investment_stocks_predict_trend import agent_7
from investment_stocks_predict_trend import agent_8
from investment_stocks_predict_trend import agent_9
from investment_stocks_predict_trend import agent_10
from investment_stocks_predict_trend import agent_11
from investment_stocks_predict_trend import agent_12
from investment_stocks_predict_trend import predict_1
from investment_stocks_predict_trend import predict_2
from investment_stocks_predict_trend.predict_3 import PredictClassification_3
from investment_stocks_predict_trend.predict_4 import PredictRegression_4
from investment_stocks_predict_trend.predict_5 import PredictClassification_5
from investment_stocks_predict_trend import backtest_1
from investment_stocks_predict_trend import backtest_2
from investment_stocks_predict_trend import backtest_3
from investment_stocks_predict_trend import backtest_4
from investment_stocks_predict_trend import backtest_5
from investment_stocks_predict_trend import preprocess_1
from investment_stocks_predict_trend import preprocess_2
from investment_stocks_predict_trend import simulate_trade_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand")

    args = parser.parse_args()

    if args.subcommand == "select_company.export_stock_prices":
        select_company.export_stock_prices()
    elif args.subcommand == "select_company.analysis":
        select_company.analysis()
    elif args.subcommand == "select_company.analysis_2":
        select_company.analysis_2()
    elif args.subcommand == "agent_1":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_1")
        agent_1.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_2":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_2")
        agent_2.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_3":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_3")
        agent_3.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_4":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_4")
        agent_4.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_5":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_5")
        agent_5.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_6":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_6")
        agent_6.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_7":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_7")
        agent_7.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_8":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_8")
        agent_8.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_9":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_9")
        agent_9.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_10":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_10")
        agent_10.execute(experiment, max_episode=500)
        experiment.end()
    elif args.subcommand == "agent_11":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_11")
        agent_11.execute(experiment)
        experiment.end()
    elif args.subcommand == "agent_12":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="agent_12")
        agent_12.execute(experiment)
        experiment.end()
    elif args.subcommand == "predict_1":
        predict_1.execute()
    elif args.subcommand == "predict_2":
        experiment = Experiment(api_key=os.environ["COMET_ML_API_KEY"], project_name="predict_2")
        predict_2.execute(experiment)
        experiment.end()
    elif args.subcommand == "predict_3":
        PredictClassification_3().execute()
    elif args.subcommand == "predict_4":
        PredictRegression_4().execute()
    elif args.subcommand == "predict_5":
        PredictClassification_5().execute()
    elif args.subcommand == "preprocess_1":
        preprocess_1.execute()
    elif args.subcommand == "preprocess_2":
        preprocess_2.execute()
    elif args.subcommand == "backtest_1":
        backtest_1.execute()
    elif args.subcommand == "backtest_1.single":
        backtest_1.execute_single()
    elif args.subcommand == "backtest_2":
        backtest_2.execute()
    elif args.subcommand == "backtest_3":
        backtest_3.execute()
    elif args.subcommand == "backtest_3.2":
        backtest_3.execute_2()
    elif args.subcommand == "backtest_3.simulate_trade":
        df_stocks = backtest_3.simulate_trade("1301", "2017-07-01", "2018-06-30", 5, 20)
        df_stocks.to_csv("local/test_3/result.1301.2018.csv")
    elif args.subcommand == "backtest_3.report":
        backtest_3.report()
    elif args.subcommand == "backtest_3.report_2":
        backtest_3.report_2()
    elif args.subcommand == "backtest_3.train_profit_rate":
        backtest_3.train_profit_rate()
    elif args.subcommand == "backtest_4":
        backtest_4.execute()
    elif args.subcommand == "backtest_5.preprocess":
        backtest_5.preprocess()
    elif args.subcommand == "backtest_5.backtest":
        backtest_5.backtest()
    elif args.subcommand == "backtest_5.backtest_single":
        backtest_5.backtest_single()
    elif args.subcommand == "simulate_trade_2":
        simulate_trade_2.execute()
    else:
        raise Exception("unknown subcommand: " + args.subcommand)
