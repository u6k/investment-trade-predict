import argparse
import os
from comet_ml import Experiment
from investment_stocks_predict_trend import select_company
from investment_stocks_predict_trend import random_forest_1
from investment_stocks_predict_trend import random_forest_2
from investment_stocks_predict_trend import agent_1
from investment_stocks_predict_trend import agent_2
from investment_stocks_predict_trend import agent_3
from investment_stocks_predict_trend import agent_4
from investment_stocks_predict_trend import agent_5
from investment_stocks_predict_trend import agent_6
from investment_stocks_predict_trend import agent_7
from investment_stocks_predict_trend import agent_8
from investment_stocks_predict_trend import agent_9


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand")

    args = parser.parse_args()

    if args.subcommand == "select_company.preprocessing":
        select_company.preprocessing()
    elif args.subcommand == "select_company.top":
        select_company.top()
    elif args.subcommand == "random_forest_1.scores":
        random_forest_1.scores()
    elif args.subcommand == "random_forest_2.scores":
        random_forest_2.scores()
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
    else:
        raise Exception("unknown subcommand: " + args.subcommand)
