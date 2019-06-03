import argparse

from investment_stocks_predict_trend import select_company
from investment_stocks_predict_trend import random_forest_1
from investment_stocks_predict_trend import random_forest_2
from investment_stocks_predict_trend import agent_1

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
        # FIXME
        df = agent_1.preprocessing()
    else:
        raise Exception("unknown subcommand: " + args.subcommand)
