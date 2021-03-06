import argparse

from sklearn.linear_model import Lasso
from predict_base import PredictRegressionBase


class PredictRegression_4(PredictRegressionBase):
    def model_fit(self, x_train, y_train):
        return Lasso().fit(x_train, y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="preprocess, or train")
    parser.add_argument("--simulate-group", help="simulate trade group")
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    pred = PredictRegression_4(
        train_start_date="2008-01-01",
        train_end_date="2017-12-31",
        test_start_date="2018-01-01",
        test_end_date="2018-12-31",
        s3_bucket="u6k",
        input_preprocess_base_path=f"ml-data/stocks/preprocess_2.{args.suffix}",
        input_simulate_base_path=f"ml-data/stocks/simulate_trade_{args.simulate_group}.{args.suffix}",
        output_base_path=f"ml-data/stocks/predict_4.simulate_trade_{args.simulate_group}.{args.suffix}"
    )

    if args.task == "preprocess":
        pred.preprocess()
    elif args.task == "train":
        pred.train()
    else:
        parser.print_help()
