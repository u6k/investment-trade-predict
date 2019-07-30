import argparse

from sklearn import ensemble
# from sklearn import ensemble, model_selection
from predict_base import PredictClassificationBase


class PredictClassification_3(PredictClassificationBase):
    def model_fit(self, x_train, y_train):
        return ensemble.RandomForestClassifier(n_estimators=500, criterion="entropy", max_depth=8).fit(x_train, y_train)
        # parameters = {
        #    "n_estimators": [100, 200, 500, 750, 1000],
        #    "criterion": ["gini", "entropy"],
        #    "max_depth": [8, 16, 64],
        # }

        # clf = model_selection.GridSearchCV(
        #    ensemble.RandomForestClassifier(),
        #    parameters,
        #    cv=5,
        #    n_jobs=-1,
        #    verbose=1
        # )

        # clf.fit(x_train, y_train)

        # print(f"best_params: {clf.best_params_}")

        # clf_best = clf.best_estimator_

        # return clf_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-group", help="simulate trade group")
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    pred = PredictClassification_3(
        job_name="predict_3",
        train_start_date="2008-01-01",
        train_end_date="2018-01-01",
        test_start_date="2018-01-01",
        test_end_date="2019-01-01",
        s3_bucket="u6k",
        input_preprocess_base_path=f"ml-data/stocks/preprocess_3.{args.suffix}",
        input_simulate_base_path=f"ml-data/stocks/simulate_trade_{args.simulate_group}.{args.suffix}",
        output_base_path=f"ml-data/stocks/predict_3.simulate_trade_{args.simulate_group}.{args.suffix}"
    )

    pred.train()
