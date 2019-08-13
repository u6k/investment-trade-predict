import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from app_logging import get_app_logger
import app_s3


def execute(s3_bucket, cluster_start_date, cluster_end_date, prices_start_date, prices_end_date, clusters, input_base_path, output_base_path):
    L = get_app_logger()
    L.info(f"start: cluster_start_date={cluster_start_date}, cluster_end_date={cluster_end_date}, prices_start_date={prices_start_date}, prices_end_date={prices_end_date}, clusters={clusters}")

    # Load data
    df_companies = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/companies.csv", index_col=0)

    df_adj = pd.DataFrame()

    for ticker_symbol in df_companies.index:
        L.info(f"Load all adjusted close price: ticker_symbol={ticker_symbol}")

        df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

        if len(df.query(f"date<'{cluster_start_date}'")) == 0 or len(df.query(f"date>='{cluster_end_date}'")) == 0:
            L.info("  skip")
            continue

        df = df.set_index("date")
        df_adj = pd.concat([df_adj, df["adjusted_close_price"]], axis=1).rename(columns={"adjusted_close_price": ticker_symbol})

    df_adj = df_adj.dropna().query(f"'{cluster_start_date}'<=index<'{cluster_end_date}'")

    app_s3.write_dataframe(df_adj, s3_bucket, f"{output_base_path}/adjusted_close_prices.all.csv")

    L.info(f"All adjusted close prices: start date={df_adj.index.min()}")
    L.info(f"All adjusted close prices: end date={df_adj.index.max()}")
    L.info(f"All adjusted close prices: records={len(df_adj)}")
    L.info(f"All adjusted close prices: columns={len(df_adj.columns)}")

    # Calc correlation coefficient
    L.info("Calc correlation cofficient")

    df_corr = df_adj.corr()
    app_s3.write_dataframe(df_corr, s3_bucket, f"{output_base_path}/corr.all.csv")

    # Analyze cluster
    L.info("Analyze cluster")

    df_adj = df_adj.T
    clf = KMeans(n_clusters=clusters, random_state=0, n_jobs=-1).fit(df_adj)

    L.info(clf)

    df_adj["cluster"] = clf.labels_
    df_companies["cluster"] = df_adj["cluster"].apply(lambda c: f"cluster_{c}")
    df_companies = df_companies.dropna()

    L.info(f"clusters={df_companies['cluster'].value_counts()}")

    app_s3.write_dataframe(df_adj, s3_bucket, f"{output_base_path}/adjusted_close_prices.clusters.csv")
    app_s3.write_dataframe(df_companies, s3_bucket, f"{output_base_path}/companies.clusters.csv")

    # Generate virtual company
    L.info("Generate virtual company")

    df_companies_cluster = pd.DataFrame()

    for cluster in df_companies["cluster"].unique():
        # Load data
        df_adj_open = pd.DataFrame()
        df_adj_high = pd.DataFrame()
        df_adj_low = pd.DataFrame()
        df_adj_close = pd.DataFrame()

        for ticker_symbol in df_companies.query(f"cluster=='{cluster}'").index:
            L.info(f"Load adjusted price: cluster={cluster}, ticker_symbol={ticker_symbol}")

            df = app_s3.read_dataframe(s3_bucket, f"{input_base_path}/stock_prices.{ticker_symbol}.csv", index_col=0)

            if len(df.query(f"date<'{prices_start_date}'")) == 0 or len(df.query(f"date>='{prices_end_date}'")) == 0:
                L.info("  skip")
                continue

            df = df.set_index("date")
            df_adj_open = pd.concat([df_adj_open, df["adjusted_open_price"]], axis=1).rename(columns={"adjusted_open_price": ticker_symbol})
            df_adj_high = pd.concat([df_adj_high, df["adjusted_high_price"]], axis=1).rename(columns={"adjusted_high_price": ticker_symbol})
            df_adj_low = pd.concat([df_adj_low, df["adjusted_low_price"]], axis=1).rename(columns={"adjusted_low_price": ticker_symbol})
            df_adj_close = pd.concat([df_adj_close, df["adjusted_close_price"]], axis=1).rename(columns={"adjusted_close_price": ticker_symbol})

        # OHLC PCA
        L.info(f"Calc ohlc pca: cluster={cluster}")

        df_adj_open = df_adj_open.dropna().query(f"'{prices_start_date}'<=index<'{prices_end_date}'")
        df_adj_high = df_adj_high.dropna().query(f"'{prices_start_date}'<=index<'{prices_end_date}'")
        df_adj_low = df_adj_low.dropna().query(f"'{prices_start_date}'<=index<'{prices_end_date}'")
        df_adj_close = df_adj_close.dropna().query(f"'{prices_start_date}'<=index<'{prices_end_date}'")

        app_s3.write_dataframe(df_adj_open, s3_bucket, f"{output_base_path}/adjusted_open_prices.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_high, s3_bucket, f"{output_base_path}/adjusted_high_prices.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_low, s3_bucket, f"{output_base_path}/adjusted_low_prices.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_close, s3_bucket, f"{output_base_path}/adjusted_close_prices.cluster_{cluster}.csv")

        clf_adj_open = PCA().fit(df_adj_open)
        clf_adj_high = PCA().fit(df_adj_high)
        clf_adj_low = PCA().fit(df_adj_low)
        clf_adj_close = PCA().fit(df_adj_close)

        df_adj_open_pc = pd.DataFrame(clf_adj_open.transform(df_adj_open), columns=[f"PC_{i}" for i in range(len(df_adj_open.columns))])
        df_adj_high_pc = pd.DataFrame(clf_adj_high.transform(df_adj_high), columns=[f"PC_{i}" for i in range(len(df_adj_high.columns))])
        df_adj_low_pc = pd.DataFrame(clf_adj_low.transform(df_adj_low), columns=[f"PC_{i}" for i in range(len(df_adj_low.columns))])
        df_adj_close_pc = pd.DataFrame(clf_adj_close.transform(df_adj_close), columns=[f"PC_{i}" for i in range(len(df_adj_close.columns))])

        L.info(f"adjusted open prices PC: {df_adj_open_pc}")
        L.info(f"adjusted high prices PC: {df_adj_high_pc}")
        L.info(f"adjusted low prices PC: {df_adj_low_pc}")
        L.info(f"adjusted close prices PC: {df_adj_close_pc}")

        app_s3.write_dataframe(df_adj_open_pc, s3_bucket, f"{output_base_path}/adjusted_open_prices_pc.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_high_pc, s3_bucket, f"{output_base_path}/adjusted_high_prices_pc.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_low_pc, s3_bucket, f"{output_base_path}/adjusted_low_prices_pc.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_close_pc, s3_bucket, f"{output_base_path}/adjusted_close_prices_pc.cluster_{cluster}.csv")

        df_adj_open_evr = pd.DataFrame(clf_adj_open.explained_variance_ratio_)
        df_adj_high_evr = pd.DataFrame(clf_adj_high.explained_variance_ratio_)
        df_adj_low_evr = pd.DataFrame(clf_adj_low.explained_variance_ratio_)
        df_adj_close_evr = pd.DataFrame(clf_adj_close.explained_variance_ratio_)

        L.info(f"adjusted open prices explained variance ratio: {df_adj_open_evr}")
        L.info(f"adjusted high prices explained variance ratio: {df_adj_high_evr}")
        L.info(f"adjusted low prices explained variance ratio: {df_adj_low_evr}")
        L.info(f"adjusted close prices explained variance ratio: {df_adj_close_evr}")

        app_s3.write_dataframe(df_adj_open_evr, s3_bucket, f"{output_base_path}/adjusted_open_prices_evr.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_high_evr, s3_bucket, f"{output_base_path}/adjusted_high_prices_evr.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_low_evr, s3_bucket, f"{output_base_path}/adjusted_low_prices_evr.cluster_{cluster}.csv")
        app_s3.write_dataframe(df_adj_close_evr, s3_bucket, f"{output_base_path}/adjusted_close_prices_evr.cluster_{cluster}.csv")

        df_cluster_prices = df_adj_open.copy()
        df_cluster_prices = df_cluster_prices.drop(df_cluster_prices.columns, axis=1)
        df_cluster_prices["date"] = df_cluster_prices.index
        df_cluster_prices["id"] = range(len(df_cluster_prices))
        df_cluster_prices = df_cluster_prices.set_index("id")
        df_cluster_prices["ticker_symbol"] = cluster

        scaler = MinMaxScaler(feature_range=(0.1, 1.0))
        scaler.fit(np.concatenate([df_adj_open_pc["PC_0"].values, df_adj_high_pc["PC_0"].values, df_adj_low_pc["PC_0"].values, df_adj_close_pc["PC_0"].values]).reshape(-1, 1))

        df_cluster_prices["open_price"] = scaler.transform(df_adj_open_pc["PC_0"].values.reshape(-1, 1))
        df_cluster_prices["high_price"] = scaler.transform(df_adj_high_pc["PC_0"].values.reshape(-1, 1))
        df_cluster_prices["low_price"] = scaler.transform(df_adj_low_pc["PC_0"].values.reshape(-1, 1))
        df_cluster_prices["close_price"] = scaler.transform(df_adj_close_pc["PC_0"].values.reshape(-1, 1))
        df_cluster_prices["volume"] = 0
        df_cluster_prices["adjusted_open_price"] = df_cluster_prices["open_price"]
        df_cluster_prices["adjusted_high_price"] = df_cluster_prices["high_price"]
        df_cluster_prices["adjusted_low_price"] = df_cluster_prices["low_price"]
        df_cluster_prices["adjusted_close_price"] = df_cluster_prices["close_price"]

        L.info(f"cluster={cluster}: prices: {df_cluster_prices}")

        app_s3.write_dataframe(df_cluster_prices, s3_bucket, f"{output_base_path}/stock_prices.{cluster}.csv")

        df_companies_cluster.at[cluster, "name"] = cluster

    app_s3.write_dataframe(df_companies_cluster, s3_bucket, f"{output_base_path}/companies.csv")

    L.info("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", help="folder name suffix (default: test)", default="test")
    args = parser.parse_args()

    execute(
        s3_bucket="u6k",
        cluster_start_date="2018-01-01",
        cluster_end_date="2019-01-01",
        prices_start_date="2008-01-01",
        prices_end_date="2019-01-01",
        clusters=10,
        input_base_path=f"ml-data/stocks/preprocess_1.{args.suffix}",
        output_base_path=f"ml-data/stocks/preprocess_4.{args.suffix}.clusters_10.2018",
    )
