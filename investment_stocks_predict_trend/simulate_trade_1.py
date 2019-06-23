def execute(df_prices_preprocessed, sma_short_len, sma_long_len, losscut_rate=0.95):
    df = df_prices_preprocessed.copy()

    # simulate
    buy_price = 0.0
    losscut_price = 0.0

    for current_id in df.index[sma_long_len:]:
        sma_short_1 = df.at[current_id-1, f"sma_{sma_short_len}"]
        sma_long_1 = df.at[current_id-1, f"sma_{sma_long_len}"]
        sma_short_2 = df.at[current_id-2, f"sma_{sma_short_len}"]
        sma_long_2 = df.at[current_id-2, f"sma_{sma_long_len}"]

        if (buy_price > 0) and (losscut_price > df.at[current_id-1, "close_price"]):
            # loss cut
            df.at[current_id, "profit"] = df.at[current_id, "close_price"] - buy_price
            df.at[current_id, "action"] = "losscut"

            buy_price = 0.0
            losscut_price = 0.0
        elif (buy_price > 0) and (sma_short_2 > sma_long_2) and (sma_short_1 <= sma_long_1):
            # sell
            df.at[current_id, "profit"] = df.at[current_id, "close_price"] - buy_price
            df.at[current_id, "action"] = "sell"

            buy_price = 0.0
            losscut_price = 0.0
        elif (buy_price == 0) and (sma_short_2 < sma_long_2) and (sma_short_1 >= sma_long_1):
            # buy
            df.at[current_id, "action"] = "buy"

            buy_price = df.at[current_id, "close_price"]
            losscut_price = buy_price * losscut_rate
        else:
            # stay
            df.at[current_id, "action"] = ""

            if losscut_price < (df.at[current_id, "close_price"] * losscut_rate):
                losscut_price = df.at[current_id, "close_price"] * losscut_rate

        df.at[current_id, "buy_price"] = buy_price
        df.at[current_id, "losscut_price"] = losscut_price

    return df
