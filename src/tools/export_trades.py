import pandas as pd
from pathlib import Path

IN = Path("data/ohlcv_7203.csv")
OUT = Path("backtests/trades_log.csv")


def run_backtest(df: pd.DataFrame):
    df = df.copy()

    # 日付
    df["date"] = pd.to_datetime(df["date"])

    # 移動平均
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()

    position = 0
    entry_price = 0
    entry_date = None

    trades = []

    for i in range(len(df)):
        row = df.iloc[i]

        if pd.isna(row["ma5"]) or pd.isna(row["ma20"]):
            continue

        # ENTRY
        if position == 0 and row["ma5"] > row["ma20"]:
            position = 1
            entry_price = row["close"]
            entry_date = row["date"]

        # EXIT
        elif position == 1 and row["ma5"] < row["ma20"]:
            exit_price = row["close"]
            exit_date = row["date"]

            pnl = exit_price - entry_price
            hold_days = (exit_date - entry_date).days

            trades.append({
                "date": entry_date.strftime("%Y-%m-%d"),
                "symbol": "7203",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "hold_days": hold_days,
            })

            position = 0

    return trades


def main():
    if not IN.exists():
        raise FileNotFoundError(f"{IN} not found")

    df = pd.read_csv(IN)

    trades = run_backtest(df)

    if not trades:
        print("tradesなし")
        return

    df_new = pd.DataFrame(trades)

    if OUT.exists():
        df_old = pd.read_csv(OUT)
        df_new = pd.concat([df_old, df_new], ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(OUT, index=False)

    print(f"saved: {len(df_new)} rows")


if __name__ == "__main__":
    main()