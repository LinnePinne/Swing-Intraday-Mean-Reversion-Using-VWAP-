import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from math import sqrt
import os

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================

markets = [
    {
        "name": "EURGBP",
        "csv": "EURGBP_1H_2012-2026.csv",
        "pip_size": 0.0001,
        # USDJPY: 0.01, NZDUSD och AUDUSD och USDCHF och USDCAD och EURUSD och GBPUSD och USDCAD: 0.0001 pip
        "spread_points_per_pip": 10.0,  # om spread_points är pipetter (vanligast). Om er kolumn redan är pips: sätt 1.0
    },
]

# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5

SLIPPAGE_PIPS = 0.12  # NZDUSD: 0.00, USDCAD: 0.10, AUDUSD och USDCHF och USDJPY och GBPUSD: 0.08, EURUSD:  0.05 pip (0.5 pipette)
FIXED_SPREAD_PIPS = 0.25  # NZDUSD: 1.2, AUDUSD och USDCHF: 0.18, USDCAD: 0.22, USDJPY och GBPUSD: 0.15, EURUSD: 0.10 pip (1 pipette) - anpassa!
COMM_PIPS_PER_SIDE = 0.25  # AUDUSD och USDCHF och USDCAD och USDJPY och GBPUSD och EURUSD: 0.02 pip per sida - exempel


def compute_stats_from_trades(trades_df: pd.DataFrame) -> dict:
    """
    Samma logik som era totala stats, men på en subset av trades.
    trades_df måste ha kolumner: pnl, equity (valfritt), is_win (valfritt)
    """
    if trades_df.empty:
        return {}

    df_ = trades_df.copy()

    # equity behövs för DD
    df_["equity"] = df_["pnl"].cumsum()

    df_["is_win"] = df_["pnl"] > 0

    gross_profit = df_.loc[df_["pnl"] > 0, "pnl"].sum()
    gross_loss = df_.loc[df_["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = df_.loc[df_["pnl"] > 0, "pnl"].mean()
    avg_loss = df_.loc[df_["pnl"] < 0, "pnl"].mean()

    winrate = df_["is_win"].mean()
    expectancy = df_["pnl"].mean()

    roll_max = df_["equity"].cummax()
    dd = df_["equity"] - roll_max
    max_dd_points = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Losing streak
    loss_streak = 0
    max_loss_streak = 0
    for is_win in df_["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    pnl_std = df_["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(df_)) if pnl_std and pnl_std > 0 else np.nan

    return {
        "Trades": int(len(df_)),
        "Total PnL (points)": float(df_["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
    }


def run_backtest_for_market(market_name: str, csv_path: str, pip_size: float, spread_points_per_pip: float = 10.0):
    print("\n" + "=" * 70)
    print(f" BACKTEST FÖR MARKNAD: {market_name} ")
    print("=" * 70 + "\n")

    # 1) Ladda data
    df = pd.read_csv(csv_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = df.sort_index()
    # ---- DEDUPE INDEX (viktigt för groupby/transform) ----
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].sort_index()

    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla kolumnerna: {required_cols}")

    def pips_to_price(pips: float) -> float:
        return float(pips) * float(pip_size)

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(COMM_PIPS_PER_SIDE)

    # volymkolumn
    if "volume" in df.columns:
        vol_col = "volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        df["volume_dummy"] = 1.0
        vol_col = "volume_dummy"

    def compute_session_anchored_vwap_and_std(data: pd.DataFrame, vol_col: str, reset_time: str = "20:00:00"):
        df_v = data.copy()

        tp = (df_v["high"] + df_v["low"] + df_v["close"]) / 3.0
        vol = df_v[vol_col].astype(float).fillna(0.0)

        df_v["tp"] = tp
        df_v["vol"] = vol
        df_v["tp_vol"] = df_v["tp"] * df_v["vol"]

        rt = pd.to_datetime(reset_time).time()

        session_date = df_v.index.floor("D")
        session_date = session_date.where(df_v.index.time >= rt, session_date - pd.Timedelta(days=1))
        df_v["session_date"] = session_date

        g = df_v.groupby("session_date", sort=False)

        df_v["cum_tp_vol"] = g["tp_vol"].cumsum()
        df_v["cum_vol"] = g["vol"].cumsum()

        vwap = df_v["cum_tp_vol"] / df_v["cum_vol"].replace(0.0, np.nan)

        # transform är OK när index är unikt (därför dedupe före denna funktion)
        std = g["tp"].transform(lambda x: x.expanding().std(ddof=0))

        return vwap, std

    df["VWAP"], df["TP_STD"] = compute_session_anchored_vwap_and_std(df, vol_col, "20:00:00")


    # session (svensk tid)
    session_start = "00:00:00"
    session_end = "07:00:00"

    session_start_t = pd.to_datetime(session_start).time()
    session_end_t = pd.to_datetime(session_end).time()

    def in_session(ts) -> bool:
        t = ts.time()

        # Normal session (ex 01:00 -> 07:00)
        if session_start_t < session_end_t:
            return (t >= session_start_t) and (t < session_end_t)

        # Wrap session (ex 23:00 -> 03:00)
        # Då är det "i session" om tiden är >= start ELLER < end
        return (t >= session_start_t) or (t < session_end_t)

    USE_SPREAD_PIPS_COL = 'spread_pips' in df.columns
    USE_SPREAD_POINTS_COL = 'spread_points' in df.columns

    def get_spread_pips(row) -> float:
        if USE_SPREAD_PIPS_COL:
            return float(row['spread_pips'])
        if USE_SPREAD_POINTS_COL:
            return float(row['spread_points']) / float(spread_points_per_pip)
        return float(FIXED_SPREAD_PIPS)

    std_mult = 2.25 # hur många std från VWAP krävs

    # 4) Backtest-loop
    trades = []
    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):

        ts = idx_list[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        # ======================
        # EXIT-logik (signal på bar i, fill på bar i+1 open)
        # ======================
        if in_position:
            exit_price = None
            exit_reason = None

            vwap = row["VWAP"]
            std = row["TP_STD"]

            if pos_direction == 'LONG':
                final_level = vwap - 0.75 * std
                if row["close"] >= final_level: #row["VWAP"]:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(SLIPPAGE_PIPS)

                    exit_price = next_row["open"] - HALF * spread_px - slip_px
                    exit_reason = 'vwap_exit'

            else:  # SHORT
                final_level = vwap + 0.75 * std
                if row["close"] <= final_level: #row["VWAP"]:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(SLIPPAGE_PIPS)

                    exit_price = next_row["open"] + HALF * spread_px + slip_px

            if exit_price is not None:
                exit_time = idx_list[i + 1]  # matchar fill på next_row open
                comm_px = commission_round_turn_price()

                if pos_direction == 'LONG':
                    pnl = (exit_price - entry_price) - comm_px
                else:
                    pnl = (entry_price - exit_price) - comm_px

                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Direction': pos_direction,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Exit Reason': exit_reason,
                    'pnl': pnl,
                })

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None

                continue  # viktigt: undvik att gå in i ny trade samma iteration

        # Sessionfilter: både signalbar och fillbar måste vara i session
        if not in_session(ts) or not in_session(idx_list[i + 1]):
            continue

        # hoppa entry-logik om vi fortfarande är i trade
        if in_position:
            continue
        # ======================
        # ENTRY-logik
        # ======================
        close_price = row["close"]
        prev_close = prev_row["close"]
        next_open = next_row['open']
        vwap = row["VWAP"]
        std = row["TP_STD"]

        if not np.isfinite(vwap) or not np.isfinite(std) or std == 0:
            continue

        # STD-bands kring VWAP
        upper_band = vwap + std_mult * std
        lower_band = vwap - std_mult * std
        prev_upper_band = prev_row["VWAP"] + std_mult * prev_row["TP_STD"]
        prev_lower_band = prev_row["VWAP"] - std_mult * prev_row["TP_STD"]

        upper_band_break = prev_close < prev_upper_band and close_price > upper_band
        lower_band_break = prev_close > prev_lower_band and close_price < lower_band

        long_signal = lower_band_break
        short_signal = upper_band_break

        if long_signal:
            pos_direction = 'LONG'
            entry_time = idx_list[i + 1]  # fill sker på nästa bar open

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(SLIPPAGE_PIPS)

            entry_price = next_open + HALF * spread_px + slip_px
            in_position = True

        elif short_signal:
            pos_direction = 'SHORT'
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(SLIPPAGE_PIPS)

            entry_price = next_open - HALF * spread_px - slip_px
            in_position = True

    # ==========================
    # 5. Resultatsammanställning
    # ==========================
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("Inga trades hittades.")
        return None, trades_df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)
    trades_df["equity"] = trades_df["pnl"].cumsum()

    # --- Extra statistik ---
    trades_df["is_win"] = trades_df["pnl"] > 0

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()  # negativt

    winrate = trades_df["is_win"].mean()

    # Expectancy per trade
    expectancy = trades_df["pnl"].mean()

    # Drawdown
    roll_max = trades_df["equity"].cummax()
    dd = trades_df["equity"] - roll_max
    max_dd = dd.min()  # negativt
    max_dd_points = abs(max_dd)  # positivt för rapportering

    # Longest losing streak (räknat i trades)
    loss_streak = 0
    max_loss_streak = 0
    for is_win in trades_df["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    # “Sharpe” på trade-nivå (inte tidsnormaliserad)
    pnl_std = trades_df["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(trades_df)) if pnl_std and pnl_std > 0 else np.nan

    stats = {
        "Market": market_name,
        "Trades": int(len(trades_df)),
        "Total PnL (points)": float(trades_df["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
    }

    print("\n--- STATS ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

        # ==========================
        # PER-ÅR STATS (per market)
        # ==========================
    print("\n--- PER-ÅR STATS ---")

    # vi använder Exit Time som "trade year" (rekommenderat)
    trades_df["Year"] = pd.to_datetime(trades_df["Exit Time"]).dt.year

    years = sorted(trades_df["Year"].unique().tolist())
    for y in years:
        sub = trades_df[trades_df["Year"] == y].copy()
        y_stats = compute_stats_from_trades(sub)

        if not y_stats:
            continue

        print(f"\n{market_name} - {y}:")
        print(f"Trades: {y_stats['Trades']}")
        print(f"Total PnL (points): {y_stats['Total PnL (points)']:.4f}")
        print(f"Gross Profit: {y_stats['Gross Profit']:.4f}")
        print(f"Gross Loss: {y_stats['Gross Loss']:.4f}")
        print(f"Profit Factor: {y_stats['Profit Factor']:.4f}")
        print(f"Winrate: {y_stats['Winrate']:.4f}")
        print(f"Avg Win: {y_stats['Avg Win']:.4f}")
        print(f"Avg Loss: {y_stats['Avg Loss']:.4f}")
        print(f"Expectancy (avg/trade): {y_stats['Expectancy (avg/trade)']:.4f}")
        print(f"Max Drawdown (points): {y_stats['Max Drawdown (points)']:.4f}")
        print(f"Max Losing Streak (trades): {y_stats['Max Losing Streak (trades)']}")
        print(f"Sharpe (trade-level): {y_stats['Sharpe (trade-level)']:.4f}")

    # PLOT (som du redan får)
    plt.figure(figsize=(12, 5))
    plt.plot(trades_df["Exit Time"], trades_df["equity"])
    plt.title(f"Equity curve - {market_name}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (points)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return stats, trades_df


# ==========================
# KÖR BACKTEST + SLUTSUMMERING + COMBINED EQUITY & STATS
# ==========================

all_results = []
all_trades = []

for m in markets:
    try:
        stats, trades_df = run_backtest_for_market(
            m["name"],
            m["csv"],
            m["pip_size"],
            m.get("spread_points_per_pip", 10.0),
        )
        if stats is not None and trades_df is not None:
            trades_df["Market"] = m["name"]
            all_results.append(stats)
            all_trades.append(trades_df)
    except Exception as e:
        print(f"\n*** FEL för {m['name']} ({m['csv']}): {e}\n")
