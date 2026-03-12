import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

plt.style.use("default")

# ============================================================
# PORTFOLIO CONFIG
# ============================================================
START_CAPITAL = 50_000.0
TOTAL_EXPOSURE = 10.0  # total notional ≈ equity * TOTAL_EXPOSURE
SESSION_START = "00:00:00"
SESSION_END = "07:00:00"
VWAP_RESET = "20:00:00"

# Strategy params (locked)
ENTRY_STD = 2.25
EXIT_STD = 0.75

# ============================================================
# BOOTSTRAP CONFIG
# ============================================================
BOOT_BLOCK_SIZE = 20
BOOT_N = 5000
BOOT_SEED = 42
BLOCK_SIZES_TO_TEST = (10, 20, 40, 60)

# ============================================================
# MARKETS
# ============================================================
PORTFOLIO_MARKETS = ["EURUSD", "EURCHF", "GBPCHF", "EURCAD", "USDCHF"]
RATES_ONLY_MARKETS = ["USDCAD", "GBPUSD"]  # needed for EURCAD quote->USD conversion + GBP base conversion
ALL_MARKETS = PORTFOLIO_MARKETS + RATES_ONLY_MARKETS

# Pip sizes
PIP_SIZE = {
    "EURUSD": 0.0001,
    "USDCHF": 0.0001,
    "EURCHF": 0.0001,
    "GBPCHF": 0.0001,
    "EURCAD": 0.0001,
    "GBPUSD": 0.0001,  # rate only
    "USDCAD": 0.0001,  # rates only
}

# ============================================================
# COST MODEL (PIPS)
# ============================================================
HALF = 0.5
COMM_PIPS_PER_SIDE = 0.25  # fixed (FTMO-style), same for all
DEFAULT_COSTS = {"slippage_pips": 0.10, "spread_pips": 0.20}

COSTS_BY_MARKET = {
    "EURUSD": {"slippage_pips": 0.10, "spread_pips": 0.20},
    "GBPUSD": {"slippage_pips": 0.12, "spread_pips": 0.25},
    "USDCHF": {"slippage_pips": 0.10, "spread_pips": 0.22},
    "EURCHF": {"slippage_pips": 0.10, "spread_pips": 0.22},
    "GBPCHF": {"slippage_pips": 0.15, "spread_pips": 0.30},
    "EURCAD": {"slippage_pips": 0.15, "spread_pips": 0.30},
    "USDCAD": {"slippage_pips": 0.12, "spread_pips": 0.25},  # rate-only, costs not used
}

# CSV naming convention: "{MARKET}_1H_2012-2026.csv"
def csv_name(mkt: str) -> str:
    return f"{mkt}_1H_2012-2026.csv"


# ============================================================
# STATS HELPERS
# ============================================================
TRADING_DAYS_PER_YEAR = 252

def max_drawdown_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min() * 100.0)

def max_intraday_drawdown_pct(equity_hourly: pd.Series) -> float:
    df = equity_hourly.to_frame("eq").dropna()
    df["date"] = df.index.date
    worst = 0.0
    for _, sub in df.groupby("date"):
        peak = sub["eq"].cummax()
        dd = sub["eq"] / peak - 1.0
        worst = min(worst, float(dd.min()))
    return float(worst * 100.0)

def sharpe_sortino_annualized(equity_hourly: pd.Series):
    daily = equity_hourly.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 30:
        return np.nan, np.nan

    ann = np.sqrt(TRADING_DAYS_PER_YEAR)
    mu = rets.mean()
    sig = rets.std(ddof=1)
    sharpe = float((mu / sig) * ann) if sig > 0 else np.nan

    downside = rets[rets < 0]
    dstd = downside.std(ddof=1)
    sortino = float((mu / dstd) * ann) if dstd > 0 else np.nan
    return sharpe, sortino

def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return np.nan
    t0 = eq.index[0]
    t1 = eq.index[-1]
    years = (t1 - t0).days / 365.25
    if years <= 0:
        return np.nan
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)

def equity_to_daily_returns(equity_hourly: pd.Series) -> pd.Series:
    daily = equity_hourly.resample("1D").last().dropna()
    return daily.pct_change().dropna()


# ============================================================
# FX CONVERSION HELPERS
# ============================================================
def split_pair(mkt: str):
    return mkt[:3], mkt[3:]

def get_rate_at(ts: pd.Timestamp, series: pd.Series) -> float:
    ts = pd.Timestamp(ts)
    idx = series.index
    if ts in idx:
        return float(series.loc[ts])
    j = idx.searchsorted(ts, side="right") - 1
    if j < 0:
        return float(series.iloc[0])
    return float(series.iloc[j])

def base_to_usd_rate(mkt: str, ts: pd.Timestamp, rates: dict) -> float:
    base, _ = split_pair(mkt)
    if base == "USD":
        return 1.0

    pair = base + "USD"
    if pair in rates:
        return get_rate_at(ts, rates[pair])

    inv = "USD" + base
    if inv in rates:
        return 1.0 / get_rate_at(ts, rates[inv])

    raise ValueError(f"Missing base->USD conversion for {mkt}. Need {pair} or {inv} in rates.")

def quote_to_usd_rate(mkt: str, ts: pd.Timestamp, rates: dict) -> float:
    _, quote = split_pair(mkt)

    if quote == "USD":
        return 1.0

    pair = quote + "USD"
    if pair in rates:
        return get_rate_at(ts, rates[pair])

    inv = "USD" + quote
    if inv in rates:
        return 1.0 / get_rate_at(ts, rates[inv])

    raise ValueError(f"Missing quote->USD conversion for {mkt}. Need {pair} or {inv} in rates.")

def compute_trade_pnl_usd(tr: pd.Series, units_base: float, rates: dict) -> float:
    mkt = tr["Market"]
    entry = float(tr["Entry Price"])
    exitp = float(tr["Exit Price"])
    sgn = 1.0 if tr["Direction"] == "LONG" else -1.0

    pnl_quote = units_base * (exitp - entry) * sgn
    q2u = quote_to_usd_rate(mkt, tr["Exit Time"], rates)
    return float(pnl_quote * q2u)


# ============================================================
# VWAP + STD
# ============================================================
def compute_session_anchored_vwap_and_std(data: pd.DataFrame, vol_col: str, reset_time: str):
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
    std = g["tp"].transform(lambda x: x.expanding().std(ddof=0))
    return vwap, std


# ============================================================
# SINGLE MARKET BACKTEST
# ============================================================
def run_backtest_for_market(market_name: str, csv_path: str, pip_size: float, spread_points_per_pip: float = 10.0):
    print("\n" + "=" * 70)
    print(f" BACKTEST FÖR MARKNAD: {market_name} ")
    print("=" * 70 + "\n")

    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].sort_index()

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla kolumnerna: {required_cols}")

    if "volume" in df.columns:
        vol_col = "volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        df["volume_dummy"] = 1.0
        vol_col = "volume_dummy"

    df["VWAP"], df["TP_STD"] = compute_session_anchored_vwap_and_std(df, vol_col, VWAP_RESET)

    session_start_t = pd.to_datetime(SESSION_START).time()
    session_end_t = pd.to_datetime(SESSION_END).time()

    def in_session(ts) -> bool:
        t = ts.time()
        if session_start_t < session_end_t:
            return (t >= session_start_t) and (t < session_end_t)
        return (t >= session_start_t) or (t < session_end_t)

    costs = COSTS_BY_MARKET.get(market_name, DEFAULT_COSTS)
    SLIPPAGE_PIPS = float(costs["slippage_pips"])
    FIXED_SPREAD_PIPS = float(costs["spread_pips"])
    COMM = float(COMM_PIPS_PER_SIDE)

    def pips_to_price(pips: float) -> float:
        return float(pips) * float(pip_size)

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(COMM)

    USE_SPREAD_PIPS_COL = "spread_pips" in df.columns
    USE_SPREAD_POINTS_COL = "spread_points" in df.columns

    def get_spread_pips(row) -> float:
        if USE_SPREAD_PIPS_COL:
            return float(row["spread_pips"])
        if USE_SPREAD_POINTS_COL:
            return float(row["spread_points"]) / float(spread_points_per_pip)
        return float(FIXED_SPREAD_PIPS)

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

        # EXIT
        if in_position:
            exit_price = None
            exit_reason = None

            vwap = row["VWAP"]
            std = row["TP_STD"]

            if np.isfinite(vwap) and np.isfinite(std) and std != 0:
                if pos_direction == "LONG":
                    final_level = vwap - EXIT_STD * std
                    if row["close"] >= final_level:
                        spread_pips = get_spread_pips(next_row)
                        spread_px = pips_to_price(spread_pips)
                        slip_px = pips_to_price(SLIPPAGE_PIPS)
                        exit_price = next_row["open"] - HALF * spread_px - slip_px
                        exit_reason = "final_exit"
                else:
                    final_level = vwap + EXIT_STD * std
                    if row["close"] <= final_level:
                        spread_pips = get_spread_pips(next_row)
                        spread_px = pips_to_price(spread_pips)
                        slip_px = pips_to_price(SLIPPAGE_PIPS)
                        exit_price = next_row["open"] + HALF * spread_px + slip_px
                        exit_reason = "final_exit"

            if exit_price is not None:
                exit_time = idx_list[i + 1]
                comm_px = commission_round_turn_price()

                if pos_direction == "LONG":
                    pnl = (exit_price - entry_price) - comm_px
                else:
                    pnl = (entry_price - exit_price) - comm_px

                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Direction": pos_direction,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Exit Reason": exit_reason,
                    "pnl": pnl,
                })

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None
                continue

        if not in_session(ts) or not in_session(idx_list[i + 1]):
            continue
        if in_position:
            continue

        close_price = row["close"]
        prev_close = prev_row["close"]
        next_open = next_row["open"]
        vwap = row["VWAP"]
        std = row["TP_STD"]

        if not np.isfinite(vwap) or not np.isfinite(std) or std == 0:
            continue

        upper_band = vwap + ENTRY_STD * std
        lower_band = vwap - ENTRY_STD * std
        prev_upper_band = prev_row["VWAP"] + ENTRY_STD * prev_row["TP_STD"]
        prev_lower_band = prev_row["VWAP"] - ENTRY_STD * prev_row["TP_STD"]

        upper_band_break = prev_close < prev_upper_band and close_price > upper_band
        lower_band_break = prev_close > prev_lower_band and close_price < lower_band

        if lower_band_break:
            pos_direction = "LONG"
            entry_time = idx_list[i + 1]
            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(SLIPPAGE_PIPS)
            entry_price = next_open + HALF * spread_px + slip_px
            in_position = True

        elif upper_band_break:
            pos_direction = "SHORT"
            entry_time = idx_list[i + 1]
            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(SLIPPAGE_PIPS)
            entry_price = next_open - HALF * spread_px - slip_px
            in_position = True

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("Inga trades hittades.")
        return None, trades_df, df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)
    trades_df["equity"] = trades_df["pnl"].cumsum()

    trades_df["is_win"] = trades_df["pnl"] > 0
    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf
    expectancy = trades_df["pnl"].mean()
    pnl_std = trades_df["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(trades_df)) if pnl_std and pnl_std > 0 else np.nan

    print("--- STATS (price units) ---")
    print("Market:", market_name)
    print("Trades:", len(trades_df))
    print("Profit Factor:", float(profit_factor))
    print("Sharpe (trade-level):", float(sharpe_trade) if np.isfinite(sharpe_trade) else sharpe_trade)

    return {"Market": market_name}, trades_df, df


# ============================================================
# HOURLY EQUITY FOR ONE MARKET (for correlation/weights)
# ============================================================
def build_hourly_equity_single_market(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    rates: dict,
    exposure: float = 1.0,
    start_capital: float = 1.0,
):
    mkt = trades_df["Market"].iloc[0]

    trades = trades_df.copy()
    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"])
    trades["Exit Time"] = pd.to_datetime(trades["Exit Time"])
    trades = trades.sort_values("Entry Time").reset_index(drop=True)
    trades["entry_hour"] = trades["Entry Time"].dt.floor("h")
    trades["exit_hour"] = trades["Exit Time"].dt.floor("h")

    entries_by_hour = {h: sub for h, sub in trades.groupby("entry_hour")}
    exits_by_hour = {h: sub for h, sub in trades.groupby("exit_hour")}

    idx = price_df.index
    close = price_df["close"].astype(float)

    cash = float(start_capital)
    equity = pd.Series(np.nan, index=idx, dtype=float)

    open_pos = False
    direction = None
    entry_price = None
    units_base = 0.0

    for t in idx:
        if t in exits_by_hour and open_pos:
            tr = exits_by_hour[t].iloc[0]
            pnl_usd = compute_trade_pnl_usd(tr, units_base, rates)
            cash += pnl_usd
            open_pos = False
            direction = None
            entry_price = None
            units_base = 0.0

        unreal = 0.0
        if open_pos:
            sgn = 1.0 if direction == "LONG" else -1.0
            pnl_quote = units_base * (float(close.loc[t]) - float(entry_price)) * sgn
            unreal = pnl_quote * quote_to_usd_rate(mkt, t, rates)

        eq_now = cash + unreal

        if t in entries_by_hour and not open_pos:
            tr = entries_by_hour[t].iloc[0]
            notional_usd = eq_now * exposure
            b2u = base_to_usd_rate(mkt, t, rates)
            units_base = notional_usd / b2u if b2u != 0 else 0.0

            open_pos = True
            direction = tr["Direction"]
            entry_price = float(tr["Entry Price"])

        unreal = 0.0
        if open_pos:
            sgn = 1.0 if direction == "LONG" else -1.0
            pnl_quote = units_base * (float(close.loc[t]) - float(entry_price)) * sgn
            unreal = pnl_quote * quote_to_usd_rate(mkt, t, rates)

        equity.loc[t] = cash + unreal

    return equity


# ============================================================
# ERC WEIGHTS
# ============================================================
def compute_erc_weights(returns_df: pd.DataFrame, ridge: float = 1e-6):
    r = returns_df.dropna().copy()
    cols = list(r.columns)
    X = r.values.astype(float)

    cov = np.cov(X, rowvar=False, ddof=1).astype(float)
    cov = cov + float(ridge) * np.eye(cov.shape[0])

    from scipy.optimize import minimize

    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = [(1e-12, 1.0) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def risk_contrib(w):
        w = np.asarray(w, dtype=float).reshape(-1, 1)
        port_var = (w.T @ cov @ w).item()
        port_vol = np.sqrt(max(port_var, 1e-18))
        mrc = (cov @ w) / port_vol
        rc = (w * mrc).reshape(-1)
        return rc

    def objective(w):
        rc = risk_contrib(w)
        if np.any(rc <= 0) or not np.all(np.isfinite(rc)):
            return 1e12
        return float(np.sum((np.log(rc) - np.mean(np.log(rc))) ** 2))

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 5000, "ftol": 1e-14, "disp": False},
    )
    if not res.success:
        raise RuntimeError(f"ERC optimization failed: {res.message}")

    w = np.asarray(res.x, dtype=float)
    w = np.clip(w, 0.0, 1.0)
    w = w / w.sum()

    erc_w = pd.Series(w, index=cols, name="erc_weight")
    cov_df = pd.DataFrame(cov, index=cols, columns=cols)
    return erc_w, cov_df

def erc_risk_contributions(weights: pd.Series, cov: pd.DataFrame):
    w = np.asarray(weights.values, dtype=float).reshape(-1, 1)
    S = np.asarray(cov.values, dtype=float)

    port_var = (w.T @ S @ w).item()
    port_vol = float(np.sqrt(max(port_var, 0.0)))

    if port_vol == 0.0:
        rc = np.zeros((len(weights),), dtype=float)
        return pd.Series(rc, index=weights.index, name="risk_contrib"), port_vol

    mrc = (S @ w) / port_vol
    rc = (w * mrc).reshape(-1)
    return pd.Series(rc, index=weights.index, name="risk_contrib"), port_vol


# ============================================================
# PORTFOLIO HOURLY MTM EQUITY (weights=None => equal)
# ============================================================
def build_hourly_portfolio_equity(
    trades_all: pd.DataFrame,
    prices_by_market: dict,
    rates: dict,
    total_exposure: float = 1.0,
    start_capital: float = START_CAPITAL,
    weights: dict | None = None,
):
    trades_all = trades_all.copy()
    trades_all["Entry Time"] = pd.to_datetime(trades_all["Entry Time"])
    trades_all["Exit Time"] = pd.to_datetime(trades_all["Exit Time"])
    trades_all = trades_all.sort_values("Entry Time").reset_index(drop=True)

    mkts = sorted(trades_all["Market"].unique().tolist())
    if len(mkts) == 0:
        raise ValueError("No markets in trades_all.")

    n_mkts = len(mkts)
    if weights is None:
        w = {m: 1.0 / n_mkts for m in mkts}
    else:
        w_raw = {m: float(weights.get(m, 0.0)) for m in mkts}
        s = float(sum(w_raw.values()))
        if s <= 0:
            w = {m: 1.0 / n_mkts for m in mkts}
        else:
            w = {m: (v / s) for m, v in w_raw.items()}

    t0 = min(prices_by_market[m]["close"].index.min() for m in mkts)
    t1 = max(prices_by_market[m]["close"].index.max() for m in mkts)
    idx = pd.date_range(t0, t1, freq="h")

    closes = {m: prices_by_market[m]["close"].reindex(idx).ffill() for m in mkts}

    trades_all["entry_hour"] = trades_all["Entry Time"].dt.floor("h")
    trades_all["exit_hour"] = trades_all["Exit Time"].dt.floor("h")
    entries_by_hour = {h: sub for h, sub in trades_all.groupby("entry_hour")}
    exits_by_hour = {h: sub for h, sub in trades_all.groupby("exit_hour")}

    state = {
        m: {"open": False, "dir": None, "entry": None, "units_base": 0.0}
        for m in mkts
    }

    cash = float(start_capital)
    equity = pd.Series(np.nan, index=idx, dtype=float)

    for t in idx:
        # 1) exits
        if t in exits_by_hour:
            for _, tr in exits_by_hour[t].iterrows():
                m = tr["Market"]
                if not state[m]["open"]:
                    continue
                pnl_usd = compute_trade_pnl_usd(tr, state[m]["units_base"], rates)
                cash += pnl_usd
                state[m] = {"open": False, "dir": None, "entry": None, "units_base": 0.0}

        # 2) current equity for sizing
        unreal = 0.0
        for m in mkts:
            if state[m]["open"]:
                sgn = 1.0 if state[m]["dir"] == "LONG" else -1.0
                entryp = float(state[m]["entry"])
                px = float(closes[m].loc[t])
                pnl_quote = state[m]["units_base"] * (px - entryp) * sgn
                q2u = quote_to_usd_rate(m, t, rates)
                unreal += pnl_quote * q2u
        eq_now = cash + unreal

        # 3) entries
        if t in entries_by_hour:
            for _, tr in entries_by_hour[t].iterrows():
                m = tr["Market"]
                if state[m]["open"]:
                    continue

                notional_usd = eq_now * float(total_exposure) * float(w[m])
                b2u = base_to_usd_rate(m, t, rates)
                units_base = notional_usd / b2u if b2u != 0 else 0.0

                state[m]["open"] = True
                state[m]["dir"] = tr["Direction"]
                state[m]["entry"] = float(tr["Entry Price"])
                state[m]["units_base"] = float(units_base)

        # 4) mtm equity
        unreal = 0.0
        for m in mkts:
            if state[m]["open"]:
                sgn = 1.0 if state[m]["dir"] == "LONG" else -1.0
                entryp = float(state[m]["entry"])
                px = float(closes[m].loc[t])
                pnl_quote = state[m]["units_base"] * (px - entryp) * sgn
                q2u = quote_to_usd_rate(m, t, rates)
                unreal += pnl_quote * q2u

        equity.loc[t] = cash + unreal

    return equity


# ============================================================
# BOOTSTRAP FUNCTIONS
# ============================================================
def bootstrap_stats(boot_returns: np.ndarray):
    """
    Returns dict of arrays:
      - cagr
      - sharpe
      - sortino
      - max_dd
      - ann_vol
      - end_mult
    """
    n_boot, T = boot_returns.shape
    years = T / 252.0

    cagr_arr = np.empty(n_boot, dtype=float)
    sharpe_arr = np.empty(n_boot, dtype=float)
    sortino_arr = np.empty(n_boot, dtype=float)
    max_dd_arr = np.empty(n_boot, dtype=float)
    ann_vol_arr = np.empty(n_boot, dtype=float)
    end_mult_arr = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        r = boot_returns[i]
        eq = np.cumprod(1.0 + r)

        end_mult_arr[i] = float(eq[-1])
        cagr_arr[i] = float(eq[-1] ** (1.0 / years) - 1.0)

        sd = float(np.std(r, ddof=1))
        mu = float(np.mean(r))
        ann_vol_arr[i] = float(np.sqrt(252.0) * sd) if sd > 0 else np.nan
        sharpe_arr[i] = float(np.sqrt(252.0) * mu / sd) if sd > 0 else np.nan

        dn = r[r < 0]
        dsd = float(np.std(dn, ddof=1)) if len(dn) > 1 else 0.0
        sortino_arr[i] = float(np.sqrt(252.0) * mu / dsd) if dsd > 0 else np.nan

        peak = np.maximum.accumulate(eq)
        dd = eq / peak - 1.0
        max_dd_arr[i] = float(dd.min())  # negative

    return {
        "cagr": cagr_arr,
        "sharpe": sharpe_arr,
        "sortino": sortino_arr,
        "ann_vol": ann_vol_arr,
        "max_dd": max_dd_arr,
        "end_mult": end_mult_arr,
    }

def summarize_bootstrap(name: str, stats: dict, percentiles=(1, 5, 25, 50, 75, 95, 99)):
    print("\n" + "=" * 70)
    print(f"BOOTSTRAP SUMMARY: {name}")
    print("=" * 70)

    for k in ["cagr", "sharpe", "sortino", "ann_vol", "max_dd", "end_mult"]:
        x = np.asarray(stats[k], dtype=float)
        x = x[np.isfinite(x)]

        print(f"\n{k}")
        if len(x) == 0:
            print("  no finite values")
            continue

        p = np.percentile(x, percentiles)
        print(f"  mean   : {np.mean(x): .6f}")
        print(f"  median : {np.median(x): .6f}")
        for perc, val in zip(percentiles, p):
            print(f"  p{perc:>2}: {val: .6f}")

def paired_block_bootstrap(
    r_equal: pd.Series,
    r_erc: pd.Series,
    block_size: int = 20,
    n_boot: int = 5000,
    seed: int = 42,
):
    """
    Paired block bootstrap:
    same sampled blocks are applied to both series.
    """
    df = pd.concat([r_equal.rename("equal"), r_erc.rename("erc")], axis=1).dropna()

    x_equal = df["equal"].values.astype(float)
    x_erc = df["erc"].values.astype(float)

    T = len(df)
    if T <= block_size + 5:
        raise ValueError(f"Not enough aligned returns ({T}) for block_size={block_size}.")

    rng = np.random.default_rng(seed)

    boot_equal = np.empty((n_boot, T), dtype=float)
    boot_erc = np.empty((n_boot, T), dtype=float)

    max_start = T - block_size

    for b in range(n_boot):
        idxs = []
        while len(idxs) < T:
            start = int(rng.integers(0, max_start + 1))
            idxs.extend(range(start, start + block_size))
        idxs = np.array(idxs[:T], dtype=int)

        boot_equal[b, :] = x_equal[idxs]
        boot_erc[b, :] = x_erc[idxs]

    return boot_equal, boot_erc

def diff_bootstrap_stats(stats_equal: dict, stats_erc: dict):
    out = {}
    for k in stats_equal.keys():
        out[k] = np.asarray(stats_erc[k], dtype=float) - np.asarray(stats_equal[k], dtype=float)
    return out

def summarize_diff(name: str, x: np.ndarray, percentiles=(1, 5, 25, 50, 75, 95, 99)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    if len(x) == 0:
        print("No finite values.")
        return

    p = np.percentile(x, percentiles)
    print(f"mean   : {np.mean(x): .6f}")
    print(f"median : {np.median(x): .6f}")
    for perc, val in zip(percentiles, p):
        print(f"p{perc:>2}: {val: .6f}")

def probability_erc_better(diff_stats: dict):
    """
    diff_stats contains ERC - Equal.
    Higher is better for: cagr, sharpe, sortino, max_dd, end_mult
    Lower is better for: ann_vol
    """
    out = {}

    for k, x in diff_stats.items():
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) == 0:
            out[k] = np.nan
            continue

        if k == "ann_vol":
            out[k] = float(np.mean(x < 0))
        else:
            out[k] = float(np.mean(x > 0))

    return out

def plot_bootstrap_hist(stats_equal: dict, stats_erc: dict, key: str, title: str, bins: int = 60):
    x1 = np.asarray(stats_equal[key], dtype=float)
    x2 = np.asarray(stats_erc[key], dtype=float)
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]

    plt.figure(figsize=(10, 5))
    plt.hist(x1, bins=bins, alpha=0.5, label="Equal")
    plt.hist(x2, bins=bins, alpha=0.5, label="ERC")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_diff_hist(diff_stats: dict, key: str, title: str, bins: int = 60):
    x = np.asarray(diff_stats[key], dtype=float)
    x = x[np.isfinite(x)]

    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=bins, alpha=0.7)
    plt.axvline(0.0, linestyle="--")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def run_paired_bootstrap_for_block_sizes(
    r_equal: pd.Series,
    r_erc: pd.Series,
    block_sizes=(10, 20, 40, 60),
    n_boot: int = 5000,
    seed: int = 42,
):
    """
    Runs paired block bootstrap for multiple block sizes.
    Returns:
        results_table: pd.DataFrame with probabilities + diff summaries
        all_results: dict with full stats/diffs per block size
    """
    rows = []
    all_results = {}

    for i, block_size in enumerate(block_sizes):
        boot_equal, boot_erc = paired_block_bootstrap(
            r_equal=r_equal,
            r_erc=r_erc,
            block_size=block_size,
            n_boot=n_boot,
            seed=seed + i,
        )

        stats_equal = bootstrap_stats(boot_equal)
        stats_erc = bootstrap_stats(boot_erc)
        diff_stats = diff_bootstrap_stats(stats_equal, stats_erc)
        p_better = probability_erc_better(diff_stats)

        row = {
            "block_size": block_size,
            "P_CAGR": p_better["cagr"],
            "P_Sharpe": p_better["sharpe"],
            "P_Sortino": p_better["sortino"],
            "P_AnnVol": p_better["ann_vol"],
            "P_MaxDD": p_better["max_dd"],
            "P_EndMult": p_better["end_mult"],
            "dCAGR_mean": np.nanmean(diff_stats["cagr"]),
            "dSharpe_mean": np.nanmean(diff_stats["sharpe"]),
            "dSortino_mean": np.nanmean(diff_stats["sortino"]),
            "dAnnVol_mean": np.nanmean(diff_stats["ann_vol"]),
            "dMaxDD_mean": np.nanmean(diff_stats["max_dd"]),
            "dEndMult_mean": np.nanmean(diff_stats["end_mult"]),
            "dCAGR_med": np.nanmedian(diff_stats["cagr"]),
            "dSharpe_med": np.nanmedian(diff_stats["sharpe"]),
            "dSortino_med": np.nanmedian(diff_stats["sortino"]),
            "dAnnVol_med": np.nanmedian(diff_stats["ann_vol"]),
            "dMaxDD_med": np.nanmedian(diff_stats["max_dd"]),
            "dEndMult_med": np.nanmedian(diff_stats["end_mult"]),
        }
        rows.append(row)

        all_results[block_size] = {
            "stats_equal": stats_equal,
            "stats_erc": stats_erc,
            "diff_stats": diff_stats,
            "p_better": p_better,
        }

    results_table = pd.DataFrame(rows).sort_values("block_size").reset_index(drop=True)
    return results_table, all_results

def print_block_size_summary_table(results_table: pd.DataFrame):
    print("\n" + "=" * 120)
    print("PAIRED BLOCK BOOTSTRAP SENSITIVITY BY BLOCK SIZE")
    print("=" * 120)

    show_cols = [
        "block_size",
        "P_CAGR",
        "P_Sharpe",
        "P_Sortino",
        "P_AnnVol",
        "P_MaxDD",
        "P_EndMult",
        "dSharpe_mean",
        "dMaxDD_mean",
        "dCAGR_mean",
    ]

    tbl = results_table[show_cols].copy()

    for c in tbl.columns:
        if c != "block_size":
            tbl[c] = tbl[c].astype(float).round(4)

    print(tbl.to_string(index=False))

def plot_probability_vs_block_size(results_table: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(results_table["block_size"], results_table["P_Sharpe"], marker="o", label="P(ERC better Sharpe)")
    plt.plot(results_table["block_size"], results_table["P_Sortino"], marker="o", label="P(ERC better Sortino)")
    plt.plot(results_table["block_size"], results_table["P_MaxDD"], marker="o", label="P(ERC better MaxDD)")
    plt.plot(results_table["block_size"], results_table["P_AnnVol"], marker="o", label="P(ERC better AnnVol)")
    plt.axhline(0.5, linestyle="--")
    plt.xlabel("Block size")
    plt.ylabel("Probability")
    plt.title("Paired Bootstrap Sensitivity by Block Size")
    plt.legend()
    plt.tight_layout()
    plt.show()

def summarize_selected_block_size(all_results: dict, block_size: int):
    if block_size not in all_results:
        print(f"Block size {block_size} not found.")
        return

    res = all_results[block_size]
    diff_stats = res["diff_stats"]
    p_better = res["p_better"]

    print("\n" + "=" * 70)
    print(f"DETAILED PAIRED RESULTS FOR BLOCK SIZE = {block_size}")
    print("=" * 70)

    print("\nP(ERC better than Equal)")
    for key in ["cagr", "sharpe", "sortino", "ann_vol", "max_dd", "end_mult"]:
        print(f"{key:>8}: {p_better[key]:.4f}")

    summarize_diff(f"DIFF CAGR (ERC - Equal), block={block_size}", diff_stats["cagr"])
    summarize_diff(f"DIFF SHARPE (ERC - Equal), block={block_size}", diff_stats["sharpe"])
    summarize_diff(f"DIFF SORTINO (ERC - Equal), block={block_size}", diff_stats["sortino"])
    summarize_diff(f"DIFF ANN_VOL (ERC - Equal), block={block_size}", diff_stats["ann_vol"])
    summarize_diff(f"DIFF MAX_DD (ERC - Equal), block={block_size}", diff_stats["max_dd"])
    summarize_diff(f"DIFF END_MULT (ERC - Equal), block={block_size}", diff_stats["end_mult"])


# ============================================================
# PORTFOLIO REPORT
# ============================================================
def print_portfolio_report(name: str, equity_hourly: pd.Series):
    end_eq = float(equity_hourly.dropna().iloc[-1])
    mdd = max_drawdown_pct(equity_hourly)
    idd = max_intraday_drawdown_pct(equity_hourly)
    sh, so = sharpe_sortino_annualized(equity_hourly)
    growth = cagr(equity_hourly)

    print("\n" + "=" * 70)
    print(f"PORTFOLIO RESULTS (CASH, HOURLY MTM) - {name}")
    print("=" * 70)
    print("Start capital:", START_CAPITAL)
    print("End equity:", end_eq)
    print("CAGR:", growth)
    print("Max DD %:", mdd)
    print("Max Intraday DD %:", idd)
    print("Sharpe (ann, daily):", sh)
    print("Sortino (ann, daily):", so)


# ============================================================
# RUN: BACKTEST ALL + BUILD TRADES
# ============================================================
all_trades = []
prices_by_market = {}

for mkt in ALL_MARKETS:
    pip_size = PIP_SIZE[mkt]
    stats, trades_df, price_df = run_backtest_for_market(
        mkt,
        csv_name(mkt),
        pip_size,
        10.0,
    )

    prices_by_market[mkt] = price_df[["open", "high", "low", "close"]].copy()

    if mkt in PORTFOLIO_MARKETS and trades_df is not None and not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["Market"] = mkt
        all_trades.append(trades_df)

if len(all_trades) == 0:
    raise RuntimeError("No portfolio trades found. Check your CSV paths and strategy filters.")

trades_all = pd.concat(all_trades, ignore_index=True)

# Rates needed
rates = {
    "EURUSD": prices_by_market["EURUSD"]["close"],
    "USDCHF": prices_by_market["USDCHF"]["close"],
    "GBPUSD": prices_by_market["GBPUSD"]["close"],
    "USDCAD": prices_by_market["USDCAD"]["close"],
}

# ============================================================
# BUILD RETURNS_DF FOR ERC (per-market daily returns)
# ============================================================
market_daily_returns = {}

for mkt in PORTFOLIO_MARKETS:
    tdf = trades_all[trades_all["Market"] == mkt].copy()
    if tdf.empty:
        continue

    eq_mkt_hourly = build_hourly_equity_single_market(
        trades_df=tdf,
        price_df=prices_by_market[mkt],
        rates=rates,
        exposure=1.0,
        start_capital=1.0,
    )

    market_daily_returns[mkt] = equity_to_daily_returns(eq_mkt_hourly)

returns_df = pd.concat(market_daily_returns, axis=1).dropna()
returns_df.columns = list(market_daily_returns.keys())

# ============================================================
# ERC WEIGHTS + CHECK
# ============================================================
erc_w, cov = compute_erc_weights(returns_df, ridge=1e-6)

print("\n" + "=" * 70)
print("ERC WEIGHTS")
print("=" * 70)
print(erc_w.sort_values(ascending=False))
print("Sum weights:", float(erc_w.sum()))

rc, pvol = erc_risk_contributions(erc_w, cov)
print("\nRisk contributions (RC / mean(RC)) ~ 1.0 => ERC ok:")
print((rc / rc.mean()).round(4))

# ============================================================
# PORTFOLIO SIM: Equal vs ERC
# ============================================================
equity_equal = build_hourly_portfolio_equity(
    trades_all=trades_all,
    prices_by_market={m: prices_by_market[m] for m in PORTFOLIO_MARKETS},
    rates=rates,
    total_exposure=TOTAL_EXPOSURE,
    start_capital=START_CAPITAL,
    weights=None,
)

equity_erc = build_hourly_portfolio_equity(
    trades_all=trades_all,
    prices_by_market={m: prices_by_market[m] for m in PORTFOLIO_MARKETS},
    rates=rates,
    total_exposure=TOTAL_EXPOSURE,
    start_capital=START_CAPITAL,
    weights=erc_w.to_dict(),
)

# ============================================================
# REPORT: RAW PORTFOLIO RESULTS
# ============================================================
print_portfolio_report("EQUAL WEIGHTED", equity_equal)
print_portfolio_report("ERC WEIGHTED", equity_erc)

plt.figure(figsize=(12, 5))
plt.plot(equity_equal.index, equity_equal.values, label="Equal")
plt.plot(equity_erc.index, equity_erc.values, label="ERC")
plt.title("Portfolio Equity (hourly MTM): Equal vs ERC")
plt.xlabel("Time")
plt.ylabel("Equity (USD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
erc_w.sort_values(ascending=False).plot(kind="bar")
plt.title("ERC weights")
plt.ylabel("Weight")
plt.tight_layout()
plt.show()

# ============================================================
# DAILY RETURNS OF BOTH PORTFOLIOS
# ============================================================
r_equal = equity_to_daily_returns(equity_equal)
r_erc = equity_to_daily_returns(equity_erc)

# ============================================================
# SINGLE PAIRED BLOCK BOOTSTRAP
# ============================================================
boot_equal, boot_erc = paired_block_bootstrap(
    r_equal=r_equal,
    r_erc=r_erc,
    block_size=BOOT_BLOCK_SIZE,
    n_boot=BOOT_N,
    seed=BOOT_SEED,
)

stats_equal = bootstrap_stats(boot_equal)
stats_erc = bootstrap_stats(boot_erc)

summarize_bootstrap(f"EQUAL (paired block={BOOT_BLOCK_SIZE}, n={BOOT_N})", stats_equal)
summarize_bootstrap(f"ERC   (paired block={BOOT_BLOCK_SIZE}, n={BOOT_N})", stats_erc)

diff_stats = diff_bootstrap_stats(stats_equal, stats_erc)

print("\n" + "=" * 70)
print("PAIRED BLOCK BOOTSTRAP: ERC - EQUAL")
print("=" * 70)

summarize_diff("DIFF CAGR (ERC - Equal)", diff_stats["cagr"])
summarize_diff("DIFF SHARPE (ERC - Equal)", diff_stats["sharpe"])
summarize_diff("DIFF SORTINO (ERC - Equal)", diff_stats["sortino"])
summarize_diff("DIFF ANN_VOL (ERC - Equal)", diff_stats["ann_vol"])
summarize_diff("DIFF MAX_DD (ERC - Equal)", diff_stats["max_dd"])
summarize_diff("DIFF END_MULT (ERC - Equal)", diff_stats["end_mult"])

p_better = probability_erc_better(diff_stats)

print("\n" + "=" * 70)
print("P(ERC better than Equal) - PAIRED")
print("=" * 70)
for key in ["cagr", "sharpe", "sortino", "ann_vol", "max_dd", "end_mult"]:
    print(f"{key:>8}: {p_better[key]:.4f}")

plot_bootstrap_hist(stats_equal, stats_erc, "sharpe", "Paired Bootstrap Sharpe Distribution")
plot_bootstrap_hist(stats_equal, stats_erc, "max_dd", "Paired Bootstrap Max Drawdown Distribution")
plot_bootstrap_hist(stats_equal, stats_erc, "cagr", "Paired Bootstrap CAGR Distribution")
plot_bootstrap_hist(stats_equal, stats_erc, "end_mult", "Paired Bootstrap End Multiple Distribution")

plot_diff_hist(diff_stats, "sharpe", "Paired Bootstrap Difference: Sharpe (ERC - Equal)")
plot_diff_hist(diff_stats, "sortino", "Paired Bootstrap Difference: Sortino (ERC - Equal)")
plot_diff_hist(diff_stats, "max_dd", "Paired Bootstrap Difference: Max DD (ERC - Equal)")
plot_diff_hist(diff_stats, "cagr", "Paired Bootstrap Difference: CAGR (ERC - Equal)")
plot_diff_hist(diff_stats, "ann_vol", "Paired Bootstrap Difference: Annualized Vol (ERC - Equal)")
plot_diff_hist(diff_stats, "end_mult", "Paired Bootstrap Difference: End Multiple (ERC - Equal)")

# ============================================================
# MULTI BLOCK-SIZE PAIRED BOOTSTRAP SENSITIVITY
# ============================================================
results_table, all_block_results = run_paired_bootstrap_for_block_sizes(
    r_equal=r_equal,
    r_erc=r_erc,
    block_sizes=BLOCK_SIZES_TO_TEST,
    n_boot=BOOT_N,
    seed=BOOT_SEED,
)

print_block_size_summary_table(results_table)

# Optional: save table
# results_table.to_csv("paired_bootstrap_blocksize_sensitivity.csv", index=False)

plot_probability_vs_block_size(results_table)

# Optional: detailed output for one chosen block size
summarize_selected_block_size(all_block_results, block_size=20)
