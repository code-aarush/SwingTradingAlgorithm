"""
SwingTrend Strategy Script
Date: 2025-11-04

This script implements the SwingTrend strategy described in the checklist:
- Indicator-only entries & exits (SMA-44, EMA-50, RSI-14, ADX-14, ATR-14)
- No price-action / volume rules included
- Volatility-based position sizing (risk per trade = 1% default)
- Simple vectorized backtester that accepts OHLCV CSV input

Usage:
    python3 swingtrend_strategy.py --data path/to/ohlcv.csv --capital 1000000 --risk 0.01

CSV columns required: Date, Open, High, Low, Close, Volume
Date must be parsable by pandas.to_datetime

Outputs:
 - Prints a performance summary to stdout
 - Writes a trades.csv file with trade-by-trade details
 - Writes an equity_curve.csv file with daily equity values

Notes:
 - This is intended for manual verification and educational backtesting.
 - Slippage, commissions, and margin effects are simplistic (configurable below).
 - Use heavy caution before trading live; backtest on each ticker individually.
"""

import argparse
import pandas as pd
import numpy as np
from math import floor

# ---------- Indicator implementations (vectorized) ----------

def sma(series, period):
    return series.rolling(period, min_periods=period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def adx(df, period=14):
    # Returns ADX, +DI, -DI (all series)
    high = df['High']; low = df['Low']; close = df['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    # Smooth DM
    plus_dm_sm = plus_dm.rolling(window=period, min_periods=1).sum()
    minus_dm_sm = minus_dm.rolling(window=period, min_periods=1).sum()
    plus_di = 100 * (plus_dm_sm / atr_series.replace(0, np.nan))
    minus_di = 100 * (minus_dm_sm / atr_series.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_series = dx.rolling(window=period, min_periods=1).mean()
    return adx_series, plus_di, minus_di

# ---------- Strategy logic (indicator-only) ----------

def generate_indicators(df):
    df = df.copy()
    df['SMA_44'] = sma(df['Close'], 44)
    df['EMA_50'] = ema(df['Close'], 50)
    df['RSI_14'] = rsi(df['Close'], 14)
    df['ATR_14'] = atr(df, 14)
    df['ADX_14'], df['+DI'], df['-DI'] = adx(df, 14)
    return df

def generate_signals(df):
    """
    Signals:
     - Long entry when:
         Close > SMA_44 for 2 consecutive days
         EMA_50 > SMA_44
         55 <= RSI_14 <= 68
         ADX_14 > 25 and +DI > -DI
     - Short entry symmetrical
     Exits:
         - TP = +10% from entry
         - Initial SL = 3 * ATR
         - After +5% move, trail by 2 * ATR (handled in backtester)
         - Exit also when EMA_50 crosses SMA_44 opposite
    """
    df = df.copy()
    # boolean conditions per row
    df['cond_close_above_sma'] = df['Close'] > df['SMA_44']
    df['cond_close_below_sma'] = df['Close'] < df['SMA_44']
    # require 2 consecutive closes above/below sma
    df['above_sma_2'] = df['cond_close_above_sma'] & df['cond_close_above_sma'].shift(1).fillna(False)
    df['below_sma_2'] = df['cond_close_below_sma'] & df['cond_close_below_sma'].shift(1).fillna(False)

    df['ema_gt_sma'] = df['EMA_50'] > df['SMA_44']
    df['ema_lt_sma'] = df['EMA_50'] < df['SMA_44']

    df['rsi_long_ok'] = (df['RSI_14'] >= 55) & (df['RSI_14'] <= 68)
    df['rsi_short_ok'] = (df['RSI_14'] >= 32) & (df['RSI_14'] <= 45)

    df['adx_ok'] = df['ADX_14'] > 25
    df['plus_di_gt_minus'] = df['+DI'] > df['-DI']
    df['minus_di_gt_plus'] = df['-DI'] > df['+DI']

    df['long_signal'] = df['above_sma_2'] & df['ema_gt_sma'] & df['rsi_long_ok'] & df['adx_ok'] & df['plus_di_gt_minus']
    df['short_signal'] = df['below_sma_2'] & df['ema_lt_sma'] & df['rsi_short_ok'] & df['adx_ok'] & df['minus_di_gt_plus']

    return df

# ---------- Position sizing ----------

def position_size(capital, risk_pct, entry_price, atr, sl_multiplier=3):
    """
    Returns number of shares to buy/sell given risk (percentage of capital)
    SL distance = sl_multiplier * atr
    """
    risk_money = capital * risk_pct
    stop_distance = sl_multiplier * atr
    if stop_distance <= 0 or np.isnan(stop_distance):
        return 0, stop_distance
    qty = floor(risk_money / stop_distance)
    return int(qty), stop_distance

# ---------- Simple backtester (vectorized loop-based for trade execution) ----------

def backtest(df, capital=1_000_000, risk_pct=0.01, sl_multiplier=3, tp_pct=0.10, trail_at_gain=0.05, trail_multiplier=2.0, commission_per_trade=0.0, slippage_pct=0.0005):
    """
    Runs a simple backtest. Assumptions:
     - Enter at next day's Open after signal True on day t
     - Exit on TP or SL intraday (simulated using next-day open & daily high/low)
     - Trailing stop enforced by updating stop price each day once price has moved in favor
    """
    trades = []
    equity = capital
    cash = capital
    position = None  # dict with keys: 'side','entry_price','qty','sl','tp','entry_index','max_favourable_price'
    equity_curve = []

    for i in range(len(df)-1):  # we reference i and i+1 for entry at next open
        row = df.iloc[i]
        next_row = df.iloc[i+1]
        date = next_row.name  # use next day as execution day

        # record current equity
        if position is None:
            equity_curve.append({'Date': next_row.name, 'Equity': equity})
        else:
            # mark-to-market using close price
            mtm = position['qty'] * (next_row['Close'] - position['entry_price']) * (1 if position['side']=='long' else -1)
            equity_curve.append({'Date': next_row.name, 'Equity': equity + mtm})

        # If no position, check for signal at i
        if position is None:
            if row.get('long_signal', False):
                # compute sizing
                qty, stop_distance = position_size(equity, risk_pct, next_row['Open'], row['ATR_14'], sl_multiplier=sl_multiplier)
                if qty > 0:
                    entry_price = next_row['Open'] * (1 + slippage_pct)  # assume slippage on entry
                    sl_price = entry_price - stop_distance
                    tp_price = entry_price * (1 + tp_pct)
                    position = {
                        'side': 'long', 'entry_price': entry_price, 'qty': qty,
                        'sl': sl_price, 'tp': tp_price, 'entry_index': i+1,
                        'max_favourable': entry_price, 'sl_multiplier': sl_multiplier
                    }
                    cash -= qty * entry_price + commission_per_trade
                    trades.append({'EntryDate': next_row.name, 'Side': 'Long', 'Entry': entry_price, 'Qty': qty, 'SL': sl_price, 'TP': tp_price})
            elif row.get('short_signal', False):
                qty, stop_distance = position_size(equity, risk_pct, next_row['Open'], row['ATR_14'], sl_multiplier=sl_multiplier)
                if qty > 0:
                    entry_price = next_row['Open'] * (1 - slippage_pct)
                    sl_price = entry_price + stop_distance
                    tp_price = entry_price * (1 - tp_pct)
                    position = {
                        'side': 'short', 'entry_price': entry_price, 'qty': qty,
                        'sl': sl_price, 'tp': tp_price, 'entry_index': i+1,
                        'max_favourable': entry_price, 'sl_multiplier': sl_multiplier
                    }
                    cash += qty * entry_price - commission_per_trade  # short proceeds
                    trades.append({'EntryDate': next_row.name, 'Side': 'Short', 'Entry': entry_price, 'Qty': qty, 'SL': sl_price, 'TP': tp_price})

        else:
            # manage existing position using next_row's high/low to check TP/SL intraday
            # update max_favourable price
            if position['side'] == 'long':
                # update max favourable
                if next_row['High'] > position['max_favourable']:
                    position['max_favourable'] = next_row['High']
                # trailing activation
                if (position['max_favourable'] / position['entry_price'] - 1) >= trail_at_gain:
                    # compute new trailing stop
                    new_stop = position['max_favourable'] - trail_multiplier * next_row['ATR_14']
                    if new_stop > position['sl']:
                        position['sl'] = new_stop
                # check TP hit intraday
                if next_row['High'] >= position['tp']:
                    exit_price = position['tp'] * (1 - slippage_pct)
                    profit = (exit_price - position['entry_price']) * position['qty']
                    equity += profit - commission_per_trade
                    trades[-1].update({'ExitDate': next_row.name, 'Exit': exit_price, 'P&L': profit})
                    position = None
                    continue
                # check SL hit
                if next_row['Low'] <= position['sl']:
                    exit_price = position['sl'] * (1 - slippage_pct)
                    profit = (exit_price - position['entry_price']) * position['qty']
                    equity += profit - commission_per_trade
                    trades[-1].update({'ExitDate': next_row.name, 'Exit': exit_price, 'P&L': profit})
                    position = None
                    continue
                # check EMA/SMA flip exit on close
                if next_row['EMA_50'] < next_row['SMA_44']:
                    exit_price = next_row['Close'] * (1 - slippage_pct)
                    profit = (exit_price - position['entry_price']) * position['qty']
                    equity += profit - commission_per_trade
                    trades[-1].update({'ExitDate': next_row.name, 'Exit': exit_price, 'P&L': profit, 'ExitReason': 'EMA_SMA_flip'})
                    position = None
                    continue

            else:  # short
                if next_row['Low'] < position['max_favourable']:
                    position['max_favourable'] = next_row['Low']
                if (1 - position['max_favourable'] / position['entry_price']) >= trail_at_gain:
                    new_stop = position['max_favourable'] + trail_multiplier * next_row['ATR_14']
                    if new_stop < position['sl']:
                        position['sl'] = new_stop
                # TP
                if next_row['Low'] <= position['tp']:
                    exit_price = position['tp'] * (1 + slippage_pct)
                    profit = (position['entry_price'] - exit_price) * position['qty']
                    equity += profit - commission_per_trade
                    trades[-1].update({'ExitDate': next_row.name, 'Exit': exit_price, 'P&L': profit})
                    position = None
                    continue
                # SL
                if next_row['High'] >= position['sl']:
                    exit_price = position['sl'] * (1 + slippage_pct)
                    profit = (position['entry_price'] - exit_price) * position['qty']
                    equity += profit - commission_per_trade
                    trades[-1].update({'ExitDate': next_row.name, 'Exit': exit_price, 'P&L': profit})
                    position = None
                    continue
                # EMA/SMA flip exit
                if next_row['EMA_50'] > next_row['SMA_44']:
                    exit_price = next_row['Close'] * (1 + slippage_pct)
                    profit = (position['entry_price'] - exit_price) * position['qty']
                    equity += profit - commission_per_trade
                    trades[-1].update({'ExitDate': next_row.name, 'Exit': exit_price, 'P&L': profit, 'ExitReason': 'EMA_SMA_flip'})
                    position = None
                    continue

    # close any open position at final close price
    if position is not None:
        final = df.iloc[-1]
        if position['side']=='long':
            exit_price = final['Close'] * (1 - slippage_pct)
            profit = (exit_price - position['entry_price']) * position['qty']
            equity += profit - commission_per_trade
            trades[-1].update({'ExitDate': final.name, 'Exit': exit_price, 'P&L': profit, 'ExitReason': 'EndClose'})
        else:
            exit_price = final['Close'] * (1 + slippage_pct)
            profit = (position['entry_price'] - exit_price) * position['qty']
            equity += profit - commission_per_trade
            trades[-1].update({'ExitDate': final.name, 'Exit': exit_price, 'P&L': profit, 'ExitReason': 'EndClose'})
        position = None
        equity_curve.append({'Date': final.name, 'Equity': equity})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    return trades_df, equity_df

# ---------- Utilities ----------

def perf_summary(trades_df, equity_df, capital):
    if trades_df.empty:
        print("No trades executed.")
        return
    wins = trades_df[trades_df['P&L']>0]
    losses = trades_df[trades_df['P&L']<=0]
    total_return = (equity_df['Equity'].iloc[-1] / capital - 1) * 100
    win_rate = len(wins) / len(trades_df) * 100
    avg_win = wins['P&L'].mean() if not wins.empty else 0
    avg_loss = losses['P&L'].mean() if not losses.empty else 0
    avg_rr = (wins['P&L'].mean() / -losses['P&L'].mean()) if (not wins.empty and not losses.empty and losses['P&L'].mean()!=0) else np.nan
    print("Trades:", len(trades_df))
    print(f"Total return: {total_return:.2f}%")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Avg win: {avg_win:.2f}, Avg loss: {avg_loss:.2f}, Avg W/L ratio: {avg_rr:.2f}")

# ---------- Main (CLI) ----------

def main():
    parser = argparse.ArgumentParser(description='SwingTrend strategy backtester')
    parser.add_argument('--data', required=True, help='CSV file with OHLCV data (Date,Open,High,Low,Close,Volume)')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Starting capital in INR')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (fraction of capital)')
    parser.add_argument('--sl_mult', type=float, default=3.0, help='Stop-loss multiplier times ATR')
    parser.add_argument('--tp', type=float, default=0.10, help='Take-profit fraction (10% default)')
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=['Date'], index_col='Date').sort_index()
    df = generate_indicators(df)
    df = generate_signals(df)
    trades_df, equity_df = backtest(df, capital=args.capital, risk_pct=args.risk, sl_multiplier=args.sl_mult, tp_pct=args.tp)
    trades_df.to_csv('trades.csv', index=False)
    equity_df.to_csv('equity_curve.csv')
    perf_summary(trades_df, equity_df, args.capital)
    print("Wrote trades.csv and equity_curve.csv to current directory.")

if __name__ == '__main__':
    main()
