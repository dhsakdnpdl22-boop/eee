"""
NDX 일봉 RSI 30 크로스업 매수 + Bear 다이버전스 매도 백테스트 — Streamlit 앱
매수: 일봉 RSI가 30 미만으로 내려갔다가 30 이상으로 다시 올라올 때 현금의 N% 투입
매도: Bear 다이버전스 확인 시 보유 주식의 N% 매도 (기존 유지)
실행: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ══════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="NDX RSI크로스업 백테스트",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 NDX RSI 30 크로스업 매수 + Bear 다이버전스 매도 백테스트")
st.caption("매수: 일봉 RSI < 30 이후 30 이상 회복 시 현금 일부 투입 | 매도: Bear 다이버전스")

# ══════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 파라미터 설정")

    st.subheader("기간 & 자본")
    START           = st.text_input("시작일", "1985-01-01")
    INITIAL_CAPITAL = st.number_input("초기 자본 (USD)", value=100_000, step=10_000)

    st.subheader("RSI 매수 설정")
    RSI_PERIOD       = st.slider("RSI 기간", 7, 21, 14)
    RSI_BUY_THRESH   = st.slider("매수 RSI 임계값 (기본 30)", 10, 40, 30,
                                  help="RSI가 이 값 미만으로 내려갔다가 다시 이 값 이상으로 올라올 때 매수")
    RSI_BUY_PCT      = st.slider("매수 비율 (현금 대비 %)", 5, 100, 30,
                                  help="크로스업 발생 시 보유 현금의 이 비율만큼 매수") / 100
    RSI_BUY_COOLDOWN = st.slider("매수 쿨다운 (봉 수)", 0, 20, 5,
                                  help="동일 사이클 내 연속 매수를 막는 최소 간격 (0=제한없음)")

    st.subheader("Bear 다이버전스 매도 설정")
    LEFT            = st.slider("Pivot 좌측 봉", 2, 10, 5)
    RIGHT           = st.slider("Pivot 우측 봉", 1, 5, 2)
    MIN_RANGE       = st.slider("최소 피벗 간격", 3, 15, 5)
    MAX_RANGE       = st.slider("최대 피벗 간격", 20, 100, 60)
    BEAR_SELL_PCT   = st.slider("Bear 신호 매도비율 (%)", 5, 50, 10) / 100
    BEAR_SELL_BASIS = st.radio(
        "매도비율 기준",
        ["총자산 기준", "보유주식 기준"],
        horizontal=True,
    )

    st.subheader("매매 비용")
    COMMISSION   = st.number_input("수수료 (편도 %)", value=0.03, step=0.01, format="%.2f") / 100
    SLIPPAGE     = st.number_input("슬리피지 (편도 %)", value=0.05, step=0.01, format="%.2f") / 100
    COST_ONE_WAY = COMMISSION + SLIPPAGE

    st.subheader("레버리지 ETF 비용")
    EXPENSE   = st.number_input("운용보수 (%/년)", value=0.95, step=0.05, format="%.2f") / 100
    SPREAD_2X = st.number_input("2x 스왑비용 (%/년)", value=0.80, step=0.05, format="%.2f") / 100
    SPREAD_3X = st.number_input("3x 스왑비용 (%/년)", value=2.05, step=0.05, format="%.2f") / 100

    st.subheader("SMA 필터")
    SMA_PERIOD = st.slider("SMA 기간", 50, 300, 200)

    st.subheader("세금")
    TAX_RATE      = st.number_input("세율 (%)", value=22.0, step=1.0, format="%.1f") / 100
    TAX_DEDUCTION = st.number_input("세금 공제액 (USD)", value=1923, step=100)
    TAX_MONTH     = 5
    apply_tax     = st.toggle("한국 해외주식 세금 적용", value=True)

# ══════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_data(start):
    raw = yf.download("^NDX", start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].copy().dropna()
    df["ret"] = df["Close"].pct_change().fillna(0.0)
    df["rf"]  = 0.0
    return df

# ══════════════════════════════════════════════
# 핵심 함수
# ══════════════════════════════════════════════
def make_leverage(base_df, leverage, spread):
    expense_d = EXPENSE / 252.0
    spread_d  = spread  / 252.0
    lev_ret   = (leverage * base_df["ret"]
                 - (leverage - 1) * base_df["rf"]
                 - expense_d - spread_d)
    return (1.0 + lev_ret).cumprod()

def calc_rsi(series, period=14):
    delta    = series.diff()
    up       = delta.clip(lower=0.0)
    down     = -delta.clip(upper=0.0)
    avg_gain = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def pivot_high(series, left, right):
    vals = series.to_numpy(dtype=float)
    n, out = len(vals), np.zeros(len(vals), dtype=bool)
    for i in range(left, n - right):
        c = vals[i]
        if not np.isfinite(c): continue
        if np.all(c > vals[i-left:i]) and np.all(c > vals[i+1:i+1+right]):
            out[i] = True
    return pd.Series(out, index=series.index)

def bearish_divergence(weekly_high, weekly_close, rsi_w, left, right, min_range, max_range):
    rsi_piv = pivot_high(rsi_w, left=left, right=right)
    piv_idx = np.flatnonzero(rsi_piv.to_numpy())
    confirm = np.zeros(len(rsi_w), dtype=bool)
    rows = []
    for j in range(1, len(piv_idx)):
        p1, p2 = piv_idx[j-1], piv_idx[j]
        dist = p2 - p1
        if not (min_range <= dist <= max_range): continue
        h1, h2       = weekly_high.iloc[p1],   weekly_high.iloc[p2]
        rsi1, rsi2   = rsi_w.iloc[p1],         rsi_w.iloc[p2]
        close1,close2= weekly_close.iloc[p1],  weekly_close.iloc[p2]
        if np.isnan([h1, h2, rsi1, rsi2, close1, close2]).any(): continue
        if (h2 > h1) and (rsi2 < rsi1):
            cidx = p2 + right
            cdate = pd.NaT
            if cidx < len(rsi_w):
                confirm[cidx] = True
                cdate = rsi_w.index[cidx]
            rows.append(dict(type="bear",
                pivot1_date=rsi_w.index[p1], pivot2_date=rsi_w.index[p2],
                confirm_date=cdate, bars_between=dist,
                price_1=round(h1,2), price_2=round(h2,2),
                rsi_1=round(rsi1,2), rsi_2=round(rsi2,2)))
    return pd.Series(confirm, index=rsi_w.index), pd.DataFrame(rows)

def weekly_to_daily_event(weekly_sig, daily_idx):
    aligned = weekly_sig.reindex(daily_idx).fillna(False)
    return aligned.astype(bool)

def make_rsi_crossup_signal(daily_rsi, threshold=30, cooldown=5):
    """
    RSI가 threshold 미만으로 내려간 이후 threshold 이상으로 올라오는 날 매수 신호.
    cooldown: 마지막 매수 신호 이후 최소 n봉은 재발생 억제.
    """
    rsi_vals = daily_rsi.to_numpy(dtype=float)
    n = len(rsi_vals)
    signal = np.zeros(n, dtype=bool)
    was_below = False
    last_signal_i = -999

    for i in range(1, n):
        prev = rsi_vals[i-1]
        curr = rsi_vals[i]
        if np.isnan(prev) or np.isnan(curr):
            continue
        # 한 번이라도 threshold 아래로 내려간 적 있어야
        if prev < threshold:
            was_below = True
        # 크로스업 감지
        if was_below and curr >= threshold and prev < threshold:
            if (i - last_signal_i) > cooldown:
                signal[i] = True
                last_signal_i = i
                was_below = False  # 사이클 리셋

    return pd.Series(signal, index=daily_rsi.index)

# ── 세금
class KoreanTaxTracker:
    def __init__(self):
        self.annual_gains = {}
        self.pending_tax  = {}
    def record_sale(self, year, gain):
        self.annual_gains[year] = self.annual_gains.get(year, 0.0) + gain
    def compute_year_end_tax(self, year):
        gain    = self.annual_gains.get(year, 0.0)
        taxable = max(0.0, gain - TAX_DEDUCTION)
        tax     = taxable * TAX_RATE
        if tax > 0:
            self.pending_tax[year + 1] = tax
        return tax
    def get_tax_payment(self, year, month):
        if month == TAX_MONTH and year in self.pending_tax:
            return self.pending_tax.pop(year)
        return 0.0

# ══════════════════════════════════════════════
# 백테스트: RSI 크로스업 매수 + Bear 다이버전스 매도
# ══════════════════════════════════════════════
def backtest_rsi_cross(price_curve, signal_df, bear_col,
                       initial_capital=100_000,
                       bear_sell_pct=0.10,
                       bear_sell_basis="총자산 기준",
                       rsi_buy_pct=0.30,
                       _apply_tax=True):
    """
    signal_df에 'rsi_crossup' 컬럼(매수), bear_col(매도) 포함.
    매수: 현금의 rsi_buy_pct 비율 투입 (여러 번 가능)
    매도: bear 신호 시 보유 주식의 bear_sell_pct 비율 매도
    """
    price_curve = price_curve.reindex(signal_df.index).dropna()
    work = signal_df.loc[price_curve.index].copy()
    n = len(work)

    asset  = np.empty(n); cash = np.empty(n)
    equity = np.empty(n); cost_b = np.empty(n)
    asset[0] = 0.0; cash[0] = initial_capital
    equity[0] = initial_capital; cost_b[0] = 0.0

    tax_tracker = KoreanTaxTracker()
    trade_rows  = []
    prev_year   = work.index[0].year

    for i in range(1, n):
        date = work.index[i]; year, month = date.year, date.month

        # 세금 납부
        tax_due = tax_tracker.get_tax_payment(year, month) if _apply_tax else 0.0
        if tax_due > 0:
            if cash[i-1] >= tax_due:
                cash[i-1] -= tax_due
            else:
                shortfall  = tax_due - cash[i-1]
                cost_ratio = cost_b[i-1] / asset[i-1] if asset[i-1] > 0 else 0.0
                gain_ratio = 1.0 - cost_ratio
                sell_for_tax = shortfall / (1.0 - gain_ratio * TAX_RATE) if gain_ratio * TAX_RATE < 1 else shortfall
                sell_for_tax = min(sell_for_tax, asset[i-1])
                proceeds = sell_for_tax * (1.0 - COST_ONE_WAY)
                tax_tracker.record_sale(year, sell_for_tax * gain_ratio)
                cost_b[i-1] -= sell_for_tax * cost_ratio
                asset[i-1]  -= sell_for_tax
                cash[i-1]    = max(cash[i-1] + proceeds - tax_due, 0.0)

        if _apply_tax and year != prev_year:
            tax_tracker.compute_year_end_tax(prev_year)
        prev_year = year

        # 가격 변화 반영 (주식만)
        ret = price_curve.iloc[i] / price_curve.iloc[i-1] - 1.0
        asset[i]  = asset[i-1] * (1.0 + ret)
        cash[i]   = cash[i-1]
        cost_b[i] = cost_b[i-1]
        equity[i] = asset[i] + cash[i]

        prev_buy  = bool(work["rsi_crossup"].iloc[i-1])
        prev_bear = bool(work[bear_col].iloc[i-1])

        realized_gain = 0.0; trade_type = None; trade_amount = 0.0

        # ── 매수: RSI 크로스업
        if prev_buy and cash[i] > 1e-6:
            invest_cash = cash[i] * rsi_buy_pct
            if invest_cash > 1e-6:
                cost        = invest_cash * COST_ONE_WAY
                net_invest  = invest_cash - cost
                asset[i]   += net_invest
                cash[i]    -= invest_cash
                cost_b[i]  += net_invest
                trade_type  = "rsi_buy"
                trade_amount = invest_cash

        # ── 매도: Bear 다이버전스
        elif prev_bear and asset[i] > 1e-6:
            if bear_sell_basis == "총자산 기준":
                sell_target = equity[i] * bear_sell_pct
            else:
                sell_target = asset[i] * bear_sell_pct
            actual_sell  = min(sell_target, asset[i])
            cost         = actual_sell * COST_ONE_WAY
            proceeds     = actual_sell - cost
            cost_ratio   = cost_b[i] / asset[i] if asset[i] > 0 else 0.0
            gain_this    = actual_sell * (1.0 - cost_ratio)
            realized_gain = gain_this
            if _apply_tax: tax_tracker.record_sale(year, gain_this)
            cost_b[i] -= actual_sell * cost_ratio
            asset[i]  -= actual_sell
            cash[i]   += proceeds
            trade_type  = "bear_sell"
            trade_amount = actual_sell

        equity[i] = asset[i] + cash[i]

        if trade_type and trade_amount > 1e-6:
            trade_rows.append(dict(
                date=date, signal=trade_type,
                close=work["Close"].iloc[i],
                trade_amt=round(trade_amount, 2),
                realized_gain=round(realized_gain, 2),
                cash_after=round(cash[i], 2),
                asset_after=round(asset[i], 2),
                equity_after=round(equity[i], 2),
            ))

    if _apply_tax:
        tax_tracker.compute_year_end_tax(work.index[-1].year)

    result = pd.DataFrame(index=work.index, data={
        "asset":  asset,
        "cash":   cash,
        "equity": equity,
        "cost_b": cost_b,
        "sw": np.where(equity > 0, asset / equity, 0.0),
        "cw": np.where(equity > 0, cash  / equity, 0.0),
    })
    return result, pd.DataFrame(trade_rows)

# ══════════════════════════════════════════════
# SMA200 벤치마크 백테스트 (비교용)
# ══════════════════════════════════════════════
def backtest_sma200(price_curve, signal_df, initial_capital=100_000, _apply_tax=True):
    price_curve = price_curve.reindex(signal_df.index).dropna()
    work = signal_df.loc[price_curve.index].copy()
    n = len(work)
    asset  = np.empty(n); cash = np.empty(n)
    equity = np.empty(n); cost_b = np.empty(n)
    asset[0] = initial_capital; cash[0] = 0.0
    equity[0] = initial_capital; cost_b[0] = initial_capital
    in_market = True
    tax_tracker = KoreanTaxTracker()
    trade_rows = []; prev_year = work.index[0].year
    for i in range(1, n):
        date = work.index[i]; year, month = date.year, date.month
        tax_due = tax_tracker.get_tax_payment(year, month) if _apply_tax else 0.0
        if tax_due > 0:
            if cash[i-1] >= tax_due:
                cash[i-1] -= tax_due
            else:
                shortfall  = tax_due - cash[i-1]
                cost_ratio = cost_b[i-1] / asset[i-1] if asset[i-1] > 0 else 0.0
                sell_amt   = min(shortfall * 1.1, asset[i-1])
                proceeds   = sell_amt * (1.0 - COST_ONE_WAY)
                tax_tracker.record_sale(year, sell_amt * (1.0 - cost_ratio))
                cost_b[i-1] -= sell_amt * cost_ratio
                asset[i-1]  -= sell_amt
                cash[i-1]    = max(cash[i-1] + proceeds - tax_due, 0.0)
        if _apply_tax and year != prev_year:
            tax_tracker.compute_year_end_tax(prev_year)
        prev_year = year
        ret = price_curve.iloc[i] / price_curve.iloc[i-1] - 1.0
        asset[i] = asset[i-1] * (1.0 + ret) if in_market else asset[i-1]
        cash[i]  = cash[i-1]; cost_b[i] = cost_b[i-1]
        equity[i] = asset[i] + cash[i]
        prev_sma_bull = bool(work["sma_bull"].iloc[i-1])
        if prev_sma_bull and not in_market and cash[i] > 1e-6:
            invest = cash[i]; net = invest * (1.0 - COST_ONE_WAY)
            asset[i] += net; cash[i] = 0.0; cost_b[i] += net
            in_market = True
            trade_rows.append(dict(date=date, signal="sma_buy",
                close=work["Close"].iloc[i], trade_amt=round(invest,2),
                equity_after=round(equity[i],2)))
        elif not prev_sma_bull and in_market and asset[i] > 1e-6:
            sell_amt   = asset[i]; proceeds = sell_amt * (1.0 - COST_ONE_WAY)
            cost_ratio = cost_b[i] / asset[i] if asset[i] > 0 else 0.0
            gain = sell_amt * (1.0 - cost_ratio)
            if _apply_tax: tax_tracker.record_sale(year, gain)
            cost_b[i] = 0.0; asset[i] = 0.0; cash[i] += proceeds
            in_market = False
            trade_rows.append(dict(date=date, signal="sma_sell",
                close=work["Close"].iloc[i], trade_amt=round(sell_amt,2),
                equity_after=round(equity[i]+cash[i],2)))
        equity[i] = asset[i] + cash[i]
    if _apply_tax:
        tax_tracker.compute_year_end_tax(work.index[-1].year)
    result = pd.DataFrame(index=work.index, data={
        "asset": asset, "cash": cash, "equity": equity,
        "sw": np.where(equity>0, asset/equity, 0.0),
        "cw": np.where(equity>0, cash/equity,  0.0),
    })
    return result, pd.DataFrame(trade_rows)

# ══════════════════════════════════════════════
# 통계
# ══════════════════════════════════════════════
def calc_stats(eq, label):
    eq = eq.dropna()
    r  = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr  = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan
    mdd   = (eq / eq.cummax() - 1.0).min()
    sharpe= (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
    calmar= cagr / abs(mdd) if mdd != 0 else np.nan
    return dict(전략=label, CAGR=f"{cagr*100:.2f}%", MDD=f"{mdd*100:.2f}%",
                Sharpe=f"{sharpe:.2f}", Calmar=f"{calmar:.2f}",
                최종자산=f"${eq.iloc[-1]:,.0f}")

def underwater_months(equity):
    rolling_max = equity.cummax()
    return round((equity < rolling_max).sum() / 21.0, 1)

def below_initial_months(equity, initial_capital):
    return round((equity < initial_capital).sum() / 21.0, 1)

def decade_analysis(equity_dict, initial_capital):
    common_idx = list(equity_dict.values())[0].index
    for s in equity_dict.values():
        common_idx = common_idx.intersection(s.index)
    start_year = common_idx[0].year
    end_year   = common_idx[-1].year
    decade_starts = list(range(start_year, end_year + 1, 10))
    rows = []
    for k, ds in enumerate(decade_starts):
        de      = decade_starts[k+1] if k+1 < len(decade_starts) else end_year + 1
        label   = f"{ds}–{min(de-1, end_year)}"
        mask    = (common_idx.year >= ds) & (common_idx.year < de)
        sub_idx = common_idx[mask]
        if len(sub_idx) < 2: continue
        row = {"구간": label}
        for name, eq in equity_dict.items():
            sub        = eq.loc[sub_idx]
            seg_initial = sub.iloc[0]
            years      = (sub.index[-1] - sub.index[0]).days / 365.25
            cagr       = (sub.iloc[-1]/sub.iloc[0])**(1/years)-1 if years>0 else np.nan
            uw_dd      = underwater_months(sub)
            uw_init    = below_initial_months(sub, seg_initial)
            row[f"{name}_CAGR"]    = cagr
            row[f"{name}_UW_DD"]   = uw_dd
            row[f"{name}_UW_INIT"] = uw_init
        rows.append(row)
    row_all = {"구간": "전체"}
    for name, eq in equity_dict.items():
        sub  = eq.loc[common_idx]
        years = (sub.index[-1] - sub.index[0]).days / 365.25
        cagr  = (sub.iloc[-1]/sub.iloc[0])**(1/years)-1 if years>0 else np.nan
        row_all[f"{name}_CAGR"]    = cagr
        row_all[f"{name}_UW_DD"]   = underwater_months(sub)
        row_all[f"{name}_UW_INIT"] = below_initial_months(sub, initial_capital)
    rows.append(row_all)
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════
# 메인 계산 (캐시)
# ══════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def run_all(start, rsi_period, rsi_buy_thresh, rsi_buy_pct, rsi_buy_cooldown,
            left, right, min_range, max_range,
            expense, spread_2x, spread_3x, cost_one_way,
            sma_period, bear_sell_pct, bear_sell_basis,
            initial_capital, apply_tax_flag,
            tax_rate, tax_deduction):
    df = load_data(start)
    df["px_1x"] = (1.0 + df["ret"]).cumprod()
    df["px_2x"] = make_leverage(df, 2, spread_2x)
    df["px_3x"] = make_leverage(df, 3, spread_3x)

    # ── 주봉 Bear 다이버전스
    _last_td = df["Close"].resample("W-FRI").apply(
        lambda x: x.index[-1] if len(x) > 0 else pd.NaT)
    wh_raw = df["High"].resample("W-FRI").max()
    wc_raw = df["Close"].resample("W-FRI").last()

    def reindex_ltd(s):
        ni = _last_td.reindex(s.index); sc = s.copy(); sc.index = ni.values
        sc = sc[sc.index.notna()]; sc = sc[~sc.index.duplicated(keep="last")]
        return sc.sort_index()

    weekly_high  = reindex_ltd(wh_raw)
    weekly_close = reindex_ltd(wc_raw)
    weekly_rsi   = calc_rsi(weekly_close, rsi_period)

    bear_confirm_w, bear_table = bearish_divergence(
        weekly_high, weekly_close, weekly_rsi, left, right, min_range, max_range)

    df["bear_event"] = weekly_to_daily_event(bear_confirm_w, df.index)
    df["sma200"]     = df["Close"].rolling(sma_period, min_periods=sma_period).mean()
    df["sma_bull"]   = (df["Close"] > df["sma200"]).fillna(False)

    # ── 일봉 RSI + 크로스업 신호
    df["daily_rsi"]  = calc_rsi(df["Close"], rsi_period)
    df["rsi_crossup"] = make_rsi_crossup_signal(
        df["daily_rsi"], threshold=rsi_buy_thresh, cooldown=rsi_buy_cooldown)

    # ── 백테스트 (세후)
    res_1x, tr_1x = backtest_rsi_cross(df["px_1x"], df, "bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, rsi_buy_pct, apply_tax_flag)
    res_2x, tr_2x = backtest_rsi_cross(df["px_2x"], df, "bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, rsi_buy_pct, apply_tax_flag)
    res_3x, tr_3x = backtest_rsi_cross(df["px_3x"], df, "bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, rsi_buy_pct, apply_tax_flag)

    # ── 세전
    res_1x_pre, _ = backtest_rsi_cross(df["px_1x"], df, "bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, rsi_buy_pct, False)
    res_2x_pre, _ = backtest_rsi_cross(df["px_2x"], df, "bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, rsi_buy_pct, False)
    res_3x_pre, _ = backtest_rsi_cross(df["px_3x"], df, "bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, rsi_buy_pct, False)

    # ── SMA200 벤치마크
    res_sma_1x, tr_sma_1x = backtest_sma200(df["px_1x"], df, initial_capital, apply_tax_flag)
    res_sma_2x, tr_sma_2x = backtest_sma200(df["px_2x"], df, initial_capital, apply_tax_flag)
    res_sma_3x, tr_sma_3x = backtest_sma200(df["px_3x"], df, initial_capital, apply_tax_flag)

    # ── Buy & Hold
    bh_1x = initial_capital * df["px_1x"] / df["px_1x"].iloc[0]
    bh_2x = initial_capital * df["px_2x"] / df["px_2x"].iloc[0]
    bh_3x = initial_capital * df["px_3x"] / df["px_3x"].iloc[0]

    return (df, bear_table,
            bh_1x, bh_2x, bh_3x,
            res_1x, res_2x, res_3x,
            res_1x_pre, res_2x_pre, res_3x_pre,
            tr_1x, tr_2x, tr_3x,
            res_sma_1x, res_sma_2x, res_sma_3x,
            tr_sma_1x, tr_sma_2x, tr_sma_3x)

# ══════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════
with st.spinner("데이터 다운로드 & 백테스트 계산 중..."):
    (df, bear_table,
     bh_1x, bh_2x, bh_3x,
     res_1x, res_2x, res_3x,
     res_1x_pre, res_2x_pre, res_3x_pre,
     tr_1x, tr_2x, tr_3x,
     res_sma_1x, res_sma_2x, res_sma_3x,
     tr_sma_1x, tr_sma_2x, tr_sma_3x) = run_all(
        START, RSI_PERIOD, RSI_BUY_THRESH, RSI_BUY_PCT, RSI_BUY_COOLDOWN,
        LEFT, RIGHT, MIN_RANGE, MAX_RANGE,
        EXPENSE, SPREAD_2X, SPREAD_3X, COST_ONE_WAY,
        SMA_PERIOD, BEAR_SELL_PCT, BEAR_SELL_BASIS,
        INITIAL_CAPITAL, apply_tax,
        TAX_RATE, TAX_DEDUCTION)

# ══════════════════════════════════════════════
# 탭 구성
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 전략 비교", "📉 신호 차트", "🗓️ 10년 단위 분석", "📋 매매 기록", "📈 비중 & MDD"
])

COLORS = {"BH": "#888780", "RSI크로스": "#378ADD", "SMA200": "#BA7517"}

# ──────────────────────────────────────────────
# 탭 1: 전략 비교
# ──────────────────────────────────────────────
with tab1:
    lev_choice = st.radio("레버리지", ["1x","2x","3x"], horizontal=True, key="lev1")
    lev_map = {
        "1x": (bh_1x, res_1x["equity"], res_1x_pre["equity"],
                       res_sma_1x["equity"]),
        "2x": (bh_2x, res_2x["equity"], res_2x_pre["equity"],
                       res_sma_2x["equity"]),
        "3x": (bh_3x, res_3x["equity"], res_3x_pre["equity"],
                       res_sma_3x["equity"]),
    }
    bh_eq, strat_eq, strat_eq_pre, sma_eq = lev_map[lev_choice]
    tax_label = "세후" if apply_tax else "세전"

    # 지표 카드
    summary_rows = [
        calc_stats(bh_eq,       f"BH {lev_choice}"),
        calc_stats(strat_eq,    f"RSI크로스 {lev_choice} ({tax_label})"),
        calc_stats(sma_eq,      f"SMA200 {lev_choice} ({tax_label})"),
    ]
    cols = st.columns(3)
    for col, row in zip(cols, summary_rows):
        with col:
            st.metric(row["전략"], row["최종자산"],
                      f"CAGR {row['CAGR']} | MDD {row['MDD']}")

    # 누적 수익 그래프
    fig = go.Figure()
    for eq, name, color in [
        (bh_eq,    "BH",       COLORS["BH"]),
        (strat_eq, "RSI크로스", COLORS["RSI크로스"]),
        (sma_eq,   "SMA200",   COLORS["SMA200"]),
    ]:
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name,
            line=dict(color=color, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>"+name+"</extra>"))
    fig.update_yaxes(type="log", tickprefix="$", tickformat=",")
    fig.update_layout(title=f"누적 수익 비교 ({lev_choice}, 로그 스케일)",
                      hovermode="x unified", height=480,
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

    # 성과 요약표
    st.subheader("성과 요약표 (세전 / 세후 비교)")
    all_stats = []
    for lev_label_s, bh_eq_l, pre_eq, post_eq, sma_eq_l in [
        ("1x", bh_1x, res_1x_pre["equity"], res_1x["equity"], res_sma_1x["equity"]),
        ("2x", bh_2x, res_2x_pre["equity"], res_2x["equity"], res_sma_2x["equity"]),
        ("3x", bh_3x, res_3x_pre["equity"], res_3x["equity"], res_sma_3x["equity"]),
    ]:
        all_stats.append(calc_stats(bh_eq_l,  f"BH {lev_label_s}"))
        all_stats.append(calc_stats(pre_eq,   f"RSI크로스 {lev_label_s} 세전"))
        all_stats.append(calc_stats(post_eq,  f"RSI크로스 {lev_label_s} 세후"))
        all_stats.append(calc_stats(sma_eq_l, f"SMA200 {lev_label_s}"))

    def highlight_rows(row):
        if "세후" in row["전략"]:
            return ["background-color: #1a2a3a"] * len(row)
        elif "세전" in row["전략"]:
            return ["background-color: #1a1a2a"] * len(row)
        return [""] * len(row)

    stats_df = pd.DataFrame(all_stats)
    st.dataframe(stats_df.style.apply(highlight_rows, axis=1),
                 use_container_width=True, hide_index=True)

    # 세금 영향
    st.subheader("세금 영향 (세전 → 세후)")
    delta_rows = []
    for lev_label_s, pre_eq, post_eq in [
        ("1x", res_1x_pre["equity"], res_1x["equity"]),
        ("2x", res_2x_pre["equity"], res_2x["equity"]),
        ("3x", res_3x_pre["equity"], res_3x["equity"]),
    ]:
        def pv(s): return float(s.replace("%","").replace("+",""))
        def uv(s): return float(s.replace("$","").replace(",",""))
        pre_s  = calc_stats(pre_eq,  "pre")
        post_s = calc_stats(post_eq, "post")
        delta_rows.append({
            "전략": f"RSI크로스 {lev_label_s}",
            "세전 CAGR":    pre_s["CAGR"],
            "세후 CAGR":    post_s["CAGR"],
            "CAGR 차이":    f"{pv(post_s['CAGR'])-pv(pre_s['CAGR']):+.2f}%p",
            "세전 최종자산": pre_s["최종자산"],
            "세후 최종자산": post_s["최종자산"],
            "최종자산 차이": f"${uv(post_s['최종자산'])-uv(pre_s['최종자산']):+,.0f}",
        })

    def highlight_delta(row):
        styles = [""] * len(row)
        idx = list(row.index)
        for col_name in ["CAGR 차이", "최종자산 차이"]:
            if col_name in idx:
                val = row[col_name].replace("%p","").replace("$","").replace(",","")
                try:
                    num = float(val)
                    styles[idx.index(col_name)] = f"background-color: {'#1a2e1a' if num>=0 else '#2e1a1a'}"
                except: pass
        return styles

    st.dataframe(pd.DataFrame(delta_rows).style.apply(highlight_delta, axis=1),
                 use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# 탭 2: 신호 차트
# ──────────────────────────────────────────────
with tab2:
    date_range = st.slider("기간 선택",
        min_value=df.index[0].to_pydatetime(),
        max_value=df.index[-1].to_pydatetime(),
        value=(df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()),
        format="YYYY-MM-DD", key="date_slider")

    dff = df.loc[date_range[0]:date_range[1]]
    buy_dates  = dff.index[dff["rsi_crossup"]]
    bear_dates = dff.index[dff["bear_event"]]

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.72, 0.28],
                         subplot_titles=["NDX 가격 + 신호", f"일봉 RSI ({RSI_PERIOD})"])

    # 가격 + SMA
    fig2.add_trace(go.Scatter(x=dff.index, y=dff["Close"],
        name="NDX", line=dict(color="#B4B2A9", width=1),
        hovertemplate="%{x|%Y-%m-%d} %{y:,.0f}<extra>NDX</extra>"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=dff.index, y=dff["sma200"],
        name=f"SMA{SMA_PERIOD}", line=dict(color="#444441", width=1.2, dash="dot"),
        hovertemplate="%{y:,.0f}<extra>SMA</extra>"), row=1, col=1)

    # 매수 신호 (RSI 크로스업)
    if len(buy_dates):
        fig2.add_trace(go.Scatter(x=buy_dates, y=dff.loc[buy_dates, "Close"],
            mode="markers", name=f"RSI 크로스업 매수 ({len(buy_dates)})",
            marker=dict(symbol="triangle-up", size=11, color="#1D9E75",
                        line=dict(color="#085041", width=1)),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra>RSI매수</extra>"), row=1, col=1)

    # Bear 다이버전스 매도
    if len(bear_dates):
        fig2.add_trace(go.Scatter(x=bear_dates, y=dff.loc[bear_dates, "Close"],
            mode="markers", name=f"Bear 다이버전스 매도 ({len(bear_dates)})",
            marker=dict(symbol="triangle-down", size=11, color="#D85A30",
                        line=dict(color="#712B13", width=1)),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra>Bear매도</extra>"), row=1, col=1)

    # RSI 패널
    fig2.add_trace(go.Scatter(x=dff.index, y=dff["daily_rsi"],
        name="RSI", line=dict(color="#7F77DD", width=1),
        hovertemplate="%{y:.1f}<extra>RSI</extra>"), row=2, col=1)
    fig2.add_hline(y=RSI_BUY_THRESH, line_dash="dash", line_color="#1D9E75",
                   annotation_text=f"매수 임계 {RSI_BUY_THRESH}",
                   annotation_position="right", row=2, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="#D85A30", row=2, col=1)

    # RSI 30 미만 구간 음영
    rsi_below = dff["daily_rsi"] < RSI_BUY_THRESH
    if rsi_below.any():
        fig2.add_trace(go.Scatter(
            x=dff.index, y=np.where(rsi_below, dff["daily_rsi"], np.nan),
            fill="tozeroy", fillcolor="rgba(29,158,117,0.12)",
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=True, name=f"RSI < {RSI_BUY_THRESH} 구간",
            hoverinfo="skip"), row=2, col=1)

    fig2.update_yaxes(type="log", row=1, col=1)
    fig2.update_layout(height=620, hovermode="x unified",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig2, use_container_width=True)

    # 신호 통계
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"RSI 크로스업 매수 신호 ({len(buy_dates)}건)")
        if len(buy_dates) > 0:
            buy_info = pd.DataFrame({
                "날짜": buy_dates,
                "종가": dff.loc[buy_dates, "Close"].values.round(2),
                "RSI": dff.loc[buy_dates, "daily_rsi"].values.round(2),
            })
            st.dataframe(buy_info, use_container_width=True, hide_index=True)
        else:
            st.info("신호 없음")
    with col2:
        st.subheader(f"Bear 다이버전스 매도 ({len(bear_dates)}건)")
        if len(bear_dates) > 0:
            bear_info = pd.DataFrame({
                "날짜": bear_dates,
                "종가": dff.loc[bear_dates, "Close"].values.round(2),
            })
            st.dataframe(bear_info, use_container_width=True, hide_index=True)
            st.subheader("Bear 다이버전스 상세")
            st.dataframe(bear_table, use_container_width=True, hide_index=True)
        else:
            st.info("신호 없음")

# ──────────────────────────────────────────────
# 탭 3: 10년 단위 분석
# ──────────────────────────────────────────────
with tab3:
    st.info("각 구간 시작 시점 기준 CAGR 및 원금/고점 미회복 기간(개월)입니다.")
    lev3 = st.radio("레버리지", ["1x","2x","3x"], horizontal=True, key="lev3")
    lev3_map = {
        "1x": (bh_1x, res_1x["equity"], res_sma_1x["equity"]),
        "2x": (bh_2x, res_2x["equity"], res_sma_2x["equity"]),
        "3x": (bh_3x, res_3x["equity"], res_sma_3x["equity"]),
    }
    _bh, _strat, _sma = lev3_map[lev3]
    decade_df = decade_analysis({"BH": _bh, "RSI크로스": _strat, "SMA200": _sma}, INITIAL_CAPITAL)
    decade_plot = decade_df[decade_df["구간"] != "전체"].copy()
    decade_all  = decade_df[decade_df["구간"] == "전체"].iloc[0]

    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1,c2,c3], ["BH","RSI크로스","SMA200"]):
        with col:
            st.metric(f"{name} 전체 CAGR", f"{decade_all[f'{name}_CAGR']*100:.2f}%")
            st.caption(f"고점 미회복: {decade_all[f'{name}_UW_DD']:.0f}개월　|　원금 미회복: {decade_all[f'{name}_UW_INIT']:.0f}개월")

    st.divider()
    color_list = [COLORS["BH"], COLORS["RSI크로스"], COLORS["SMA200"]]

    fig3a = go.Figure()
    for name, color in zip(["BH","RSI크로스","SMA200"], color_list):
        fig3a.add_trace(go.Bar(
            x=decade_plot["구간"], y=decade_plot[f"{name}_CAGR"]*100,
            name=name, marker_color=color,
            hovertemplate="%{x}<br>CAGR: %{y:.2f}%<extra>"+name+"</extra>"))
    fig3a.add_hline(y=0, line_color="#888", line_width=0.8)
    fig3a.update_layout(title="10년 단위 CAGR (%)", barmode="group",
                        yaxis_ticksuffix="%", height=360,
                        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig3a, use_container_width=True)

    uw_type = st.radio("원금손실기간 기준",
        ["시작원금 미회복 (진짜 손실 기간)", "고점 미회복 (drawdown 기간)"],
        horizontal=True, key="uw_type")
    uw_col   = "UW_INIT" if "시작원금" in uw_type else "UW_DD"
    uw_label = "시작원금 미회복 (개월)" if "시작원금" in uw_type else "고점 미회복 (개월)"

    fig3b = go.Figure()
    for name, color in zip(["BH","RSI크로스","SMA200"], color_list):
        fig3b.add_trace(go.Bar(
            x=decade_plot["구간"], y=decade_plot[f"{name}_{uw_col}"],
            name=name, marker_color=color,
            hovertemplate="%{x}<br>%{y:.0f}개월<extra>"+name+"</extra>"))
    fig3b.update_layout(title=f"10년 단위 {uw_label}", barmode="group",
                        yaxis_ticksuffix="개월", height=360,
                        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig3b, use_container_width=True)

    display_df = decade_df.copy()
    for name in ["BH","RSI크로스","SMA200"]:
        display_df[f"{name} CAGR"] = display_df[f"{name}_CAGR"].apply(
            lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
        display_df[f"{name} 원금미회복"] = display_df[f"{name}_UW_INIT"].apply(
            lambda x: f"{x:.0f}개월" if pd.notna(x) else "N/A")
        display_df[f"{name} 고점미회복"] = display_df[f"{name}_UW_DD"].apply(
            lambda x: f"{x:.0f}개월" if pd.notna(x) else "N/A")
    show_cols = ["구간",
                 "BH CAGR","BH 원금미회복","BH 고점미회복",
                 "RSI크로스 CAGR","RSI크로스 원금미회복","RSI크로스 고점미회복",
                 "SMA200 CAGR","SMA200 원금미회복","SMA200 고점미회복"]
    st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# 탭 4: 매매 기록
# ──────────────────────────────────────────────
with tab4:
    st.subheader("매매 기록")
    col_l, col_r = st.columns(2)
    with col_l:
        strat4 = st.selectbox("전략", ["RSI크로스업", "SMA200"])
    with col_r:
        lev4 = st.selectbox("레버리지", ["1x","2x","3x"])

    tr_map = {
        ("RSI크로스업","1x"): tr_1x,
        ("RSI크로스업","2x"): tr_2x,
        ("RSI크로스업","3x"): tr_3x,
        ("SMA200","1x"):      tr_sma_1x,
        ("SMA200","2x"):      tr_sma_2x,
        ("SMA200","3x"):      tr_sma_3x,
    }
    trades = tr_map[(strat4, lev4)]
    if len(trades):
        sig_counts = trades["signal"].value_counts()
        sig_labels = {
            "rsi_buy":  "🟢 RSI 크로스업 매수",
            "bear_sell":"🔴 Bear 다이버전스 매도",
            "sma_buy":  "🔵 SMA 매수",
            "sma_sell": "🟡 SMA 매도",
        }
        cnt_cols = st.columns(len(sig_counts))
        for col, (sig, cnt) in zip(cnt_cols, sig_counts.items()):
            col.metric(sig_labels.get(sig, sig), f"{cnt}회")
        st.dataframe(trades, use_container_width=True, hide_index=True)
        csv = trades.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV 다운로드", csv,
            file_name=f"trades_{strat4}_{lev4}.csv", mime="text/csv")
    else:
        st.info("매매 기록 없음")

# ──────────────────────────────────────────────
# 탭 5: 비중 & MDD
# ──────────────────────────────────────────────
with tab5:
    lev5 = st.radio("레버리지", ["1x","2x","3x"], horizontal=True, key="lev5")
    lev5_map = {
        "1x": (bh_1x, res_1x, res_sma_1x),
        "2x": (bh_2x, res_2x, res_sma_2x),
        "3x": (bh_3x, res_3x, res_sma_3x),
    }
    _bh_eq, _strat_res, _sma_res = lev5_map[lev5]

    # Drawdown
    fig5a = go.Figure()
    for eq, name, color in [
        (_bh_eq,              "BH",       COLORS["BH"]),
        (_strat_res["equity"],"RSI크로스", COLORS["RSI크로스"]),
        (_sma_res["equity"],  "SMA200",   COLORS["SMA200"]),
    ]:
        dd = (eq / eq.cummax() - 1.0) * 100
        fig5a.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name,
            fill="tozeroy", line=dict(color=color, width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>"+name+"</extra>"))
    fig5a.update_yaxes(ticksuffix="%")
    fig5a.update_layout(title=f"Drawdown ({lev5})", height=350,
                        hovermode="x unified", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig5a, use_container_width=True)

    # 비중
    fig5b = go.Figure()
    fig5b.add_trace(go.Scatter(x=_strat_res.index, y=_strat_res["sw"]*100,
        name="주식", fill="tozeroy", line=dict(color=COLORS["RSI크로스"], width=1)))
    fig5b.add_trace(go.Scatter(x=_strat_res.index, y=_strat_res["cw"]*100,
        name="현금", fill="tozeroy", line=dict(color="#B4B2A9", width=1)))
    fig5b.update_yaxes(ticksuffix="%", range=[0,100])
    fig5b.update_layout(title=f"RSI크로스 {lev5} — 주식/현금 비중",
                        height=280, hovermode="x unified",
                        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig5b, use_container_width=True)

    # 현금 잔고 추이
    fig5c = go.Figure()
    fig5c.add_trace(go.Scatter(x=_strat_res.index, y=_strat_res["cash"],
        name="현금 잔고", fill="tozeroy",
        line=dict(color="#F5A623", width=1),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>현금</extra>"))
    fig5c.update_yaxes(tickprefix="$", tickformat=",")
    fig5c.update_layout(title=f"현금 잔고 추이 ({lev5})", height=260,
                        hovermode="x unified", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig5c, use_container_width=True)

st.divider()
buy_count  = int(df["rsi_crossup"].sum())
bear_count = int(df["bear_event"].sum())
st.caption(
    f"데이터: Yahoo Finance (^NDX) | 마지막 갱신: {df.index[-1].date()} | "
    f"세금 적용: {'예' if apply_tax else '아니오'} | "
    f"RSI 크로스업 매수: {buy_count}회 (RSI<{RSI_BUY_THRESH}→이상 회복, 현금의 {RSI_BUY_PCT*100:.0f}%) | "
    f"Bear 다이버전스 매도: {bear_count}회 ({BEAR_SELL_PCT*100:.0f}% 매도)"
)
