"""
NDX Weekly RSI Divergence + SMA200 백테스트 — Streamlit 앱
실행: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ══════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="NDX 백테스트 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 NDX 주봉 RSI 다이버전스 + SMA200 백테스트")
st.caption("매일 새로고침하면 최신 데이터로 자동 재계산됩니다.")

# ══════════════════════════════════════════════
# 사이드바 — 파라미터 설정
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 파라미터 설정")

    st.subheader("기간 & 자본")
    START           = st.text_input("시작일", "1985-01-01")
    INITIAL_CAPITAL = st.number_input("초기 자본 (USD)", value=100_000, step=10_000)

    st.subheader("RSI 다이버전스")
    RSI_PERIOD = st.slider("RSI 기간", 7, 21, 14)
    LEFT       = st.slider("Pivot 좌측 봉", 2, 10, 5)
    RIGHT      = st.slider("Pivot 우측 봉", 1, 5, 2)
    MIN_RANGE  = st.slider("최소 피벗 간격", 3, 15, 5)
    MAX_RANGE  = st.slider("최대 피벗 간격", 20, 100, 60)

    st.subheader("매매 비용")
    COMMISSION   = st.number_input("수수료 (편도 %)", value=0.03, step=0.01, format="%.2f") / 100
    SLIPPAGE     = st.number_input("슬리피지 (편도 %)", value=0.05, step=0.01, format="%.2f") / 100
    COST_ONE_WAY = COMMISSION + SLIPPAGE

    st.subheader("레버리지 ETF 비용")
    EXPENSE   = st.number_input("운용보수 (%/년)", value=0.95, step=0.05, format="%.2f") / 100
    SPREAD_2X = st.number_input("2x 스왑비용 (%/년)", value=0.80, step=0.05, format="%.2f") / 100
    SPREAD_3X = st.number_input("3x 스왑비용 (%/년)", value=2.05, step=0.05, format="%.2f") / 100

    st.subheader("기타")
    BEAR_SELL_PCT = st.slider("Bear 신호 매도비율 (%)", 5, 50, 10) / 100
    BEAR_SELL_BASIS = st.radio(
        "매도비율 기준",
        ["총자산 기준", "보유주식 기준"],
        horizontal=True,
        help="총자산 기준: equity × 비율 / 보유주식 기준: asset × 비율"
    )
    SMA_PERIOD    = st.slider("SMA 기간", 50, 300, 200)

    # ── 현금 이자율 ─────────────────────────────
    st.subheader("현금 이자율")
    USE_CASH_RATE = st.toggle("현금 이자 반영", value=True,
                               help="현금 보유 시 단기금리(^IRX, 13주 T-bill) 이자를 복리로 적용합니다.")
    CASH_RATE_SPREAD = st.slider("스프레드 차감 (bp)", 0, 100, 10,
                                  help="T-bill 대비 실제 수령 이자의 차이. 예: 10bp = 0.10%p 차감.") / 100 / 100
    # ──────────────────────────────────────────────

    # ── RSI 선제투입 ─────────────────────────────
    st.subheader("RSI 선제투입 (Divergence 전략 전용)")
    USE_RSI_PREBUY   = st.toggle("RSI 선제투입 활성화", value=True,
                                  help="일봉 RSI가 임계값 미만일 때 현금 일부를 선제 투입합니다.")
    RSI_PREBUY_THRESH = st.slider("선제투입 RSI 임계값", 10, 40, 30,
                                   help="일봉 RSI가 이 값 미만이면 선제투입 실행")
    RSI_PREBUY_PCT    = st.slider("선제투입 비율 (현금 대비 %)", 5, 50, 20,
                                   help="보유 현금의 이 비율만큼 매수. Bear 신호 전까지 사이클 내 1회만 실행.") / 100
    # ──────────────────────────────────────────────

    TAX_RATE      = st.number_input("세율 (%)", value=22.0, step=1.0, format="%.1f") / 100
    TAX_DEDUCTION = st.number_input("세금 공제액 (USD)", value=1923, step=100)
    TAX_MONTH     = 5

    apply_tax = st.toggle("한국 해외주식 세금 적용", value=True)

# ══════════════════════════════════════════════
# 캐시된 데이터 로드
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

@st.cache_data(ttl=3600)
def load_tbill(start):
    """
    ^IRX: 13주(3개월) T-bill 연수익률(%). 야후파이낸스 제공.
    - 일별 실제 금리 시계열로 리샘플 후 결측 forward-fill
    - 반환: pd.Series (일별 연이율, 소수점 형태 e.g. 0.0520)
    """
    raw = yf.download("^IRX", start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    rate = raw["Close"].dropna() / 100.0   # % → 소수
    # 거래일 기준으로 forward-fill (주말·공휴일 제거됨)
    rate = rate[~rate.index.duplicated(keep="last")].sort_index()
    return rate

# ══════════════════════════════════════════════
# 핵심 함수들
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

def pivot_low(series, left, right):
    vals = series.to_numpy(dtype=float)
    n, out = len(vals), np.zeros(len(vals), dtype=bool)
    for i in range(left, n - right):
        c = vals[i]
        if not np.isfinite(c): continue
        if np.all(c < vals[i-left:i]) and np.all(c < vals[i+1:i+1+right]):
            out[i] = True
    return pd.Series(out, index=series.index)

def pivot_high(series, left, right):
    vals = series.to_numpy(dtype=float)
    n, out = len(vals), np.zeros(len(vals), dtype=bool)
    for i in range(left, n - right):
        c = vals[i]
        if not np.isfinite(c): continue
        if np.all(c > vals[i-left:i]) and np.all(c > vals[i+1:i+1+right]):
            out[i] = True
    return pd.Series(out, index=series.index)

def bullish_divergence(weekly_low, weekly_close, rsi_w, left, right, min_range, max_range):
    rsi_piv = pivot_low(rsi_w, left=left, right=right)
    piv_idx = np.flatnonzero(rsi_piv.to_numpy())
    confirm = np.zeros(len(rsi_w), dtype=bool)
    rows = []
    for j in range(1, len(piv_idx)):
        p1, p2 = piv_idx[j-1], piv_idx[j]
        dist = p2 - p1
        if not (min_range <= dist <= max_range): continue
        low1, low2   = weekly_low.iloc[p1],   weekly_low.iloc[p2]
        rsi1, rsi2   = rsi_w.iloc[p1],        rsi_w.iloc[p2]
        close1,close2= weekly_close.iloc[p1], weekly_close.iloc[p2]
        if np.isnan([low1, low2, rsi1, rsi2, close1, close2]).any(): continue
        if (low2 < low1) and (rsi2 > rsi1):
            cidx = p2 + right
            cdate = pd.NaT
            if cidx < len(rsi_w):
                confirm[cidx] = True
                cdate = rsi_w.index[cidx]
            rows.append(dict(type="bull",
                pivot1_date=rsi_w.index[p1], pivot2_date=rsi_w.index[p2],
                confirm_date=cdate, bars_between=dist,
                price_1=round(low1,2), price_2=round(low2,2),
                rsi_1=round(rsi1,2), rsi_2=round(rsi2,2)))
    return pd.Series(confirm, index=rsi_w.index), pd.DataFrame(rows)

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

def reindex_to_last_trading_day(series, last_td_map):
    new_idx = last_td_map.reindex(series.index)
    s = series.copy()
    s.index = new_idx.values
    s = s[s.index.notna()]
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()

def weekly_to_daily_event(weekly_sig, daily_idx):
    aligned = weekly_sig.reindex(daily_idx).fillna(False)
    return aligned.astype(bool)

def build_daily_rate_array(daily_idx, tbill_rate, spread, use_cash_rate):
    """
    일별 현금 이자율 배열 반환 (연이율 소수).
    ^IRX를 일별 인덱스에 맞게 forward-fill.
    데이터 없는 구간(^IRX 시작 전 등)은 0으로 처리.
    """
    if not use_cash_rate:
        return np.zeros(len(daily_idx))
    rate_aligned = tbill_rate.reindex(daily_idx, method="ffill").fillna(0.0)
    rate_adj = (rate_aligned - spread).clip(lower=0.0)
    return rate_adj.to_numpy()

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
# backtest
# ══════════════════════════════════════════════
def backtest(price_curve, signal_df, bull_col, bear_col,
             initial_capital=100_000, bear_sell_pct=0.10,
             bear_sell_basis="총자산 기준",
             _apply_tax=True,
             use_rsi_prebuy=False,
             rsi_prebuy_thresh=30,
             rsi_prebuy_pct=0.20,
             daily_rate=None):          # ★ 추가
    price_curve = price_curve.reindex(signal_df.index).dropna()
    work = signal_df.loc[price_curve.index].copy()
    n = len(work)

    # daily_rate 배열 정렬
    if daily_rate is None:
        cash_rate_arr = np.zeros(n)
    else:
        cash_rate_arr = pd.Series(daily_rate).reindex(
            range(len(daily_rate))).to_numpy()[:n]
        # signal_df 인덱스에 맞게 재정렬
        _rate_s = pd.Series(daily_rate, index=signal_df.index)
        cash_rate_arr = _rate_s.reindex(work.index).fillna(0.0).to_numpy()

    daily_rsi = calc_rsi(work["Close"], RSI_PERIOD)

    asset  = np.empty(n); cash = np.empty(n)
    equity = np.empty(n); cost_b = np.empty(n)
    asset[0] = initial_capital; cash[0] = 0.0
    equity[0] = initial_capital; cost_b[0] = initial_capital

    tax_tracker = KoreanTaxTracker()
    trade_rows  = []
    prev_year   = work.index[0].year
    prebuy_done = False

    for i in range(1, n):
        date = work.index[i]; year, month = date.year, date.month

        # ★ 현금 이자 적용 (전날 현금 잔고에 일일 이자 누적)
        daily_interest = cash[i-1] * (cash_rate_arr[i] / 252.0)
        cash[i-1] += daily_interest

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

        ret = price_curve.iloc[i] / price_curve.iloc[i-1] - 1.0
        asset[i] = asset[i-1] * (1.0 + ret)
        cash[i]  = cash[i-1]; cost_b[i] = cost_b[i-1]
        equity[i] = asset[i] + cash[i]

        prev_bull = bool(work[bull_col].iloc[i-1])
        prev_bear = bool(work[bear_col].iloc[i-1])

        realized_gain = 0.0; trade_type = None; trade_amount = 0.0

        if prev_bull and cash[i] > 1e-6:
            invest_cash = cash[i]; cost = invest_cash * COST_ONE_WAY
            net_invest  = invest_cash - cost
            asset[i] += net_invest; cash[i] = 0.0; cost_b[i] += net_invest
            trade_type = "bull_buy"; trade_amount = invest_cash

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
            realized_gain= gain_this
            if _apply_tax: tax_tracker.record_sale(year, gain_this)
            cost_b[i] -= actual_sell * cost_ratio
            asset[i]  -= actual_sell; cash[i] += proceeds
            trade_type = "bear_sell"; trade_amount = actual_sell
            prebuy_done = False

        elif (use_rsi_prebuy
              and not prebuy_done
              and not prev_bull
              and not prev_bear
              and cash[i] > 1e-6):
            rsi_val = daily_rsi.iloc[i-1]
            if pd.notna(rsi_val) and rsi_val < rsi_prebuy_thresh:
                invest_cash = cash[i] * rsi_prebuy_pct
                if invest_cash > 1e-6:
                    cost       = invest_cash * COST_ONE_WAY
                    net_invest = invest_cash - cost
                    asset[i]  += net_invest; cash[i] -= invest_cash
                    cost_b[i] += net_invest
                    trade_type  = "rsi_prebuy"; trade_amount = invest_cash
                    prebuy_done = True

        equity[i] = asset[i] + cash[i]

        if trade_type and trade_amount > 1e-6:
            trade_rows.append(dict(date=date, signal=trade_type,
                close=work["Close"].iloc[i], trade_amt=round(trade_amount,2),
                realized_gain=round(realized_gain,2),
                equity_after=round(equity[i],2)))

    if _apply_tax:
        tax_tracker.compute_year_end_tax(work.index[-1].year)

    result = pd.DataFrame(index=work.index, data={
        "asset": asset, "cash": cash, "equity": equity, "cost_b": cost_b,
        "sw": np.where(equity > 0, asset/equity, 0.0),
        "cw": np.where(equity > 0, cash/equity,  0.0),
    })
    return result, pd.DataFrame(trade_rows)

def backtest_sma200(price_curve, signal_df, initial_capital=100_000,
                    _apply_tax=True, daily_rate=None):   # ★ 추가
    price_curve = price_curve.reindex(signal_df.index).dropna()
    work = signal_df.loc[price_curve.index].copy()
    n = len(work)

    if daily_rate is None:
        cash_rate_arr = np.zeros(n)
    else:
        _rate_s = pd.Series(daily_rate, index=signal_df.index)
        cash_rate_arr = _rate_s.reindex(work.index).fillna(0.0).to_numpy()

    asset  = np.empty(n); cash = np.empty(n)
    equity = np.empty(n); cost_b = np.empty(n)
    asset[0] = initial_capital; cash[0] = 0.0
    equity[0] = initial_capital; cost_b[0] = initial_capital
    in_market = True
    tax_tracker = KoreanTaxTracker()
    trade_rows = []; prev_year = work.index[0].year
    for i in range(1, n):
        date = work.index[i]; year, month = date.year, date.month

        # ★ 현금 이자 적용
        daily_interest = cash[i-1] * (cash_rate_arr[i] / 252.0)
        cash[i-1] += daily_interest

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
    is_uw = equity < rolling_max
    return round(is_uw.sum() / 21.0, 1)

def below_initial_months(equity, initial_capital):
    is_below = equity < initial_capital
    return round(is_below.sum() / 21.0, 1)

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
        sub        = eq.loc[common_idx]
        seg_initial = initial_capital
        years      = (sub.index[-1] - sub.index[0]).days / 365.25
        cagr       = (sub.iloc[-1]/sub.iloc[0])**(1/years)-1 if years>0 else np.nan
        uw_dd      = underwater_months(sub)
        uw_init    = below_initial_months(sub, seg_initial)
        row_all[f"{name}_CAGR"]    = cagr
        row_all[f"{name}_UW_DD"]   = uw_dd
        row_all[f"{name}_UW_INIT"] = uw_init
    rows.append(row_all)
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def run_all(start, rsi_period, left, right, min_range, max_range,
            expense, spread_2x, spread_3x, cost_one_way,
            sma_period, bear_sell_pct, bear_sell_basis,
            initial_capital, apply_tax_flag,
            tax_rate, tax_deduction,
            use_rsi_prebuy, rsi_prebuy_thresh, rsi_prebuy_pct,
            use_cash_rate, cash_rate_spread):            # ★ 추가
    df = load_data(start)
    df["px_1x"] = (1.0 + df["ret"]).cumprod()
    df["px_2x"] = make_leverage(df, 2, spread_2x)
    df["px_3x"] = make_leverage(df, 3, spread_3x)

    # ★ T-bill 금리 로드 & 일별 배열 생성
    tbill_raw  = load_tbill(start)
    daily_rate = build_daily_rate_array(df.index, tbill_raw,
                                        cash_rate_spread, use_cash_rate)
    # signal_df 인덱스에 맞는 Series
    daily_rate_s = pd.Series(daily_rate, index=df.index)

    _last_td = df["Close"].resample("W-FRI").apply(
        lambda x: x.index[-1] if len(x) > 0 else pd.NaT)
    wh_raw = df["High"].resample("W-FRI").max()
    wl_raw = df["Low"].resample("W-FRI").min()
    wc_raw = df["Close"].resample("W-FRI").last()

    def reindex_ltd(s):
        ni = _last_td.reindex(s.index)
        sc = s.copy(); sc.index = ni.values
        sc = sc[sc.index.notna()]; sc = sc[~sc.index.duplicated(keep="last")]
        return sc.sort_index()

    weekly_high  = reindex_ltd(wh_raw)
    weekly_low   = reindex_ltd(wl_raw)
    weekly_close = reindex_ltd(wc_raw)
    weekly_rsi   = calc_rsi(weekly_close, rsi_period)

    bull_confirm_w, bull_table = bullish_divergence(
        weekly_low, weekly_close, weekly_rsi, left, right, min_range, max_range)
    bear_confirm_w, bear_table = bearish_divergence(
        weekly_high, weekly_close, weekly_rsi, left, right, min_range, max_range)

    df["bull_event"] = weekly_to_daily_event(bull_confirm_w, df.index)
    df["bear_event"] = weekly_to_daily_event(bear_confirm_w, df.index)

    df["sma200"]       = df["Close"].rolling(sma_period, min_periods=sma_period).mean()
    df["sma_bull"]     = (df["Close"] > df["sma200"]).fillna(False)
    df["sma_cross_up"] = (df["sma_bull"]) & (~df["sma_bull"].shift(1).fillna(False))
    df["sma_cross_down"]= (~df["sma_bull"]) & (df["sma_bull"].shift(1).fillna(False))
    df["daily_rsi"] = calc_rsi(df["Close"], rsi_period)

    # ── 세후 결과
    res_div_1x, tr_div_1x = backtest(df["px_1x"], df, "bull_event","bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, apply_tax_flag,
        use_rsi_prebuy, rsi_prebuy_thresh, rsi_prebuy_pct, daily_rate_s)
    res_div_2x, tr_div_2x = backtest(df["px_2x"], df, "bull_event","bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, apply_tax_flag,
        use_rsi_prebuy, rsi_prebuy_thresh, rsi_prebuy_pct, daily_rate_s)
    res_div_3x, tr_div_3x = backtest(df["px_3x"], df, "bull_event","bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, apply_tax_flag,
        use_rsi_prebuy, rsi_prebuy_thresh, rsi_prebuy_pct, daily_rate_s)

    res_sma_1x, tr_sma_1x = backtest_sma200(df["px_1x"], df, initial_capital,
                                              apply_tax_flag, daily_rate_s)
    res_sma_2x, tr_sma_2x = backtest_sma200(df["px_2x"], df, initial_capital,
                                              apply_tax_flag, daily_rate_s)
    res_sma_3x, tr_sma_3x = backtest_sma200(df["px_3x"], df, initial_capital,
                                              apply_tax_flag, daily_rate_s)

    # ── 세전 결과
    res_div_1x_pre, _ = backtest(df["px_1x"], df, "bull_event","bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, False,
        use_rsi_prebuy, rsi_prebuy_thresh, rsi_prebuy_pct, daily_rate_s)
    res_div_2x_pre, _ = backtest(df["px_2x"], df, "bull_event","bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, False,
        use_rsi_prebuy, rsi_prebuy_thresh, rsi_prebuy_pct, daily_rate_s)
    res_div_3x_pre, _ = backtest(df["px_3x"], df, "bull_event","bear_event",
        initial_capital, bear_sell_pct, bear_sell_basis, False,
        use_rsi_prebuy, rsi_prebuy_thresh, rsi_prebuy_pct, daily_rate_s)

    res_sma_1x_pre, _ = backtest_sma200(df["px_1x"], df, initial_capital,
                                         False, daily_rate_s)
    res_sma_2x_pre, _ = backtest_sma200(df["px_2x"], df, initial_capital,
                                         False, daily_rate_s)
    res_sma_3x_pre, _ = backtest_sma200(df["px_3x"], df, initial_capital,
                                         False, daily_rate_s)

    bh_1x = initial_capital * df["px_1x"] / df["px_1x"].iloc[0]
    bh_2x = initial_capital * df["px_2x"] / df["px_2x"].iloc[0]
    bh_3x = initial_capital * df["px_3x"] / df["px_3x"].iloc[0]

    return (df, bull_table, bear_table,
            bh_1x, bh_2x, bh_3x,
            res_div_1x, res_div_2x, res_div_3x,
            res_sma_1x, res_sma_2x, res_sma_3x,
            tr_div_1x, tr_div_2x, tr_div_3x,
            tr_sma_1x, tr_sma_2x, tr_sma_3x,
            res_div_1x_pre, res_div_2x_pre, res_div_3x_pre,
            res_sma_1x_pre, res_sma_2x_pre, res_sma_3x_pre,
            daily_rate_s)   # ★ 반환에 추가

# ══════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════
with st.spinner("데이터 다운로드 & 백테스트 계산 중..."):
    (df, bull_table, bear_table,
     bh_1x, bh_2x, bh_3x,
     res_div_1x, res_div_2x, res_div_3x,
     res_sma_1x, res_sma_2x, res_sma_3x,
     tr_div_1x, tr_div_2x, tr_div_3x,
     tr_sma_1x, tr_sma_2x, tr_sma_3x,
     res_div_1x_pre, res_div_2x_pre, res_div_3x_pre,
     res_sma_1x_pre, res_sma_2x_pre, res_sma_3x_pre,
     daily_rate_s) = run_all(
        START, RSI_PERIOD, LEFT, RIGHT, MIN_RANGE, MAX_RANGE,
        EXPENSE, SPREAD_2X, SPREAD_3X, COST_ONE_WAY,
        SMA_PERIOD, BEAR_SELL_PCT, BEAR_SELL_BASIS,
        INITIAL_CAPITAL, apply_tax,
        TAX_RATE, TAX_DEDUCTION,
        USE_RSI_PREBUY, RSI_PREBUY_THRESH, RSI_PREBUY_PCT,
        USE_CASH_RATE, CASH_RATE_SPREAD)   # ★

# ══════════════════════════════════════════════
# 탭 구성
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 전략 비교", "📉 신호 차트", "🗓️ 10년 단위 분석",
    "📋 매매 기록", "📈 비중 & MDD", "💰 현금 이자율", "🔬 전진분석 (WFA)"
])

COLORS = {"BH":"#888780","Div":"#378ADD","SMA200":"#BA7517"}

# ──────────────────────────────────────────────
# 탭 1: 전략 비교
# ──────────────────────────────────────────────
with tab1:
    lev_choice = st.radio("레버리지", ["1x","2x","3x"], horizontal=True, key="lev1")
    lev_map = {
        "1x": (bh_1x, res_div_1x["equity"], res_sma_1x["equity"],
                       res_div_1x_pre["equity"], res_sma_1x_pre["equity"]),
        "2x": (bh_2x, res_div_2x["equity"], res_sma_2x["equity"],
                       res_div_2x_pre["equity"], res_sma_2x_pre["equity"]),
        "3x": (bh_3x, res_div_3x["equity"], res_sma_3x["equity"],
                       res_div_3x_pre["equity"], res_sma_3x_pre["equity"]),
    }
    bh_eq, div_eq, sma_eq, div_eq_pre, sma_eq_pre = lev_map[lev_choice]

    tax_label = "세후" if apply_tax else "세전"
    cash_label = f" +현금이자(T-bill-{CASH_RATE_SPREAD*100*100:.0f}bp)" if USE_CASH_RATE else ""
    summary_rows = [
        calc_stats(bh_eq,  f"BH {lev_choice}"),
        calc_stats(div_eq, f"Div {lev_choice} ({tax_label}{cash_label})"),
        calc_stats(sma_eq, f"SMA200 {lev_choice} ({tax_label}{cash_label})"),
    ]
    cols = st.columns(3)
    for col, row in zip(cols, summary_rows):
        with col:
            st.metric(row["전략"], row["최종자산"],
                      f"CAGR {row['CAGR']} | MDD {row['MDD']}")

    fig = go.Figure()
    for eq, name in [(bh_eq,"BH"),(div_eq,"Div"),(sma_eq,"SMA200")]:
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name,
            line=dict(color=COLORS[name], width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>"+name+"</extra>"))
    fig.update_yaxes(type="log", tickprefix="$", tickformat=",")
    fig.update_layout(title=f"누적 수익 비교 ({lev_choice}, 로그 스케일)",
                      hovermode="x unified", height=480,
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("성과 요약표 (세전 / 세후 비교)")
    all_stats = []
    for lev_label, bh_eq_l, div_pre, div_post, sma_pre, sma_post in [
        ("1x", bh_1x,
         res_div_1x_pre["equity"], res_div_1x["equity"],
         res_sma_1x_pre["equity"], res_sma_1x["equity"]),
        ("2x", bh_2x,
         res_div_2x_pre["equity"], res_div_2x["equity"],
         res_sma_2x_pre["equity"], res_sma_2x["equity"]),
        ("3x", bh_3x,
         res_div_3x_pre["equity"], res_div_3x["equity"],
         res_sma_3x_pre["equity"], res_sma_3x["equity"]),
    ]:
        all_stats.append(calc_stats(bh_eq_l,  f"BH {lev_label}"))
        all_stats.append(calc_stats(div_pre,  f"Div {lev_label} 세전"))
        all_stats.append(calc_stats(div_post, f"Div {lev_label} 세후"))
        all_stats.append(calc_stats(sma_pre,  f"SMA200 {lev_label} 세전"))
        all_stats.append(calc_stats(sma_post, f"SMA200 {lev_label} 세후"))
    stats_df = pd.DataFrame(all_stats)

    def highlight_rows(row):
        if "세후" in row["전략"]:
            return ["background-color: #1a2a3a"] * len(row)
        elif "세전" in row["전략"]:
            return ["background-color: #1a1a2a"] * len(row)
        else:
            return [""] * len(row)

    st.dataframe(stats_df.style.apply(highlight_rows, axis=1),
                 use_container_width=True, hide_index=True)

    st.subheader("세금 영향 (세전 → 세후 변화)")
    delta_rows = []
    for lev_label, div_pre, div_post, sma_pre, sma_post in [
        ("1x", res_div_1x_pre["equity"], res_div_1x["equity"],
                res_sma_1x_pre["equity"], res_sma_1x["equity"]),
        ("2x", res_div_2x_pre["equity"], res_div_2x["equity"],
                res_sma_2x_pre["equity"], res_sma_2x["equity"]),
        ("3x", res_div_3x_pre["equity"], res_div_3x["equity"],
                res_sma_3x_pre["equity"], res_sma_3x["equity"]),
    ]:
        for strat, pre_eq, post_eq in [
            (f"Div {lev_label}", div_pre, div_post),
            (f"SMA200 {lev_label}", sma_pre, sma_post),
        ]:
            pre_s  = calc_stats(pre_eq,  "pre")
            post_s = calc_stats(post_eq, "post")
            def pv(s): return float(s.replace("%","").replace("+",""))
            def uv(s): return float(s.replace("$","").replace(",",""))
            delta_rows.append({
                "전략": strat,
                "세전 CAGR":   pre_s["CAGR"],
                "세후 CAGR":   post_s["CAGR"],
                "CAGR 차이":   f"{pv(post_s['CAGR'])-pv(pre_s['CAGR']):+.2f}%p",
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
                    color = "#1a2e1a" if num >= 0 else "#2e1a1a"
                    styles[idx.index(col_name)] = f"background-color: {color}"
                except:
                    pass
        return styles

    delta_df = pd.DataFrame(delta_rows)
    st.dataframe(delta_df.style.apply(highlight_delta, axis=1),
                 use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# 탭 2: NDX + 신호 차트
# ──────────────────────────────────────────────
with tab2:
    date_range = st.slider("기간 선택",
        min_value=df.index[0].to_pydatetime(),
        max_value=df.index[-1].to_pydatetime(),
        value=(df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()),
        format="YYYY-MM-DD", key="date_slider")

    dff = df.loc[date_range[0]:date_range[1]]
    bull_dates = dff.index[dff["bull_event"]]
    bear_dates = dff.index[dff["bear_event"]]

    prebuy_dates = pd.DatetimeIndex([])
    if USE_RSI_PREBUY and len(tr_div_1x):
        pb_mask = tr_div_1x["signal"] == "rsi_prebuy"
        if pb_mask.any():
            pb_all = pd.DatetimeIndex(tr_div_1x.loc[pb_mask, "date"])
            prebuy_dates = pb_all[(pb_all >= date_range[0]) & (pb_all <= date_range[1])]

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.75, 0.25],
                         subplot_titles=["NDX 가격 + 신호", f"RSI ({RSI_PERIOD}, 일봉)"])

    fig2.add_trace(go.Scatter(x=dff.index, y=dff["Close"],
        name="NDX", line=dict(color="#B4B2A9", width=1),
        hovertemplate="%{x|%Y-%m-%d} %{y:,.0f}<extra>NDX</extra>"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=dff.index, y=dff["sma200"],
        name=f"SMA{SMA_PERIOD}", line=dict(color="#444441", width=1.2, dash="dot"),
        hovertemplate="%{y:,.0f}<extra>SMA</extra>"), row=1, col=1)

    if len(bull_dates):
        fig2.add_trace(go.Scatter(x=bull_dates, y=dff.loc[bull_dates,"Close"],
            mode="markers", name=f"Bull ({len(bull_dates)})",
            marker=dict(symbol="triangle-up", size=11, color="#1D9E75",
                        line=dict(color="#085041",width=1)),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra>Bull</extra>"), row=1, col=1)
    if len(bear_dates):
        fig2.add_trace(go.Scatter(x=bear_dates, y=dff.loc[bear_dates,"Close"],
            mode="markers", name=f"Bear ({len(bear_dates)})",
            marker=dict(symbol="triangle-down", size=11, color="#D85A30",
                        line=dict(color="#712B13",width=1)),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra>Bear</extra>"), row=1, col=1)

    if USE_RSI_PREBUY and len(prebuy_dates) > 0:
        fig2.add_trace(go.Scatter(
            x=prebuy_dates,
            y=dff.loc[prebuy_dates, "Close"] if len(prebuy_dates) > 0 else [],
            mode="markers",
            name=f"RSI 선제투입 ({len(prebuy_dates)})",
            marker=dict(symbol="circle", size=9, color="#F5A623",
                        line=dict(color="#A0630A", width=1)),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra>RSI선제투입</extra>"),
            row=1, col=1)

    fig2.add_trace(go.Scatter(x=dff.index, y=dff["daily_rsi"],
        name="RSI", line=dict(color="#7F77DD", width=1),
        hovertemplate="%{y:.1f}<extra>RSI</extra>"), row=2, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="#D85A30", row=2, col=1)
    if USE_RSI_PREBUY:
        fig2.add_hline(y=RSI_PREBUY_THRESH, line_dash="dot", line_color="#F5A623",
                       annotation_text=f"선제투입 {RSI_PREBUY_THRESH}",
                       annotation_position="right",
                       row=2, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="#1D9E75", row=2, col=1)

    fig2.update_yaxes(type="log", row=1, col=1)
    fig2.update_layout(height=600, hovermode="x unified",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Bull 다이버전스 ({len(bull_table)}건)")
        st.dataframe(bull_table, use_container_width=True, hide_index=True)
    with col2:
        st.subheader(f"Bear 다이버전스 ({len(bear_table)}건)")
        st.dataframe(bear_table, use_container_width=True, hide_index=True)

    if USE_RSI_PREBUY:
        pb_all_trades = tr_div_1x[tr_div_1x["signal"] == "rsi_prebuy"] if len(tr_div_1x) else pd.DataFrame()
        st.subheader(f"RSI 선제투입 ({len(pb_all_trades)}건, 1x 기준)")
        if len(pb_all_trades):
            st.dataframe(pb_all_trades, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# 탭 3: 10년 단위 분석
# ──────────────────────────────────────────────
with tab3:
    st.info("각 구간 시작 시점에 투자했을 때의 연환산 수익률(CAGR)과 원금손실기간(개월)입니다.")
    lev3 = st.radio("레버리지", ["1x","2x","3x"], horizontal=True, key="lev3")
    lev3_map = {
        "1x": (bh_1x, res_div_1x["equity"], res_sma_1x["equity"]),
        "2x": (bh_2x, res_div_2x["equity"], res_sma_2x["equity"]),
        "3x": (bh_3x, res_div_3x["equity"], res_sma_3x["equity"]),
    }
    _bh, _div, _sma = lev3_map[lev3]
    decade_df = decade_analysis({"BH": _bh, "Div": _div, "SMA200": _sma}, INITIAL_CAPITAL)
    decade_plot = decade_df[decade_df["구간"] != "전체"].copy()
    decade_all  = decade_df[decade_df["구간"] == "전체"].iloc[0]

    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1,c2,c3], ["BH","Div","SMA200"]):
        with col:
            cagr_val  = f"{decade_all[f'{name}_CAGR']*100:.2f}%"
            uw_dd_val = f"{decade_all[f'{name}_UW_DD']:.0f}개월"
            uw_in_val = f"{decade_all[f'{name}_UW_INIT']:.0f}개월"
            st.metric(f"{name} 전체 CAGR", cagr_val)
            st.caption(f"고점 미회복: {uw_dd_val}　|　원금 미회복: {uw_in_val}")

    st.divider()

    fig3a = go.Figure()
    for name, color in COLORS.items():
        fig3a.add_trace(go.Bar(
            x=decade_plot["구간"], y=decade_plot[f"{name}_CAGR"]*100,
            name=name, marker_color=color,
            hovertemplate="%{x}<br>CAGR: %{y:.2f}%<extra>"+name+"</extra>"))
    fig3a.add_hline(y=0, line_color="#888", line_width=0.8)
    fig3a.update_layout(title="10년 단위 CAGR (%)", barmode="group",
                        yaxis_ticksuffix="%", height=360,
                        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig3a, use_container_width=True)

    uw_type = st.radio(
        "원금손실기간 기준",
        ["시작원금 미회복 (진짜 손실 기간)", "고점 미회복 (drawdown 기간)"],
        horizontal=True, key="uw_type"
    )
    uw_col = "UW_INIT" if "시작원금" in uw_type else "UW_DD"
    uw_label = "시작원금 미회복 (개월)" if "시작원금" in uw_type else "고점 미회복 (개월)"

    fig3b = go.Figure()
    for name, color in COLORS.items():
        fig3b.add_trace(go.Bar(
            x=decade_plot["구간"], y=decade_plot[f"{name}_{uw_col}"],
            name=name, marker_color=color,
            hovertemplate="%{x}<br>%{y:.0f}개월<extra>"+name+"</extra>"))
    fig3b.update_layout(title=f"10년 단위 {uw_label}", barmode="group",
                        yaxis_ticksuffix="개월", height=360,
                        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig3b, use_container_width=True)

    st.caption("시작원금 미회복: 해당 구간 첫날 자산보다 낮은 날 수 → 개월 환산 (21거래일=1개월)\n\n"
               "고점 미회복: 직전 최고점을 회복하지 못한 날 수 → 개월 환산")

    display_df = decade_df.copy()
    for name in ["BH","Div","SMA200"]:
        display_df[f"{name} CAGR"] = display_df[f"{name}_CAGR"].apply(
            lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
        display_df[f"{name} 원금미회복"] = display_df[f"{name}_UW_INIT"].apply(
            lambda x: f"{x:.0f}개월" if pd.notna(x) else "N/A")
        display_df[f"{name} 고점미회복"] = display_df[f"{name}_UW_DD"].apply(
            lambda x: f"{x:.0f}개월" if pd.notna(x) else "N/A")
    show_cols = ["구간",
                 "BH CAGR","BH 원금미회복","BH 고점미회복",
                 "Div CAGR","Div 원금미회복","Div 고점미회복",
                 "SMA200 CAGR","SMA200 원금미회복","SMA200 고점미회복"]
    st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# 탭 4: 매매 기록
# ──────────────────────────────────────────────
with tab4:
    st.subheader("매매 기록")
    col_l, col_r = st.columns(2)
    with col_l:
        strat4 = st.selectbox("전략", ["Divergence","SMA200"])
    with col_r:
        lev4 = st.selectbox("레버리지", ["1x","2x","3x"])

    tr_map = {
        ("Divergence","1x"): tr_div_1x,
        ("Divergence","2x"): tr_div_2x,
        ("Divergence","3x"): tr_div_3x,
        ("SMA200","1x"):     tr_sma_1x,
        ("SMA200","2x"):     tr_sma_2x,
        ("SMA200","3x"):     tr_sma_3x,
    }
    trades = tr_map[(strat4, lev4)]
    if len(trades):
        sig_counts = trades["signal"].value_counts()
        cnt_cols = st.columns(len(sig_counts))
        sig_labels = {
            "bull_buy":   "🟢 Bull 매수",
            "bear_sell":  "🔴 Bear 매도",
            "rsi_prebuy": "🟠 RSI 선제투입",
            "sma_buy":    "🔵 SMA 매수",
            "sma_sell":   "🟡 SMA 매도",
        }
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
        "1x": (bh_1x, res_div_1x, res_sma_1x),
        "2x": (bh_2x, res_div_2x, res_sma_2x),
        "3x": (bh_3x, res_div_3x, res_sma_3x),
    }
    _bh_eq, _div_res, _sma_res = lev5_map[lev5]

    fig5a = go.Figure()
    for eq, name, color in [(_bh_eq,"BH",COLORS["BH"]),
                              (_div_res["equity"],"Div",COLORS["Div"]),
                              (_sma_res["equity"],"SMA200",COLORS["SMA200"])]:
        dd = (eq / eq.cummax() - 1.0) * 100
        fig5a.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name,
            fill="tozeroy", line=dict(color=color, width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>"+name+"</extra>"))
    fig5a.update_yaxes(ticksuffix="%")
    fig5a.update_layout(title=f"Drawdown ({lev5})", height=350,
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig5a, use_container_width=True)

    for res, label, color in [(_div_res,"Divergence",COLORS["Div"]),
                               (_sma_res,"SMA200",    COLORS["SMA200"])]:
        fig5b = go.Figure()
        fig5b.add_trace(go.Scatter(x=res.index, y=res["sw"]*100,
            name="주식", fill="tozeroy", line=dict(color=color, width=1)))
        fig5b.add_trace(go.Scatter(x=res.index, y=res["cw"]*100,
            name="현금", fill="tozeroy", line=dict(color="#B4B2A9", width=1)))
        fig5b.update_yaxes(ticksuffix="%", range=[0,100])
        fig5b.update_layout(title=f"{label} {lev5} — 주식/현금 비중",
                            height=280, hovermode="x unified",
                            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig5b, use_container_width=True)

# ──────────────────────────────────────────────
# 탭 6 (신규): 현금 이자율 시각화
# ──────────────────────────────────────────────
with tab6:
    st.subheader("💰 현금 이자율 (^IRX 13주 T-bill)")

    if not USE_CASH_RATE:
        st.warning("현금 이자 반영이 꺼져 있습니다. 사이드바에서 활성화하세요.")
    else:
        rate_pct = daily_rate_s * 100
        rate_pct = rate_pct[rate_pct > 0]

        if len(rate_pct) == 0:
            st.warning("금리 데이터 없음 — ^IRX 다운로드 실패 가능성. 잠시 후 새로고침 해보세요.")
        else:
            avg_rate = rate_pct.mean()
            max_rate = rate_pct.max()
            min_rate = rate_pct[rate_pct > 0].min()
            cur_rate = rate_pct.iloc[-1]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("현재 이자율 (조정 후)", f"{cur_rate:.2f}%")
            m2.metric("평균 이자율", f"{avg_rate:.2f}%")
            m3.metric("최고", f"{max_rate:.2f}%")
            m4.metric("최저 (>0)", f"{min_rate:.2f}%")

            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(
                x=rate_pct.index, y=rate_pct.values,
                name=f"T-bill 연이율 (스프레드 -{CASH_RATE_SPREAD*100*100:.0f}bp 차감)",
                line=dict(color="#378ADD", width=1.2),
                fill="tozeroy",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>현금이자율</extra>"))
            fig6.add_hline(y=avg_rate, line_dash="dot", line_color="#F5A623",
                           annotation_text=f"평균 {avg_rate:.2f}%",
                           annotation_position="right")
            fig6.update_yaxes(ticksuffix="%")
            fig6.update_layout(
                title="현금 적용 이자율 시계열 (연이율, 스프레드 차감 후)",
                height=380, hovermode="x unified",
                legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig6, use_container_width=True)

            # 현금 이자 누적 기여 추정 (Div 1x 기준)
            st.subheader("현금 이자 누적 기여 추정 (Div 1x 기준)")
            cash_arr = res_div_1x["cash"]
            rate_aligned = daily_rate_s.reindex(cash_arr.index).fillna(0.0)
            daily_interest_series = (cash_arr.shift(1).fillna(0) * rate_aligned / 252.0)
            cumulative_interest = daily_interest_series.cumsum()

            fig6b = go.Figure()
            fig6b.add_trace(go.Scatter(
                x=cumulative_interest.index, y=cumulative_interest.values,
                name="현금 이자 누적 기여",
                line=dict(color="#1D9E75", width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>누적이자</extra>"))
            fig6b.update_yaxes(tickprefix="$", tickformat=",")
            fig6b.update_layout(
                title="현금 이자 누적 기여액 (Div 1x)",
                height=320, hovermode="x unified")
            st.plotly_chart(fig6b, use_container_width=True)

            st.caption(
                f"^IRX (13주 T-bill) 기준. 스프레드 {CASH_RATE_SPREAD*100*100:.0f}bp 차감 적용.\n"
                "^IRX 시작(1990년대 초) 이전 구간은 이자율 0으로 처리됩니다."
            )

# ──────────────────────────────────────────────
# 탭 7: Walk-Forward Analysis (전진분석)
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_wfa(start, is_years, oos_years,
            expense, spread_2x, spread_3x, cost_one_way,
            bear_sell_pct, bear_sell_basis, initial_capital, apply_tax_flag,
            tax_rate, tax_deduction, lev_label,
            use_rsi_prebuy, rsi_prebuy_thresh, rsi_prebuy_pct,
            use_cash_rate, cash_rate_spread):             # ★ 추가
    df_raw = load_data(start)
    df_raw["px_1x"] = (1.0 + df_raw["ret"]).cumprod()
    df_raw["px_2x"] = make_leverage(df_raw, 2, spread_2x)
    df_raw["px_3x"] = make_leverage(df_raw, 3, spread_3x)
    px_col = {"1x": "px_1x", "2x": "px_2x", "3x": "px_3x"}[lev_label]

    tbill_raw    = load_tbill(start)
    daily_rate_g = build_daily_rate_array(df_raw.index, tbill_raw,
                                          cash_rate_spread, use_cash_rate)
    daily_rate_s_g = pd.Series(daily_rate_g, index=df_raw.index)

    param_grid = [
        {"rsi": r, "left": l, "right": ri, "min_r": mn, "max_r": mx}
        for r  in [10, 14, 18]
        for l  in [4, 5, 6]
        for ri in [2, 3]
        for mn in [5, 7]
        for mx in [50, 60]
    ]

    def build_signals(df_sub, p):
        _last_td = df_sub["Close"].resample("W-FRI").apply(
            lambda x: x.index[-1] if len(x) > 0 else pd.NaT)
        wl = df_sub["Low"].resample("W-FRI").min()
        wh = df_sub["High"].resample("W-FRI").max()
        wc = df_sub["Close"].resample("W-FRI").last()

        def rltd(s):
            ni = _last_td.reindex(s.index); sc = s.copy()
            sc.index = ni.values
            sc = sc[sc.index.notna()]; sc = sc[~sc.index.duplicated(keep="last")]
            return sc.sort_index()

        wl = rltd(wl); wh = rltd(wh); wc = rltd(wc)
        wr = calc_rsi(wc, p["rsi"])
        bull_w, _ = bullish_divergence(wl, wc, wr, p["left"], p["right"],
                                        p["min_r"], p["max_r"])
        bear_w, _ = bearish_divergence(wh, wc, wr, p["left"], p["right"],
                                        p["min_r"], p["max_r"])
        df_sub = df_sub.copy()
        df_sub["bull_event"] = weekly_to_daily_event(bull_w, df_sub.index)
        df_sub["bear_event"] = weekly_to_daily_event(bear_w, df_sub.index)
        df_sub["sma200"]     = df_sub["Close"].rolling(200, min_periods=200).mean()
        df_sub["sma_bull"]   = (df_sub["Close"] > df_sub["sma200"]).fillna(False)
        return df_sub

    def quick_cagr(equity):
        eq = equity.dropna()
        if len(eq) < 2 or eq.iloc[0] <= 0: return np.nan
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        return (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    all_dates   = df_raw.index
    is_td       = int(is_years  * 252)
    oos_td      = int(oos_years * 252)
    step        = oos_td

    wfa_rows    = []
    oos_equities= []
    window_no   = 0

    idx = 0
    while idx + is_td + oos_td <= len(all_dates):
        is_start  = all_dates[idx]
        is_end    = all_dates[idx + is_td - 1]
        oos_start = all_dates[idx + is_td]
        oos_end_i = min(idx + is_td + oos_td - 1, len(all_dates) - 1)
        oos_end   = all_dates[oos_end_i]

        df_is  = df_raw.loc[is_start:is_end].copy()
        df_oos = df_raw.loc[oos_start:oos_end].copy()
        if len(df_is) < 100 or len(df_oos) < 20:
            idx += step; continue

        dr_is  = daily_rate_s_g.reindex(df_is.index).fillna(0.0)
        dr_oos = daily_rate_s_g.reindex(df_oos.index).fillna(0.0)

        best_cagr_is = -np.inf
        best_p       = param_grid[0]
        for p in param_grid:
            try:
                df_is_sig = build_signals(df_is, p)
                res_is, _ = backtest(df_is[px_col], df_is_sig,
                                     "bull_event", "bear_event",
                                     initial_capital=100_000,
                                     bear_sell_pct=bear_sell_pct,
                                     bear_sell_basis=bear_sell_basis,
                                     _apply_tax=False,
                                     use_rsi_prebuy=use_rsi_prebuy,
                                     rsi_prebuy_thresh=rsi_prebuy_thresh,
                                     rsi_prebuy_pct=rsi_prebuy_pct,
                                     daily_rate=dr_is)
                c = quick_cagr(res_is["equity"])
                if pd.notna(c) and c > best_cagr_is:
                    best_cagr_is = c; best_p = p
            except Exception:
                continue

        bh_is_cagr = quick_cagr(
            initial_capital * df_is[px_col] / df_is[px_col].iloc[0])

        try:
            df_oos_sig   = build_signals(df_oos, best_p)
            res_oos, _   = backtest(df_oos[px_col], df_oos_sig,
                                    "bull_event", "bear_event",
                                    initial_capital=100_000,
                                    bear_sell_pct=bear_sell_pct,
                                    bear_sell_basis=bear_sell_basis,
                                    _apply_tax=apply_tax_flag,
                                    use_rsi_prebuy=use_rsi_prebuy,
                                    rsi_prebuy_thresh=rsi_prebuy_thresh,
                                    rsi_prebuy_pct=rsi_prebuy_pct,
                                    daily_rate=dr_oos)
            oos_cagr     = quick_cagr(res_oos["equity"])
            bh_oos_cagr  = quick_cagr(
                100_000 * df_oos[px_col] / df_oos[px_col].iloc[0])
            ef_ratio     = (oos_cagr / best_cagr_is) if best_cagr_is != 0 else np.nan
            oos_mdd      = (res_oos["equity"] / res_oos["equity"].cummax() - 1).min()
            bh_oos_mdd   = (100_000 * df_oos[px_col] / df_oos[px_col].iloc[0])
            bh_oos_mdd   = (bh_oos_mdd / bh_oos_mdd.cummax() - 1).min()
            oos_equities.append(res_oos["equity"])
        except Exception:
            oos_cagr = bh_oos_cagr = ef_ratio = oos_mdd = bh_oos_mdd = np.nan

        window_no += 1
        wfa_rows.append(dict(
            창번호      = window_no,
            IS_시작     = is_start.date(),
            IS_종료     = is_end.date(),
            OOS_시작    = oos_start.date(),
            OOS_종료    = oos_end.date(),
            최적_RSI    = best_p["rsi"],
            최적_Left   = best_p["left"],
            최적_Right  = best_p["right"],
            IS_CAGR     = best_cagr_is,
            BH_IS_CAGR  = bh_is_cagr,
            OOS_CAGR    = oos_cagr,
            BH_OOS_CAGR = bh_oos_cagr,
            효율비율    = ef_ratio,
            OOS_MDD     = oos_mdd,
            BH_OOS_MDD  = bh_oos_mdd,
        ))
        idx += step

    if oos_equities:
        combined_pieces = []
        scale = 1.0
        for piece in oos_equities:
            norm = piece / piece.iloc[0] * scale * initial_capital / 100_000
            combined_pieces.append(norm)
            scale = norm.iloc[-1] / initial_capital * 100_000
        oos_curve = pd.concat(combined_pieces)
    else:
        oos_curve = pd.Series(dtype=float)

    return pd.DataFrame(wfa_rows), oos_curve


with tab7:
    st.subheader("Walk-Forward Analysis (전진분석)")
    st.info(
        "In-Sample 구간에서 최적 파라미터를 찾고, "
        "Out-of-Sample 구간에서 그 파라미터로 실제 매매합니다. "
        "OOS 성과가 IS 성과에 근접할수록 전략이 과최적화되지 않은 것입니다."
    )

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        wfa_lev   = st.radio("레버리지", ["1x","2x","3x"], horizontal=True, key="wfa_lev")
    with col_w2:
        is_years  = st.slider("In-Sample 기간 (년)", 5, 15, 10)
    with col_w3:
        oos_years = st.slider("Out-of-Sample 기간 (년)", 1, 5, 3)

    if st.button("전진분석 실행", type="primary"):
        with st.spinner(f"WFA 계산 중... (IS {is_years}년 / OOS {oos_years}년, 파라미터 탐색 포함 — 1~3분 소요)"):
            wfa_df, oos_curve = run_wfa(
                START, is_years, oos_years,
                EXPENSE, SPREAD_2X, SPREAD_3X, COST_ONE_WAY,
                BEAR_SELL_PCT, BEAR_SELL_BASIS, INITIAL_CAPITAL, apply_tax,
                TAX_RATE, TAX_DEDUCTION, wfa_lev,
                USE_RSI_PREBUY, RSI_PREBUY_THRESH, RSI_PREBUY_PCT,
                USE_CASH_RATE, CASH_RATE_SPREAD)              # ★
        st.session_state["wfa_df"]    = wfa_df
        st.session_state["oos_curve"] = oos_curve
        st.session_state["wfa_lev"]   = wfa_lev

    if "wfa_df" in st.session_state and len(st.session_state["wfa_df"]):
        wfa_df    = st.session_state["wfa_df"]
        oos_curve = st.session_state["oos_curve"]
        wfa_lev   = st.session_state.get("wfa_lev", "2x")

        valid_rows   = wfa_df.dropna(subset=["OOS_CAGR","IS_CAGR"])
        beat_bh      = (valid_rows["OOS_CAGR"] > valid_rows["BH_OOS_CAGR"]).sum()
        beat_pct     = beat_bh / len(valid_rows) * 100 if len(valid_rows) else 0
        avg_ef       = valid_rows["효율비율"].mean()
        oos_positive = (valid_rows["OOS_CAGR"] > 0).sum()
        oos_pos_pct  = oos_positive / len(valid_rows) * 100 if len(valid_rows) else 0
        avg_oos_cagr = valid_rows["OOS_CAGR"].mean() * 100
        avg_is_cagr  = valid_rows["IS_CAGR"].mean()  * 100

        if beat_pct >= 60 and avg_ef >= 0.5 and oos_pos_pct >= 70:
            verdict = "✅ 전략 유효"
            verdict_color = "success"
            verdict_desc  = f"OOS 구간의 {beat_pct:.0f}%에서 BH를 초과, 효율비율 {avg_ef:.2f} — 과최적화 위험 낮음"
        elif beat_pct >= 40 and avg_ef >= 0.3:
            verdict = "⚠️ 조건부 유효"
            verdict_color = "warning"
            verdict_desc  = f"OOS 구간의 {beat_pct:.0f}%에서 BH를 초과, 효율비율 {avg_ef:.2f} — 파라미터 민감도 확인 필요"
        else:
            verdict = "❌ 전략 유효성 낮음"
            verdict_color = "error"
            verdict_desc  = f"OOS 구간의 {beat_pct:.0f}%에서만 BH 초과, 효율비율 {avg_ef:.2f} — 과최적화 가능성 높음"

        if verdict_color == "success":
            st.success(f"**{verdict}** — {verdict_desc}")
        elif verdict_color == "warning":
            st.warning(f"**{verdict}** — {verdict_desc}")
        else:
            st.error(f"**{verdict}** — {verdict_desc}")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("OOS 평균 CAGR",    f"{avg_oos_cagr:.2f}%",
                   f"IS 평균 {avg_is_cagr:.2f}%")
        mc2.metric("BH 초과 구간 비율", f"{beat_pct:.0f}%",
                   f"{beat_bh}/{len(valid_rows)} 구간")
        mc3.metric("효율 비율 (OOS/IS)", f"{avg_ef:.2f}",
                   "0.5 이상이면 양호")
        mc4.metric("OOS 수익 구간",     f"{oos_pos_pct:.0f}%",
                   f"{oos_positive}/{len(valid_rows)} 구간")

        st.divider()

        fig_wfa1 = go.Figure()
        x_labels = [f"창 {r['창번호']}\n{r['OOS_시작']}" for _, r in wfa_df.iterrows()]

        fig_wfa1.add_trace(go.Bar(
            x=x_labels, y=wfa_df["IS_CAGR"]*100,
            name="IS CAGR (최적화)", marker_color="#B5D4F4",
            hovertemplate="창 %{x}<br>IS CAGR: %{y:.2f}%<extra></extra>"))
        fig_wfa1.add_trace(go.Bar(
            x=x_labels, y=wfa_df["OOS_CAGR"]*100,
            name="OOS CAGR (실전)", marker_color="#378ADD",
            hovertemplate="창 %{x}<br>OOS CAGR: %{y:.2f}%<extra></extra>"))
        fig_wfa1.add_trace(go.Scatter(
            x=x_labels, y=wfa_df["BH_OOS_CAGR"]*100,
            name="BH OOS CAGR", mode="lines+markers",
            line=dict(color="#888780", dash="dot", width=1.5),
            hovertemplate="BH OOS: %{y:.2f}%<extra></extra>"))
        fig_wfa1.add_hline(y=0, line_color="gray", line_width=0.8)
        fig_wfa1.update_layout(
            title="창별 IS vs OOS CAGR 비교",
            barmode="group", yaxis_ticksuffix="%", height=400,
            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_wfa1, use_container_width=True)

        fig_wfa2 = go.Figure()
        colors_ef = ["#1D9E75" if v >= 0.5 else "#D85A30"
                     for v in wfa_df["효율비율"].fillna(0)]
        fig_wfa2.add_trace(go.Bar(
            x=x_labels, y=wfa_df["효율비율"],
            name="효율비율", marker_color=colors_ef,
            hovertemplate="효율비율: %{y:.2f}<extra></extra>"))
        fig_wfa2.add_hline(y=0.5, line_color="#1D9E75", line_dash="dash",
                           annotation_text="기준선 0.5", annotation_position="right")
        fig_wfa2.add_hline(y=1.0, line_color="#888", line_dash="dot",
                           annotation_text="IS=OOS", annotation_position="right")
        fig_wfa2.update_layout(
            title="효율비율 (OOS CAGR ÷ IS CAGR)",
            height=320, legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig_wfa2, use_container_width=True)

        if len(oos_curve) > 0:
            lev_px_col = {"1x":"px_1x","2x":"px_2x","3x":"px_3x"}[wfa_lev]
            bh_oos_full = (INITIAL_CAPITAL
                           * df[lev_px_col].reindex(oos_curve.index)
                           / df[lev_px_col].reindex(oos_curve.index).iloc[0])
            fig_wfa3 = go.Figure()
            fig_wfa3.add_trace(go.Scatter(
                x=bh_oos_full.index, y=bh_oos_full.values,
                name=f"BH {wfa_lev}", line=dict(color="#888780", width=1.2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>BH</extra>"))
            fig_wfa3.add_trace(go.Scatter(
                x=oos_curve.index, y=oos_curve.values,
                name="WFA OOS 전략", line=dict(color="#378ADD", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>WFA OOS</extra>"))
            fig_wfa3.update_yaxes(type="log", tickprefix="$", tickformat=",")
            fig_wfa3.update_layout(
                title="OOS 구간 연결 누적 수익 (실전 시뮬레이션, 로그 스케일)",
                height=420, hovermode="x unified",
                legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig_wfa3, use_container_width=True)

        st.subheader("최적 파라미터 안정성")
        st.caption("구간마다 다른 파라미터가 최적이면 전략이 불안정한 것입니다.")
        param_cols = st.columns(3)
        for col, param, label in zip(
            param_cols,
            ["최적_RSI","최적_Left","최적_Right"],
            ["RSI 기간","Left 봉수","Right 봉수"]
        ):
            vc = wfa_df[param].value_counts()
            col.write(f"**{label}** 분포")
            col.dataframe(vc.reset_index().rename(
                columns={param:"값","count":"횟수"}),
                hide_index=True, use_container_width=True)

        st.subheader("창별 상세 결과")
        disp = wfa_df.copy()
        for c in ["IS_CAGR","BH_IS_CAGR","OOS_CAGR","BH_OOS_CAGR"]:
            disp[c] = disp[c].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
        for c in ["OOS_MDD","BH_OOS_MDD"]:
            disp[c] = disp[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
        disp["효율비율"] = disp["효율비율"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        csv_wfa = wfa_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("WFA 결과 CSV 다운로드", csv_wfa,
                           file_name="wfa_result.csv", mime="text/csv")
    else:
        st.info("위에서 '전진분석 실행' 버튼을 누르면 계산이 시작됩니다. (1~3분 소요)")

st.divider()
st.caption(
    f"데이터: Yahoo Finance (^NDX, ^IRX) | 마지막 갱신: {df.index[-1].date()} | "
    f"세금 적용: {'예' if apply_tax else '아니오'} | "
    f"매도기준: {BEAR_SELL_BASIS} | "
    f"현금이자: {'ON' if USE_CASH_RATE else 'OFF'}"
    + (f" (T-bill -{CASH_RATE_SPREAD*100*100:.0f}bp)" if USE_CASH_RATE else "")
    + f" | RSI 선제투입: {'ON' if USE_RSI_PREBUY else 'OFF'}"
    + (f" (RSI<{RSI_PREBUY_THRESH}, 현금의 {RSI_PREBUY_PCT*100:.0f}%)" if USE_RSI_PREBUY else "")
)
