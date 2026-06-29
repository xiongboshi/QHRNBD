"""
技术指标计算模块
包含 PBX（瀑布线）、ATR、EMA、布林带等常用指标
所有函数均以 pandas Series/DataFrame 作为输入/输出
"""
import numpy as np
import pandas as pd


# ==========================================================
# 基础指标
# ==========================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均线"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均线"""
    return series.rolling(window=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """加权移动平均线（线性权重）"""
    weights = np.arange(1, period + 1)
    def _calc(window):
        if len(window) < period:
            return np.nan
        return np.dot(window, weights) / weights.sum()
    return series.rolling(window=period).apply(_calc, raw=True)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标 RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真实波幅 ATR

    Args:
        df: 必须包含 high, low, close 列
        period: 计算周期
    Returns:
        ATR 序列
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    """布林带

    Returns:
        (中轨, 上轨, 下轨)
    """
    mid = sma(series, period)
    std = series.rolling(window=period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD 指标

    Returns:
        (diff, dea, macd_hist)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    diff = ema_fast - ema_slow
    dea = ema(diff, signal)
    macd_hist = 2 * (diff - dea)
    return diff, dea, macd_hist


# ==========================================================
# PBX 瀑布线
# ==========================================================

def pbx(close: pd.Series, periods: list = None) -> pd.DataFrame:
    """PBX 瀑布线（Polarized Fractal Efficiency）

    计算多周期 WMA 并统一到同一 DataFrame

    Args:
        close: 收盘价序列
        periods: WMA 周期列表，默认 [4, 6, 9, 13, 18, 24]

    Returns:
        DataFrame，每列对应一个周期的 PBX 值
    """
    if periods is None:
        periods = [4, 6, 9, 13, 18, 24]
    pbx_dict = {}
    for p in periods:
        pbx_dict[f"PBX_{p}"] = wma(close, p)
    return pd.DataFrame(pbx_dict, index=close.index)


def pbx_trend(pbx_df: pd.DataFrame, strict: bool = False) -> pd.Series:
    """判断 PBX 瀑布线趋势

    Args:
        pbx_df: pbx() 返回的 DataFrame
        strict: True=要求严格排列（所有周期都满足）；False=多数排列即可（宽松模式）
    Returns:
        Series: 1=多头排列（上涨趋势）, -1=空头排列（下跌趋势）, 0=震荡
    """
    # 按周期排序
    cols = sorted(pbx_df.columns, key=lambda x: int(x.split("_")[1]))

    if strict:
        # 严格：所有短期 > 所有长期
        bullish = (pbx_df[cols[0]] > pbx_df[cols[1]]) & \
                  (pbx_df[cols[1]] > pbx_df[cols[2]]) & \
                  (pbx_df[cols[3]] > pbx_df[cols[4]]) & \
                  (pbx_df[cols[4]] > pbx_df[cols[5]])
        bearish = (pbx_df[cols[0]] < pbx_df[cols[1]]) & \
                  (pbx_df[cols[1]] < pbx_df[cols[2]]) & \
                  (pbx_df[cols[3]] < pbx_df[cols[4]]) & \
                  (pbx_df[cols[4]] < pbx_df[cols[5]])
    else:
        # 宽松：最短周期 > 最长周期 即为多头趋势
        bullish = pbx_df[cols[0]] > pbx_df[cols[-1]]
        bearish = pbx_df[cols[0]] < pbx_df[cols[-1]]

    trend = pd.Series(0, index=pbx_df.index)
    trend[bullish] = 1
    trend[bearish] = -1
    return trend


# ==========================================================
# U 形形态识别
# ==========================================================

def detect_u_shape(close: pd.Series, lookback: int = 10, recovery_ratio: float = 0.5,
                   min_move_pct: float = 0.003) -> pd.Series:
    """检测 U 形底部形态

    形态特征：
    1. 先下跌（左半边）：价格从高点回落
    2. 形成底部：价格达到局部最低点
    3. 后回升（右半边）：价格从低点反弹 recovery_ratio 以上

    Args:
        close: 收盘价序列
        lookback: 回溯窗口（K线数量）
        recovery_ratio: 从底部反弹的幅度比例（相对于左半边跌幅）
        min_move_pct: 最小波动比例（用于过滤噪声）

    Returns:
        Series: 1=检测到U形买点, -1=倒U形卖点, 0=无信号
    """
    signals = pd.Series(0, index=close.index)

    for i in range(lookback * 2, len(close)):
        window = close.iloc[i - lookback * 2: i + 1]
        mid = lookback  # 窗口中间位置
        left_high = window.iloc[:mid].max()
        left_low = window.iloc[:mid].min()
        bottom = window.iloc[mid:mid+2].min()  # 底部区域
        right_price = window.iloc[-1]

        # U形买点：左半边跌幅明显，右半边从底部明显反弹
        left_drop = (left_high - left_low) / left_high  # 左半边跌幅比例
        right_recovery = (right_price - bottom) / bottom  # 右半边反弹比例

        if left_drop > min_move_pct and right_recovery > recovery_ratio * left_drop:
            signals.iloc[i] = 1

        # 倒U形卖点（顶形）：先涨后跌
        right_high = window.iloc[mid:].max()
        right_drop = (right_high - right_price) / right_high
        left_rise = (window.iloc[:mid].max() - window.iloc[0]) / window.iloc[0]

        if left_rise > min_move_pct and right_drop > recovery_ratio * left_rise:
            signals.iloc[i] = -1

    return signals


# ==========================================================
# 自定义日内辅助指标
# ==========================================================

def volume_weighted_price(df: pd.DataFrame) -> pd.Series:
    """VWAP 成交量加权平均价"""
    return (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()


def volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """历史波动率（对数收益率标准差）"""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=period).std()


def price_channels(high: pd.Series, low: pd.Series, period: int = 20):
    """价格通道（唐奇安通道）

    Returns:
        (上轨, 中轨, 下轨)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    mid = (upper + lower) / 2
    return upper, mid, lower
