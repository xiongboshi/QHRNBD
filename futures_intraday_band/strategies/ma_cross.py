"""
均线交叉策略（示例策略）

核心思路：
1. 快线（EMA5）上穿慢线（EMA20）时做多
2. 快线下穿慢线时平多/做空
3. 日内交易，收盘前平仓

演示目的：展示如何快速添加一个新策略
"""
import pandas as pd

from strategies.base_strategy import BaseStrategy
from utils.indicators import ema, atr


class MaCrossStrategy(BaseStrategy):
    """均线交叉日内策略"""

    def __init__(self, broker, params: dict = None):
        default_params = {
            "fast_period": 5,              # 快线周期
            "slow_period": 20,             # 慢线周期
            "use_filter_trend": True,      # 是否使用长期趋势过滤
            "trend_period": 60,            # 长期趋势周期
            "stop_loss_atr": 1.5,          # ATR 止损倍数
            "take_profit_atr": 3.0,        # ATR 止盈倍数
            "max_positions_per_day": 3,    # 每日最大交易次数
            "fixed_contracts": 1,          # 固定交易手数
        }
        default_params.update(params or {})
        super().__init__(broker, default_params)

        # 状态
        self._prev_fast = 0.0
        self._prev_slow = 0.0
        self._entry_price = 0.0
        self._entry_bar_index = -1
        self._stop_price = 0.0
        self._target_price = 0.0

    def prepare(self, df: pd.DataFrame):
        """预计算指标"""
        close = df["close"]
        self.indicators["ema_fast"] = ema(close, self.params["fast_period"])
        self.indicators["ema_slow"] = ema(close, self.params["slow_period"])
        self.indicators["atr"] = atr(df, 14)

        if self.params["use_filter_trend"]:
            self.indicators["trend_ema"] = ema(close, self.params["trend_period"])

    def on_bar(self, bar: pd.Series):
        """每个K线触发"""
        idx = self.current_index
        close = bar["close"]
        high, low = bar["high"], bar["low"]

        # 获取指标值
        fast = self.indicators["ema_fast"]
        slow = self.indicators["ema_slow"]
        current_atr = self.indicators["atr"]

        if idx < 1 or idx >= len(fast):
            return

        current_fast = fast.iloc[idx]
        current_slow = slow.iloc[idx]

        # ======================================
        # 持仓管理
        # ======================================
        if self.has_position:
            self._manage_position(bar, idx)
            return

        # ======================================
        # 开仓信号
        # ======================================
        if self.broker.daily_trade_count >= self.params["max_positions_per_day"]:
            return

        # 趋势过滤
        if self.params["use_filter_trend"] and "trend_ema" in self.indicators:
            trend_up = close > self.indicators["trend_ema"].iloc[idx]
        else:
            trend_up = True

        # 金叉（快线上穿慢线）
        if self._prev_fast <= self._prev_slow and current_fast > current_slow:
            if trend_up:
                self._open_long(bar, idx, current_atr.iloc[idx] if idx < len(current_atr) else 0)

        # 死叉（快线下穿慢线）
        elif self._prev_fast >= self._prev_slow and current_fast < current_slow:
            if not trend_up:
                self._open_short(bar, idx, current_atr.iloc[idx] if idx < len(current_atr) else 0)

        # 更新前值
        self._prev_fast = current_fast
        self._prev_slow = current_slow

    def _open_long(self, bar: pd.Series, idx: int, atr_val: float):
        """开多"""
        price = bar["close"]
        volume = self.params["fixed_contracts"]
        order = self.buy(volume)
        if order and order.is_filled:
            self._entry_bar_index = idx
            self._entry_price = order.fill_price
            atr_stop = max(atr_val, self._entry_price * 0.002) * self.params["stop_loss_atr"]
            self._stop_price = self._entry_price - atr_stop
            self._target_price = self._entry_price + atr_stop * (self.params["take_profit_atr"] / self.params["stop_loss_atr"])
            self.signals_count += 1

    def _open_short(self, bar: pd.Series, idx: int, atr_val: float):
        """开空"""
        price = bar["close"]
        volume = self.params["fixed_contracts"]
        order = self.sell(volume)
        if order and order.is_filled:
            self._entry_bar_index = idx
            self._entry_price = order.fill_price
            atr_stop = max(atr_val, self._entry_price * 0.002) * self.params["stop_loss_atr"]
            self._stop_price = self._entry_price + atr_stop
            self._target_price = self._entry_price - atr_stop * (self.params["take_profit_atr"] / self.params["stop_loss_atr"])
            self.signals_count += 1

    def _manage_position(self, bar: pd.Series, idx: int):
        """管理持仓"""
        holding_bars = idx - self._entry_bar_index
        close = bar["close"]
        high, low = bar["high"], bar["low"]
        is_long = self.broker.position.volume > 0

        if holding_bars < 1:
            return

        if is_long:
            if low <= self._stop_price or high >= self._target_price:
                self.close_all()
        else:
            if high >= self._stop_price or low <= self._target_price:
                self.close_all()
