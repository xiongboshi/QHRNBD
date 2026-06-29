"""
PBX 瀑布线 + U形底部形态策略（三层时间框架版）
"""
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy
from utils.indicators import pbx, pbx_trend, detect_u_shape, atr, rsi


class PbxUShapeStrategy(BaseStrategy):
    """PBX瀑布线 + U形底部形态 日内波段策略（三层时间框架版）"""

    def __init__(self, broker, params: dict = None):
        default_params = {
            "pbx_periods": [4, 6, 9, 13, 18, 24],
            "pbx_trend_bars": 3,
            "lookback": 8,
            "recovery_ratio": 0.3,
            "min_drop_pct": 0.002,
            "u_shape_smooth": 3,
            "use_hour_filter": True,
            "hour_trend_bars": 3,
            "volume_confirm": False,
            "volume_factor": 1.2,
            "use_rsi_filter": False,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "pbx_deviation_max": 0.03,
            "use_atr_stop": True,
            "atr_period": 14,
            "stop_loss_atr": 1.5,
            "take_profit_atr": 3.0,
            "use_trailing_stop": False,
            "fixed_contracts": 1,
            "use_dynamic_sizing": False,
            "risk_per_trade": 0.005,
            "max_positions_per_day": 3,
            "min_holding_bars": 2,
        }
        default_params.update(params or {})
        super().__init__(broker, default_params)

        self._daily_trend = 0
        self._current_date = None
        self._daily_data = None
        self._hour_trend = 0
        self._hour_data = None
        self._entry_bar_index = -1
        self._entry_price = 0.0
        self._stop_price = 0.0
        self._target_price = 0.0
        self._entry_tag = ""
        self._filter_count = {
            "direction_mismatch": 0,
            "hour_filter": 0,
            "other": 0,
        }

    def prepare(self, df: pd.DataFrame):
        """预计算所有指标"""
        close = df["close"]
        high, low = df["high"], df["low"]
        volume = df["volume"]

        pbx_df = pbx(close, self.params["pbx_periods"])
        self.indicators["pbx"] = pbx_df
        self.indicators["pbx_trend"] = pbx_trend(pbx_df)

        pbx_4 = pbx_df.get("PBX_4", close)
        self.indicators["pbx_deviation"] = (close - pbx_4) / pbx_4

        self.indicators["u_shape"] = detect_u_shape(
            close,
            lookback=self.params["lookback"],
            recovery_ratio=self.params["recovery_ratio"],
        )

        self.indicators["atr"] = atr(df, self.params["atr_period"])

        if self.params["use_rsi_filter"]:
            self.indicators["rsi"] = rsi(close, 14)

        if self.params["volume_confirm"]:
            self.indicators["volume_ma"] = volume.rolling(window=20).mean()

        self._build_daily_data(df)

        if self.params["use_hour_filter"]:
            self._build_hour_data(df)

    def _build_daily_data(self, df: pd.DataFrame):
        """从分钟数据构建日线数据"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return

        daily_open = df["open"].resample("D").first()
        daily_high = df["high"].resample("D").max()
        daily_low = df["low"].resample("D").min()
        daily_close = df["close"].resample("D").last()

        self._daily_data = pd.DataFrame({
            "open": daily_open,
            "high": daily_high,
            "low": daily_low,
            "close": daily_close,
        }).dropna()

        if len(self._daily_data) > 10:
            daily_pbx = pbx(self._daily_data["close"], self.params["pbx_periods"])
            self._daily_data["pbx_trend"] = pbx_trend(daily_pbx)

    def _build_hour_data(self, df: pd.DataFrame):
        """从分钟数据构建1小时线数据"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return

        hourly_open = df["open"].resample("1h").first()
        hourly_high = df["high"].resample("1h").max()
        hourly_low = df["low"].resample("1h").min()
        hourly_close = df["close"].resample("1h").last()

        self._hour_data = pd.DataFrame({
            "open": hourly_open,
            "high": hourly_high,
            "low": hourly_low,
            "close": hourly_close,
        }).dropna()

        if len(self._hour_data) > 10:
            hour_pbx = pbx(self._hour_data["close"], self.params["pbx_periods"])
            self._hour_data["pbx_trend"] = pbx_trend(hour_pbx)

    def _get_daily_trend(self, current_time) -> int:
        """获取当前日线趋势方向"""
        if self._daily_data is None or len(self._daily_data) < 10:
            return 0

        current_date = current_time.date()
        daily_idx = self._daily_data.index.searchsorted(pd.Timestamp(current_date)) - 1

        if daily_idx < 0:
            return 0

        trend_series = self._daily_data["pbx_trend"]
        if daily_idx >= len(trend_series):
            daily_idx = len(trend_series) - 1

        trend = trend_series.iloc[daily_idx] if daily_idx >= 0 else 0

        if trend == 0 and daily_idx >= 3:
            recent_trends = trend_series.iloc[max(0, daily_idx-3):daily_idx+1]
            if len(recent_trends) > 0:
                trend = int(np.round(recent_trends.mean()))

        return int(trend) if trend in [1, -1] else 0

    def _get_hour_trend(self, current_time) -> int:
        """获取当前1小时线趋势方向"""
        if self._hour_data is None or len(self._hour_data) < 10:
            return 0

        hour_idx = self._hour_data.index.searchsorted(current_time) - 1

        if hour_idx < 0:
            return 0

        trend_series = self._hour_data["pbx_trend"]
        if hour_idx >= len(trend_series):
            hour_idx = len(trend_series) - 1

        trend = trend_series.iloc[hour_idx] if hour_idx >= 0 else 0

        confirm_bars = self.params.get("hour_trend_bars", 3)
        if trend != 0 and hour_idx >= confirm_bars:
            recent_trends = trend_series.iloc[max(0, hour_idx-confirm_bars+1):hour_idx+1]
            if len(recent_trends) > 0:
                same_direction_count = sum(1 for t in recent_trends if t == trend)
                if same_direction_count < confirm_bars * 0.6:
                    return 0

        return int(trend) if trend in [1, -1] else 0

    def on_bar(self, bar: pd.Series):
        """每个K线触发"""
        idx = self.current_index
        current_time = bar.name if hasattr(bar, 'name') else self.current_bar.name

        if hasattr(current_time, 'date'):
            today = current_time.date()
            if today != self._current_date:
                self._current_date = today
                self._daily_trend = self._get_daily_trend(current_time)
                self._hour_trend = self._get_hour_trend(current_time)

                direction_map = {1: "多头 (只做多)", -1: "空头 (只做空)", 0: "震荡 (不开仓)"}
                hour_map = {1: "上升", -1: "下降", 0: "震荡"}
                # if self._daily_trend != 0:
                #     print(
                #         f"[三层框架] {today} 日线:{direction_map.get(self._daily_trend, '未知')} | "
                #         f"1小时:{hour_map.get(self._hour_trend, '未知')}"
                #     )

        if self.params["use_hour_filter"]:
            self._hour_trend = self._get_hour_trend(current_time)

        if self.has_position:
            self._manage_position(bar, idx)
            return

        if self._daily_trend == 0:
            return

        if self.broker.daily_trade_count >= self.params["max_positions_per_day"]:
            return

        if self.params["use_hour_filter"]:
            if self._daily_trend == 1:
                if self._hour_trend == -1:
                    self._filter_count["hour_filter"] += 1
                    return
            else:
                if self._hour_trend == 1:
                    self._filter_count["hour_filter"] += 1
                    return

        signal = self._generate_signal(bar, idx)

        if signal == 1 and self._daily_trend == 1:
            volume = self._calc_position_size(bar, is_long=True)
            if volume > 0:
                self._open_long(bar, idx, volume)
        elif signal == -1 and self._daily_trend == -1:
            volume = self._calc_position_size(bar, is_long=False)
            if volume > 0:
                self._open_short(bar, idx, volume)
        else:
            if signal != 0:
                self._filter_count["direction_mismatch"] += 1

    def _generate_signal(self, bar: pd.Series, idx: int) -> int:
        """生成交易信号（15分钟级别）"""
        n = len(self.indicators.get("pbx_trend", pd.Series(dtype=float)))
        if idx < self.params["lookback"] * 2 + 5 or idx >= n:
            return 0

        close = bar["close"]
        volume = bar["volume"]
        u_signal = self.indicators["u_shape"]

        current_u = u_signal.iloc[idx]

        if current_u not in [1, -1]:
            return 0

        deviation = self.indicators["pbx_deviation"].iloc[idx]
        max_dev = self.params["pbx_deviation_max"]
        if abs(deviation) > max_dev:
            return 0

        if self.params["use_rsi_filter"] and "rsi" in self.indicators:
            current_rsi = self.indicators["rsi"].iloc[idx]
            if not pd.isna(current_rsi):
                if current_u == 1 and current_rsi > self.params["rsi_overbought"]:
                    return 0
                if current_u == -1 and current_rsi < self.params["rsi_oversold"]:
                    return 0

        if self.params["volume_confirm"] and "volume_ma" in self.indicators:
            vol_ma = self.indicators["volume_ma"].iloc[idx]
            if not pd.isna(vol_ma) and vol_ma > 0:
                if volume < vol_ma * self.params["volume_factor"]:
                    return 0

        return current_u

    def _calc_position_size(self, bar: pd.Series, is_long: bool) -> int:
        """计算仓位大小"""
        if not self.params["use_dynamic_sizing"]:
            return self.params["fixed_contracts"]

        close = bar["close"]
        atr_val = self.indicators.get("atr", pd.Series(dtype=float))
        current_atr = atr_val.iloc[self.current_index] if self.current_index < len(atr_val) else 0

        risk_amount = self.broker.capital * self.params["risk_per_trade"]
        atr_value = current_atr * 10

        if atr_value > 0:
            volume = max(1, int(risk_amount / atr_value))
        else:
            volume = self.params["fixed_contracts"]

        return min(volume, self.params.get("max_position_size", 20))

    def _open_long(self, bar: pd.Series, idx: int, volume: int):
        order = self.buy(volume)
        if order and order.is_filled:
            self._entry_bar_index = idx
            self._entry_price = order.fill_price
            self._entry_tag = "long"
            self._setup_stops(bar, is_long=True)
            self.signals_count += 1

    def _open_short(self, bar: pd.Series, idx: int, volume: int):
        order = self.sell(volume)
        if order and order.is_filled:
            self._entry_bar_index = idx
            self._entry_price = order.fill_price
            self._entry_tag = "short"
            self._setup_stops(bar, is_long=False)
            self.signals_count += 1

    def _setup_stops(self, bar: pd.Series, is_long: bool):
        atr_val = 0
        if self.params["use_atr_stop"] and "atr" in self.indicators:
            idx = self.current_index
            if idx < len(self.indicators["atr"]):
                atr_val = self.indicators["atr"].iloc[idx]
        atr_stop = max(atr_val * self.params["stop_loss_atr"], self._entry_price * 0.002)

        if is_long:
            self._stop_price = self._entry_price - atr_stop
            self._target_price = self._entry_price + atr_stop * (
                self.params["take_profit_atr"] / self.params["stop_loss_atr"]
            )
        else:
            self._stop_price = self._entry_price + atr_stop
            self._target_price = self._entry_price - atr_stop * (
                self.params["take_profit_atr"] / self.params["stop_loss_atr"]
            )

    def _manage_position(self, bar: pd.Series, idx: int):
        holding_bars = idx - self._entry_bar_index
        close = bar["close"]
        high, low = bar["high"], bar["low"]
        is_long = self._entry_tag == "long"

        if holding_bars < self.params["min_holding_bars"]:
            return

        if is_long:
            unrealized_pnl_pct = (close - self._entry_price) / self._entry_price
            if unrealized_pnl_pct >= 0.01:
                self._stop_price = max(self._stop_price, self._entry_price * 0.998)
            if low <= self._stop_price:
                self.close_all()
                return
            if high >= self._target_price:
                self.close_all()
                return
        else:
            unrealized_pnl_pct = (self._entry_price - close) / self._entry_price
            if unrealized_pnl_pct >= 0.01:
                self._stop_price = min(self._stop_price, self._entry_price * 1.002)
            if high >= self._stop_price:
                self.close_all()
                return
            if low <= self._target_price:
                self.close_all()
                return

    def on_end(self):
        super().on_end()
        print(f"[过滤统计] 方向不匹配: {self._filter_count['direction_mismatch']}次, "
              f"1小时过滤: {self._filter_count['hour_filter']}次")