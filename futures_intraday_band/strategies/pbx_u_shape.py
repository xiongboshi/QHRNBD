"""
PBX 瀑布线 + U形形态策略（水平支撑假跌破入场做空）
基于波浪理论 + K线验证的MA5/MA10方向判断

核心逻辑：
1. 取最近60根K线，判断每根K线收盘价与MA5的关系
2. K线连续在MA5同侧运行 >= 5根 → 构成"一浪"
3. 用K线验证MA5方向，而不是只看交叉点
4. 判断当前价格在浪中的位置 → 确认趋势方向
"""
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy
from utils.indicators import pbx, pbx_trend, detect_u_shape, atr, rsi


class PbxUShapeStrategy(BaseStrategy):
    """PBX瀑布线 + U形形态策略（波浪理论 + K线验证方向）"""

    def __init__(self, broker, params: dict = None):
        default_params = {
            "pbx_periods": [4, 6, 9, 13, 18, 24],
            "pbx_trend_bars": 100,
            "lookback": 8,
            "recovery_ratio": 0.3,
            "min_drop_pct": 0.002,
            "u_shape_smooth": 3,
            "use_hour_filter": False,
            "hour_trend_bars": 100,
            "sweet_zone_tolerance": 0.02,
            "break_confirm_bars": 1,
            "volume_confirm": False,
            "volume_factor": 1.2,
            "use_rsi_filter": False,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "pbx_deviation_max": 0.08,
            "use_atr_stop": True,
            "atr_period": 14,
            "stop_loss_atr": 0.8,
            "take_profit_atr": 3.0,
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

        self._horizontal_level = None
        self._support_start_idx = 0
        self._support_detected = False
        self._support_lines_history = []

        self._break_state = {
            "active": False,
            "level": None,
            "break_bar": None,
        }
        self._bars_cache = []

        self._entry_bar_index = -1
        self._entry_bar = None
        self._entry_price = 0.0
        self._stop_price = 0.0
        self._target_price = 0.0
        self._entry_tag = ""

        self._filter_count = {
            "hour_filter": 0,
            "no_support": 0,
            "not_broken": 0,
            "no_rebound": 0,
        }
        self._trades_count = 0

    def prepare(self, df: pd.DataFrame):
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

        self._build_daily_data(df)
        if self.params["use_hour_filter"]:
            self._build_hour_data(df)

    def _build_daily_data(self, df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        daily_open = df["open"].resample("D").first()
        daily_high = df["high"].resample("D").max()
        daily_low = df["low"].resample("D").min()
        daily_close = df["close"].resample("D").last()
        self._daily_data = pd.DataFrame({
            "open": daily_open, "high": daily_high,
            "low": daily_low, "close": daily_close,
        }).dropna()
        if len(self._daily_data) > 10:
            daily_pbx = pbx(self._daily_data["close"], self.params["pbx_periods"])
            self._daily_data["pbx_trend"] = pbx_trend(daily_pbx)

    def _build_hour_data(self, df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        hourly_open = df["open"].resample("1h").first()
        hourly_high = df["high"].resample("1h").max()
        hourly_low = df["low"].resample("1h").min()
        hourly_close = df["close"].resample("1h").last()
        self._hour_data = pd.DataFrame({
            "open": hourly_open, "high": hourly_high,
            "low": hourly_low, "close": hourly_close,
        }).dropna()
        if len(self._hour_data) > 10:
            hour_pbx = pbx(self._hour_data["close"], self.params["pbx_periods"])
            self._hour_data["pbx_trend"] = pbx_trend(hour_pbx)
            self.indicators["hour_pbx"] = hour_pbx

    # ==========================================================
    # 核心方法：日线方向判断（波浪理论 + K线验证）
    # ==========================================================

    def _get_daily_trend(self, current_time) -> int:
        """
        获取日线趋势方向（波浪理论 + K线验证）
        
        核心逻辑：
        1. 取最近60根K线，判断每根K线收盘价与MA5的关系
        2. K线连续在MA5同侧运行 >= 5根 → 构成"一浪"
        3. 用K线验证MA5方向，而不是只看交叉点
        4. 判断当前价格在浪中的位置 → 确认趋势方向
        
        返回: 1=多头, -1=空头, 0=震荡
        """
        if self._daily_data is None or len(self._daily_data) < 60:
            return 0

        current_date = current_time.date()
        daily_idx = self._daily_data.index.searchsorted(pd.Timestamp(current_date)) - 1
        if daily_idx < 60:
            return 0

        close = self._daily_data['close']
        high = self._daily_data['high']
        low = self._daily_data['low']
        
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        
        # ==========================================================
        # 1. 提取最近60根K线，判断每根K线与MA5的关系
        # ==========================================================
        start_idx = max(0, daily_idx - 60)
        close_slice = close.iloc[start_idx:daily_idx+1]
        ma5_slice = ma5.iloc[start_idx:daily_idx+1]
        high_slice = high.iloc[start_idx:daily_idx+1]
        low_slice = low.iloc[start_idx:daily_idx+1]
        
        # 判断每根K线收盘价在MA5上方还是下方
        above_ma5 = close_slice > ma5_slice
        below_ma5 = close_slice < ma5_slice
        
        # ==========================================================
        # 2. 找出连续的趋势段（真正的浪）
        # ==========================================================
        # 连续在MA5上方的K线数量 >= 5根 → 上升浪
        # 连续在MA5下方的K线数量 >= 5根 → 下跌浪
        # 少于5根视为震荡，不构成浪
        
        current_streak = 0
        current_streak_direction = 0  # 1=上升, -1=下跌
        waves = []  # 存储所有识别到的浪
        
        for i in range(len(above_ma5)):
            if above_ma5.iloc[i]:
                if current_streak_direction == 1:
                    current_streak += 1
                else:
                    # 前一个浪结束
                    if current_streak >= 5 and current_streak_direction != 0:
                        waves.append({
                            "direction": current_streak_direction,
                            "length": current_streak,
                            "start_idx": i - current_streak,
                            "end_idx": i - 1
                        })
                    current_streak = 1
                    current_streak_direction = 1
            else:
                if current_streak_direction == -1:
                    current_streak += 1
                else:
                    if current_streak >= 5 and current_streak_direction != 0:
                        waves.append({
                            "direction": current_streak_direction,
                            "length": current_streak,
                            "start_idx": i - current_streak,
                            "end_idx": i - 1
                        })
                    current_streak = 1
                    current_streak_direction = -1
        
        # 处理最后一浪
        if current_streak >= 5 and current_streak_direction != 0:
            waves.append({
                "direction": current_streak_direction,
                "length": current_streak,
                "start_idx": len(above_ma5) - current_streak,
                "end_idx": len(above_ma5) - 1
            })
        
        # ==========================================================
        # 3. 如果没找到完整的浪，用MA位置判断
        # ==========================================================
        if len(waves) == 0:
            current_close = close_slice.iloc[-1]
            current_ma5 = ma5_slice.iloc[-1]
            if current_close > current_ma5:
                return 1
            else:
                return -1
        
        # ==========================================================
        # 4. 找到最近的一浪和前一浪
        # ==========================================================
        last_wave = waves[-1]
        prev_wave = waves[-2] if len(waves) >= 2 else None
        
        current_close = close_slice.iloc[-1]
        current_high = high_slice.iloc[-1]
        current_low = low_slice.iloc[-1]
        
        # 计算最后一浪的幅度和价格区间
        wave_start_idx = start_idx + last_wave["start_idx"]
        wave_end_idx = start_idx + last_wave["end_idx"]
        
        wave_high = high.iloc[wave_start_idx:wave_end_idx+1].max()
        wave_low = low.iloc[wave_start_idx:wave_end_idx+1].min()
        wave_mid = (wave_high + wave_low) / 2
        
        # 计算MA5斜率
        ma5_start = ma5.iloc[wave_start_idx] if wave_start_idx > 0 else ma5.iloc[0]
        ma5_end = ma5.iloc[wave_end_idx]
        ma5_slope = ma5_end - ma5_start
        
        # ==========================================================
        # 5. 判断当前浪的方向和位置
        # ==========================================================
        
        # 最后一浪是上升浪（K线在MA5上方连续运行）
        if last_wave["direction"] == 1:
            # 如果MA5还在上升，且当前价格在MA5上方 → 多头延续
            if ma5_slope > 0 and current_close > ma5.iloc[daily_idx]:
                return 1
            
            # 如果价格跌破MA10 → 可能转空
            if current_close < ma10.iloc[daily_idx]:
                return -1
            
            # 检查是否突破前高（确认多头强度）
            prev_high = high.iloc[max(0, wave_start_idx-10):wave_start_idx].max() if wave_start_idx > 10 else wave_high
            if current_high > prev_high * 1.01:
                return 1
            else:
                if current_close < wave_mid:
                    return -1
                else:
                    return 1
        
        # 最后一浪是下跌浪（K线在MA5下方连续运行）
        if last_wave["direction"] == -1:
            # MA5还在下降，且当前价格在MA5下方 → 空头延续
            if ma5_slope < 0 and current_close < ma5.iloc[daily_idx]:
                return -1
            
            # 如果价格突破MA10 → 可能转多
            if current_close > ma10.iloc[daily_idx]:
                return 1
            
            # 检查是否跌破前低（确认空头强度）
            prev_low = low.iloc[max(0, wave_start_idx-10):wave_start_idx].min() if wave_start_idx > 10 else wave_low
            if current_low < prev_low * 0.99:
                return -1
            else:
                if current_close > wave_mid:
                    return 1
                else:
                    return -1
        
        return 0

    # ==========================================================
    # 核心方法：1小时线方向判断（波浪理论 + K线验证）
    # ==========================================================

    def _get_hour_trend(self, current_time) -> int:
        """
        获取1小时线趋势方向（波浪理论 + K线验证）
        逻辑与日线一致，但周期更短
        """
        if self._hour_data is None or len(self._hour_data) < 60:
            return 0

        hour_idx = self._hour_data.index.searchsorted(current_time) - 1
        if hour_idx < 60:
            return 0

        close = self._hour_data['close']
        high = self._hour_data['high']
        low = self._hour_data['low']
        
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        
        start_idx = max(0, hour_idx - 60)
        close_slice = close.iloc[start_idx:hour_idx+1]
        ma5_slice = ma5.iloc[start_idx:hour_idx+1]
        high_slice = high.iloc[start_idx:hour_idx+1]
        low_slice = low.iloc[start_idx:hour_idx+1]
        
        above_ma5 = close_slice > ma5_slice
        below_ma5 = close_slice < ma5_slice
        
        current_streak = 0
        current_streak_direction = 0
        waves = []
        
        for i in range(len(above_ma5)):
            if above_ma5.iloc[i]:
                if current_streak_direction == 1:
                    current_streak += 1
                else:
                    if current_streak >= 5 and current_streak_direction != 0:
                        waves.append({
                            "direction": current_streak_direction,
                            "length": current_streak,
                            "start_idx": i - current_streak,
                            "end_idx": i - 1
                        })
                    current_streak = 1
                    current_streak_direction = 1
            else:
                if current_streak_direction == -1:
                    current_streak += 1
                else:
                    if current_streak >= 5 and current_streak_direction != 0:
                        waves.append({
                            "direction": current_streak_direction,
                            "length": current_streak,
                            "start_idx": i - current_streak,
                            "end_idx": i - 1
                        })
                    current_streak = 1
                    current_streak_direction = -1
        
        if current_streak >= 5 and current_streak_direction != 0:
            waves.append({
                "direction": current_streak_direction,
                "length": current_streak,
                "start_idx": len(above_ma5) - current_streak,
                "end_idx": len(above_ma5) - 1
            })
        
        if len(waves) == 0:
            current_close = close_slice.iloc[-1]
            current_ma5 = ma5_slice.iloc[-1]
            if current_close > current_ma5:
                return 1
            else:
                return -1
        
        last_wave = waves[-1]
        
        current_close = close_slice.iloc[-1]
        current_high = high_slice.iloc[-1]
        current_low = low_slice.iloc[-1]
        
        wave_start_idx = start_idx + last_wave["start_idx"]
        wave_end_idx = start_idx + last_wave["end_idx"]
        
        wave_high = high.iloc[wave_start_idx:wave_end_idx+1].max()
        wave_low = low.iloc[wave_start_idx:wave_end_idx+1].min()
        wave_mid = (wave_high + wave_low) / 2
        
        ma5_start = ma5.iloc[wave_start_idx] if wave_start_idx > 0 else ma5.iloc[0]
        ma5_end = ma5.iloc[wave_end_idx]
        ma5_slope = ma5_end - ma5_start
        
        if last_wave["direction"] == 1:
            if ma5_slope > 0 and current_close > ma5.iloc[hour_idx]:
                return 1
            if current_close < ma10.iloc[hour_idx]:
                return -1
            prev_high = high.iloc[max(0, wave_start_idx-10):wave_start_idx].max() if wave_start_idx > 10 else wave_high
            if current_high > prev_high * 1.01:
                return 1
            else:
                if current_close < wave_mid:
                    return -1
                else:
                    return 1
        
        if last_wave["direction"] == -1:
            if ma5_slope < 0 and current_close < ma5.iloc[hour_idx]:
                return -1
            if current_close > ma10.iloc[hour_idx]:
                return 1
            prev_low = low.iloc[max(0, wave_start_idx-10):wave_start_idx].min() if wave_start_idx > 10 else wave_low
            if current_low < prev_low * 0.99:
                return -1
            else:
                if current_close > wave_mid:
                    return 1
                else:
                    return -1
        
        return 0

    # ==========================================================
    # 核心：识别水平支撑
    # ==========================================================

    def _find_horizontal_support(self) -> float:
        """
        找到水平支撑（横盘区间低点）
        """
        df = pd.DataFrame(self._bars_cache)
        if len(df) < 10:
            return None

        recent_bars = min(30, len(df))
        min_low = df['low'].iloc[-recent_bars:].min()
        min_idx_local = df['low'].iloc[-recent_bars:].idxmin()
        min_pos = df.index.get_loc(min_idx_local)
        
        low_range = min_low * 1.005
        nearby_lows = df['low'].iloc[-recent_bars:][df['low'].iloc[-recent_bars:] < low_range]
        
        if len(nearby_lows) >= 2:
            self._support_start_idx = min_pos
            support_time = df.index[min_pos] if min_pos < len(df.index) else None
            for existing in self._support_lines_history:
                if abs(existing["price"] - min_low) / min_low < 0.01:
                    return None
            self._support_lines_history.append({
                "type": "horizontal",
                "price": float(min_low),
                "start_idx": int(min_pos),
                "time": str(support_time) if support_time else None,
            })
            return min_low
        
        return None

    # ==========================================================
    # 画线数据
    # ==========================================================

    def get_support_lines(self):
        """获取所有识别到的支撑线数据"""
        seen = set()
        unique_lines = []
        for line in self._support_lines_history:
            key = (line["price"], line["start_idx"])
            if key not in seen:
                seen.add(key)
                unique_lines.append(line)
        return unique_lines

    # ==========================================================
    # 主逻辑
    # ==========================================================

    def on_bar(self, bar: pd.Series):
        idx = self.current_index
        current_time = bar.name if hasattr(bar, 'name') else self.current_bar.name

        if hasattr(current_time, 'date'):
            today = current_time.date()
            if today != self._current_date:
                self._current_date = today
                self._daily_trend = self._get_daily_trend(current_time)
                self._hour_trend = self._get_hour_trend(current_time)
                self._break_state["active"] = False
                self._horizontal_level = None
                direction_map = {1: "多头", -1: "空头", 0: "震荡"}
                print(f"[方向] {today} 日线: {direction_map.get(self._daily_trend, '未知')} | 1小时: {direction_map.get(self._hour_trend, '未知')}")

        if self.params["use_hour_filter"]:
            self._hour_trend = self._get_hour_trend(current_time)

        if self.has_position:
            self._manage_position(bar, idx)
            return

        self._bars_cache.append(bar)
        if len(self._bars_cache) > 60:
            self._bars_cache = self._bars_cache[-60:]

        close = bar["close"]
        tolerance = self.params["sweet_zone_tolerance"]

        if self._horizontal_level is None:
            level = self._find_horizontal_support()
            if level is not None:
                self._horizontal_level = level
                print(f"[支撑] {current_time} 识别水平支撑: {level:.1f}")

        if self._daily_trend != -1:
            return

        if self.params["use_hour_filter"] and self._hour_trend != -1:
            self._filter_count["hour_filter"] += 1
            return

        if self.broker.daily_trade_count >= self.params["max_positions_per_day"]:
            return

        if self._horizontal_level is None:
            self._filter_count["no_support"] += 1
            return

        level = self._horizontal_level

        if not self._break_state["active"]:
            if close < level * (1 - tolerance):
                self._break_state["active"] = True
                self._break_state["level"] = level
                self._break_state["break_bar"] = bar
                print(f"[假跌破] {current_time} 跌破支撑 {level:.1f}，当前价 {close:.1f}")
            return

        if close > level * (1 + tolerance * 3):
            self._break_state["active"] = False
            return

        if close < level and close > level * (1 - tolerance * 2):
            volume = self._calc_position_size(bar, is_long=False)
            if volume > 0:
                self._open_short(bar, idx, volume)
                self._trades_count += 1
                print(f"[做空] {current_time} 反弹不过支撑 {level:.1f}，入场 @ {close:.1f}")
                self._break_state["active"] = False

    def _calc_position_size(self, bar: pd.Series, is_long: bool) -> int:
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

    def _open_short(self, bar: pd.Series, idx: int, volume: int):
        order = self.sell(volume)
        if order and order.is_filled:
            self._entry_bar_index = idx
            self._entry_bar = bar
            self._entry_price = order.fill_price
            self._entry_tag = "short"
            self._setup_stops(bar)
            self.signals_count += 1

    def _setup_stops(self, bar: pd.Series):
        atr_val = 0
        if "atr" in self.indicators:
            idx = self.current_index
            if idx < len(self.indicators["atr"]):
                atr_val = self.indicators["atr"].iloc[idx]
        
        if atr_val <= 0:
            atr_val = 15
        
        self._stop_price = self._entry_price + atr_val * self.params["stop_loss_atr"]
        self._target_price = self._entry_price - atr_val * self.params["take_profit_atr"]
        print(f"[止损/止盈] 入场: {self._entry_price:.1f} | 止损: {self._stop_price:.1f} | 止盈: {self._target_price:.1f}")

    def _manage_position(self, bar: pd.Series, idx: int):
        holding_bars = idx - self._entry_bar_index
        high, low = bar["high"], bar["low"]

        if holding_bars < self.params["min_holding_bars"]:
            return

        if self._entry_tag == "short":
            if high >= self._stop_price:
                print(f"[止损] {bar.name} 空头止损 @ {self._stop_price:.1f}")
                self.close_all()
                return
            if low <= self._target_price:
                print(f"[止盈] {bar.name} 空头止盈 @ {self._target_price:.1f}")
                self.close_all()
                return

    def on_end(self):
        super().on_end()
        print(f"[过滤统计] 1小时过滤: {self._filter_count['hour_filter']}次, "
              f"无支撑: {self._filter_count['no_support']}次")
        print(f"[交易统计] 共 {self._trades_count} 笔交易")
        print(f"[支撑记录] 共识别 {len(self._support_lines_history)} 条支撑线")