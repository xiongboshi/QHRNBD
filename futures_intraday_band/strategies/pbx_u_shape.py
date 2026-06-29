"""
PBX 瀑布线 + U形底部形态策略（三层时间框架版）

核心逻辑：
1. 【日线定方向】日线PBX排列决定今日做单大方向（多头/空头/震荡）
2. 【1小时线过滤】1小时线趋势与日线同向时才允许开仓（避免回调段入场）
3. 【15分钟入场】U形形态确认 + 多重过滤

三层框架优势：
- 日线：过滤逆势交易
- 1小时：过滤回调段，提高入场质量
- 15分钟：精确捕捉入场点
"""
import pandas as pd
import numpy as np

from strategies.base_strategy import BaseStrategy
from utils.indicators import pbx, pbx_trend, detect_u_shape, atr, rsi


class PbxUShapeStrategy(BaseStrategy):
    """PBX瀑布线 + U形底部形态 日内波段策略（三层时间框架版）"""

    def __init__(self, broker, params: dict = None):
        default_params = {
            # --- PBX 参数 ---
            "pbx_periods": [4, 6, 9, 13, 18, 24],   # PBX 周期
            "pbx_trend_bars": 3,                     # 趋势确认持续K线数

            # --- U形形态参数 ---
            "lookback": 8,                           # U形回溯窗口
            "recovery_ratio": 0.3,                   # 反弹幅度比例阈值
            "min_drop_pct": 0.002,                   # 最小左半边跌幅 0.2%
            "u_shape_smooth": 3,                     # 形态平滑窗口

            # --- 1小时线过滤参数 ---
            "use_hour_filter": True,                 # 是否启用1小时线过滤
            "hour_trend_bars": 3,                    # 1小时趋势确认K线数

            # --- 过滤器参数 ---
            "volume_confirm": False,                 # 是否用成交量确认
            "volume_factor": 1.2,                    # 成交量放大倍数
            "use_rsi_filter": False,                 # RSI极端值过滤
            "rsi_oversold": 30,                      # 超卖阈值（多头）
            "rsi_overbought": 70,                    # 超买阈值（空头）
            "pbx_deviation_max": 0.03,               # 价格与PBX_4最大偏离 3%

            # --- 止损止盈 ---
            "use_atr_stop": True,                    # ATR止损
            "atr_period": 14,                        # ATR周期
            "stop_loss_atr": 1.5,                    # 止损ATR倍数
            "take_profit_atr": 3.0,                  # 止盈ATR倍数
            "use_trailing_stop": False,              # 启用追踪止损

            # --- 仓位管理 ---
            "fixed_contracts": 1,                    # 固定手数
            "use_dynamic_sizing": False,             # 动态仓位
            "risk_per_trade": 0.005,                 # 每笔风险资金比例 0.5%
            "max_positions_per_day": 3,              # 每日最大交易次数
            "min_holding_bars": 2,                   # 最小持仓K线数
        }
        default_params.update(params or {})
        super().__init__(broker, default_params)

        # ===== 日线趋势状态 =====
        self._daily_trend = 0        # 1=多头, -1=空头, 0=震荡
        self._current_date = None    # 当前交易日
        self._daily_data = None      # 日线数据缓存

        # ===== 1小时线趋势状态 =====
        self._hour_trend = 0         # 1小时线趋势
        self._hour_data = None       # 1小时数据缓存

        # 持仓状态
        self._entry_bar_index = -1
        self._entry_price = 0.0
        self._stop_price = 0.0
        self._target_price = 0.0
        self._entry_tag = ""  # "long" / "short"

        # 日志：记录过滤次数
        self._filter_count = {
            "direction_mismatch": 0,   # 信号与日线方向不匹配
            "hour_filter": 0,          # 1小时线过滤
            "other": 0,
        }

    def prepare(self, df: pd.DataFrame):
        """预计算所有指标"""
        close = df["close"]
        high, low = df["high"], df["low"]
        volume = df["volume"]

        # 1. PBX 瀑布线（15分钟级别）
        pbx_df = pbx(close, self.params["pbx_periods"])
        self.indicators["pbx"] = pbx_df
        self.indicators["pbx_trend"] = pbx_trend(pbx_df)

        # 2. PBX 短期乖离率
        pbx_4 = pbx_df.get("PBX_4", close)
        self.indicators["pbx_deviation"] = (close - pbx_4) / pbx_4

        # 3. U 形形态（15分钟级别）
        self.indicators["u_shape"] = detect_u_shape(
            close,
            lookback=self.params["lookback"],
            recovery_ratio=self.params["recovery_ratio"],
        )

        # 4. ATR
        self.indicators["atr"] = atr(df, self.params["atr_period"])

        # 5. RSI
        if self.params["use_rsi_filter"]:
            self.indicators["rsi"] = rsi(close, 14)

        # 6. 成交量均线
        if self.params["volume_confirm"]:
            self.indicators["volume_ma"] = volume.rolling(window=20).mean()

        # 7. 构建日线数据（用于定方向）
        self._build_daily_data(df)

        # 8. 构建1小时数据（用于过滤）
        if self.params["use_hour_filter"]:
            self._build_hour_data(df)

    def _build_daily_data(self, df: pd.DataFrame):
        """从分钟数据构建日线数据"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        
        # 按日期分组，取每日OHLC
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
        
        # 计算日线PBX
        if len(self._daily_data) > 10:
            daily_pbx = pbx(self._daily_data["close"], self.params["pbx_periods"])
            self._daily_data["pbx_trend"] = pbx_trend(daily_pbx)

    def _build_hour_data(self, df: pd.DataFrame):
        """从分钟数据构建1小时线数据"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        
        # 按1小时分组
        hourly_open = df["open"].resample("1H").first()
        hourly_high = df["high"].resample("1H").max()
        hourly_low = df["low"].resample("1H").min()
        hourly_close = df["close"].resample("1H").last()
        
        self._hour_data = pd.DataFrame({
            "open": hourly_open,
            "high": hourly_high,
            "low": hourly_low,
            "close": hourly_close,
        }).dropna()
        
        # 计算1小时线PBX
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
        
        # 如果趋势为0，取最近3天的平均值
        if trend == 0 and daily_idx >= 3:
            recent_trends = trend_series.iloc[max(0, daily_idx-3):daily_idx+1]
            if len(recent_trends) > 0:
                trend = int(np.round(recent_trends.mean()))
        
        return int(trend) if trend in [1, -1] else 0

    def _get_hour_trend(self, current_time) -> int:
        """获取当前1小时线趋势方向"""
        if self._hour_data is None or len(self._hour_data) < 10:
            return 0
        
        # 找到当前时间之前最近的1小时线
        hour_idx = self._hour_data.index.searchsorted(current_time) - 1
        
        if hour_idx < 0:
            return 0
        
        trend_series = self._hour_data["pbx_trend"]
        if hour_idx >= len(trend_series):
            hour_idx = len(trend_series) - 1
        
        trend = trend_series.iloc[hour_idx] if hour_idx >= 0 else 0
        
        # 取最近N根1小时线的趋势确认
        confirm_bars = self.params.get("hour_trend_bars", 3)
        if trend != 0 and hour_idx >= confirm_bars:
            recent_trends = trend_series.iloc[max(0, hour_idx-confirm_bars+1):hour_idx+1]
            if len(recent_trends) > 0:
                # 需要大部分K线同向才确认
                same_direction_count = sum(1 for t in recent_trends if t == trend)
                if same_direction_count < confirm_bars * 0.6:
                    return 0  # 趋势不够强，视为震荡
        
        return int(trend) if trend in [1, -1] else 0

    def on_bar(self, bar: pd.Series):
        """每个K线触发"""
        idx = self.current_index
        close = bar["close"]
        current_time = bar.name if hasattr(bar, 'name') else self.current_bar.name

        # ======================================
        # 每日更新日线趋势和1小时趋势
        # ======================================
        if hasattr(current_time, 'date'):
            today = current_time.date()
            if today != self._current_date:
                self._current_date = today
                self._daily_trend = self._get_daily_trend(current_time)
                self._hour_trend = self._get_hour_trend(current_time)
                
                # 打印当日方向
                direction_map = {1: "多头 (只做多)", -1: "空头 (只做空)", 0: "震荡 (不开仓)"}
                hour_map = {1: "上升", -1: "下降", 0: "震荡"}
                if self._daily_trend != 0:
                    self.broker.logger.info(
                        f"[三层框架] {today} 日线:{direction_map.get(self._daily_trend, '未知')} | "
                        f"1小时:{hour_map.get(self._hour_trend, '未知')}"
                    )

        # 更新1小时趋势（盘中变化）
        if self.params["use_hour_filter"]:
            self._hour_trend = self._get_hour_trend(current_time)

        # ======================================
        # 持仓管理（止损止盈）
        # ======================================
        if self.has_position:
            self._manage_position(bar, idx)
            return

        # ======================================
        # 开仓信号
        # ======================================
        # 条件1：必须有日线方向
        if self._daily_trend == 0:
            return  # 震荡市不开仓

        # 条件2：每日交易次数限制
        if self.broker.daily_trade_count >= self.params["max_positions_per_day"]:
            return

        # ======================================
        # 1小时线过滤（核心新增逻辑）
        # ======================================
        if self.params["use_hour_filter"]:
            # 日线多头时，1小时线也必须多头或刚从空头转为多头
            # 日线空头时，1小时线也必须空头或刚从多头转为空头
            if self._daily_trend == 1:
                # 日线多头：只允许1小时线为多头或刚从空头转多头
                if self._hour_trend == -1:
                    self._filter_count["hour_filter"] += 1
                    return  # 1小时线空头，不开仓（等待回调结束）
            else:  # self._daily_trend == -1
                # 日线空头：只允许1小时线为空头或刚从多头转空头
                if self._hour_trend == 1:
                    self._filter_count["hour_filter"] += 1
                    return  # 1小时线多头，不开仓（等待反弹结束）

        # ======================================
        # 生成15分钟信号
        # ======================================
        signal = self._generate_signal(bar, idx)
        
        # 信号必须与日线方向一致
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

        # --- 条件过滤 ---

        # 1. U形信号过滤
        if current_u not in [1, -1]:
            return 0

        # 2. PBX乖离率过滤（防止追高/杀跌）
        deviation = self.indicators["pbx_deviation"].iloc[idx]
        max_dev = self.params["pbx_deviation_max"]
        if abs(deviation) > max_dev:
            return 0

        # 3. RSI过滤
        if self.params["use_rsi_filter"] and "rsi" in self.indicators:
            current_rsi = self.indicators["rsi"].iloc[idx]
            if not pd.isna(current_rsi):
                if current_u == 1 and current_rsi > self.params["rsi_overbought"]:
                    return 0
                if current_u == -1 and current_rsi < self.params["rsi_oversold"]:
                    return 0

        # 4. 成交量确认
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
        """开多"""
        order = self.buy(volume)
        if order and order.is_filled:
            self._entry_bar_index = idx
            self._entry_price = order.fill_price
            self._entry_tag = "long"
            self._setup_stops(bar, is_long=True)
            self.signals_count += 1

    def _open_short(self, bar: pd.Series, idx: int, volume: int):
        """开空"""
        order = self.sell(volume)
        if order and order.is_filled:
            self._entry_bar_index = idx
            self._entry_price = order.fill_price
            self._entry_tag = "short"
            self._setup_stops(bar, is_long=False)
            self.signals_count += 1

    def _setup_stops(self, bar: pd.Series, is_long: bool):
        """设置止损止盈"""
        atr_val = 0
        if self.params["use_atr_stop"] and "atr" in self.indicators:
            idx = self.current_index
            if idx < len(self.indicators["atr"]):
                atr_val = self.indicators["atr"].iloc[idx]
        atr_stop = max(atr_val * self.params["stop_loss_atr"],
                       self._entry_price * 0.002)

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
        """管理持仓"""
        holding_bars = idx - self._entry_bar_index
        close = bar["close"]
        high, low = bar["high"], bar["low"]
        is_long = self._entry_tag == "long"

        if holding_bars < self.params["min_holding_bars"]:
            return

        # 移动止盈
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
        """回测结束，打印过滤统计"""
        super().on_end()
        if self.broker.logger:
            self.broker.logger.info(
                f"[过滤统计] 方向不匹配: {self._filter_count['direction_mismatch']}次, "
                f"1小时过滤: {self._filter_count['hour_filter']}次"
            )