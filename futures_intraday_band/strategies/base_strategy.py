"""
策略基类
所有具体策略都必须继承 BaseStrategy 并实现 on_bar() 方法
"""
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from engine.broker import Broker, OrderSide, OrderType


class BaseStrategy(ABC):
    """策略基类 - 所有策略的父类"""

    def __init__(self, broker: Broker, params: dict = None):
        """
        Args:
            broker: Broker 实例（由引擎注入）
            params: 策略参数（从配置文件读取）
        """
        self.broker = broker
        self.params = params or {}
        self.name = self.__class__.__name__

        # 预计算指标缓存（由 prepare() 填充）
        self.indicators: dict[str, pd.Series | pd.DataFrame] = {}

        # 当前K线数据
        self.current_bar: Optional[pd.Series] = None
        self.current_index: int = 0

        # 交易统计（策略级别）
        self.signals_count = 0
        self.trades_count = 0

    def __str__(self):
        return f"{self.name}(params={self.params})"

    @abstractmethod
    def on_bar(self, bar: pd.Series) -> None:
        """每个K线触发一次 - 策略核心逻辑

        Args:
            bar: 当前K线数据（包含 open, high, low, close, volume）
        """
        pass

    def prepare(self, df: pd.DataFrame) -> None:
        """数据准备阶段（回测开始前调用）
        用于预计算指标，避免 on_bar 中重复计算

        Args:
            df: 全量历史数据
        """
        pass

    def on_start(self) -> None:
        """回测启动时回调"""
        pass

    def on_end(self) -> None:
        """回测结束时回调"""
        pass

    # ==========================================
    # 快捷交易方法
    # ==========================================

    def buy(self, volume: int = 1, price: float = None, order_type: OrderType = OrderType.MARKET):
        """开多 / 平空"""
        return self.broker.place_order(
            side=OrderSide.BUY,
            volume=volume,
            price=price,
            order_type=order_type,
            symbol=self._get_symbol(),
        )

    def sell(self, volume: int = 1, price: float = None, order_type: OrderType = OrderType.MARKET):
        """开空 / 平多"""
        return self.broker.place_order(
            side=OrderSide.SELL,
            volume=volume,
            price=price,
            order_type=order_type,
            symbol=self._get_symbol(),
        )

    def close_all(self):
        """全部平仓"""
        return self.broker.close_position()

    @property
    def has_position(self) -> bool:
        """是否持仓"""
        return not self.broker.position.is_flat

    @property
    def position_direction(self) -> str:
        return self.broker.position.direction

    def _get_symbol(self) -> str:
        """获取当前品种代码"""
        if self.broker.position.symbol:
            return self.broker.position.symbol
        # 从数据中推断
        return getattr(self, "_symbol", "unknown")
