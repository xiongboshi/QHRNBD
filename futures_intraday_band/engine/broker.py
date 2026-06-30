"""
模拟经纪商
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    id: int
    time: pd.Timestamp
    symbol: str
    side: OrderSide
    price: float
    volume: int
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_time: Optional[pd.Timestamp] = None
    commission: float = 0.0
    reject_reason: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_rejected(self) -> bool:
        return self.status == OrderStatus.CANCELLED


@dataclass
class Position:
    symbol: str
    volume: int = 0
    avg_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.volume == 0

    @property
    def direction(self) -> str:
        if self.volume > 0:
            return "long"
        elif self.volume < 0:
            return "short"
        return "flat"


@dataclass
class TradeRecord:
    time: pd.Timestamp
    symbol: str
    side: str
    price: float
    volume: int
    commission: float
    pnl: float = 0.0
    tags: list = field(default_factory=list)

    def __str__(self):
        action = "开多" if (self.side == "buy" and self.pnl == 0) else \
                 "平多" if (self.side == "sell" and self.pnl != 0) else \
                 "开空" if (self.side == "sell" and self.pnl == 0) else "平空"
        tag_str = f" [{','.join(self.tags)}]" if self.tags else ""
        return (f"[{self.time.strftime('%m-%d %H:%M')}] {self.symbol} {action} "
                f"{self.volume}手 @ {self.price:.1f} | "
                f"手续费:{self.commission:.2f} | 盈亏:{self.pnl:.2f}{tag_str}")


class Broker:
    """模拟经纪商"""

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_config: dict = None,
        contract_config: dict = None,
        slippage_config: dict = None,
        risk_config: dict = None,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_config = commission_config or {}
        self.contract_config = contract_config or {}
        self.slippage_config = slippage_config or {"mode": "fixed", "fixed_ticks": 1}

        self.position = Position(symbol="")
        self._order_id = 0
        self.trades: list[TradeRecord] = []
        self.all_orders: list[Order] = []

        self._current_bar: Optional[pd.Series] = None
        self.daily_trade_count = 0
        self.equity_curve: list[float] = [initial_capital]

    def set_current_bar(self, bar: pd.Series):
        self._current_bar = bar

    def _get_symbol_key(self, symbol: str) -> str:
        symbol_map = {'rb888': 'rb', 'i888': 'i', 'MA888': 'MA'}
        return symbol_map.get(symbol, symbol)

    @property
    def equity(self) -> float:
        if self.position.is_flat or self._current_bar is None:
            return self.capital

        current_price = self._current_bar['close']
        symbol_key = self._get_symbol_key(self.position.symbol or 'rb888')
        contract = self.contract_config.get(symbol_key, {})
        multiplier = contract.get("multiplier", 10)

        if self.position.volume > 0:
            unrealized = self.position.volume * (current_price - self.position.avg_price) * multiplier
        else:
            unrealized = abs(self.position.volume) * (self.position.avg_price - current_price) * multiplier

        return self.capital + unrealized

    def get_contract_value(self, price: float, volume: int, symbol: str) -> float:
        symbol_key = self._get_symbol_key(symbol)
        contract = self.contract_config.get(symbol_key, {})
        multiplier = contract.get("multiplier", 10)
        return price * multiplier * volume

    def calculate_commission(self, price: float, volume: int, side: str, symbol: str) -> float:
        symbol_key = self._get_symbol_key(symbol)
        comm = self.commission_config.get(symbol_key, {})
        multiplier = self.contract_config.get(symbol_key, {}).get("multiplier", 10)
        turnover = price * multiplier * abs(volume)
        ratio = comm.get("open_ratio" if side == "buy" else "close_today_ratio", 0.0001)
        return max(turnover * ratio, 0.01)

    def _next_order_id(self) -> int:
        self._order_id += 1
        return self._order_id

    def place_order(
        self,
        side: OrderSide,
        volume: int,
        symbol: str = "",
        tags: list = None,
    ) -> Optional[Order]:
        if volume <= 0 or self._current_bar is None:
            return None

        symbol = symbol or self.position.symbol or 'rb888'
        price = self._current_bar['close']
        commission = self.calculate_commission(price, volume, side.value, symbol)

        order = Order(
            id=self._next_order_id(),
            time=self._current_bar.name,
            symbol=symbol,
            side=side,
            price=price,
            volume=volume,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            fill_price=price,
            fill_time=self._current_bar.name,
            commission=commission,
        )
        self.all_orders.append(order)

        is_open = (side == OrderSide.BUY and self.position.volume >= 0) or \
                  (side == OrderSide.SELL and self.position.volume <= 0)

        if is_open:
            self.capital -= commission

            if self.position.is_flat:
                self.position.symbol = symbol
                self.position.volume = volume if side == OrderSide.BUY else -volume
                self.position.avg_price = price
            else:
                total_vol = abs(self.position.volume) + volume
                total_cost = abs(self.position.volume) * self.position.avg_price + volume * price
                self.position.avg_price = total_cost / total_vol
                if side == OrderSide.BUY:
                    self.position.volume += volume
                else:
                    self.position.volume -= volume

            self.trades.append(TradeRecord(
                time=order.fill_time, symbol=symbol, side=side.value,
                price=price, volume=volume, commission=commission,
                tags=tags or [],
            ))

        else:
            symbol_key = self._get_symbol_key(symbol)
            contract = self.contract_config.get(symbol_key, {})
            multiplier = contract.get("multiplier", 10)

            if self.position.volume > 0:
                pnl = volume * (price - self.position.avg_price) * multiplier
                self.position.volume -= volume
            else:
                pnl = volume * (self.position.avg_price - price) * multiplier
                self.position.volume += volume

            self.capital -= commission
            self.capital += pnl

            self.trades.append(TradeRecord(
                time=order.fill_time, symbol=symbol, side=side.value,
                price=price, volume=volume, commission=commission,
                pnl=pnl, tags=tags or [],
            ))

        self.daily_trade_count += 1
        return order

    def close_position(self, volume: int = 0, tags: list = None) -> Optional[Order]:
        if self.position.is_flat:
            return None
        close_vol = abs(self.position.volume) if volume == 0 else min(volume, abs(self.position.volume))
        if close_vol <= 0:
            return None
        side = OrderSide.SELL if self.position.volume > 0 else OrderSide.BUY
        return self.place_order(side, close_vol, tags=tags or ["close"])

    def close_today_positions(self) -> bool:
        if not self.position.is_flat:
            order = self.close_position(tags=["market_close"])
            return order is not None and order.is_filled
        return False

    def reset_daily_count(self):
        self.daily_trade_count = 0

    def record_equity(self):
        self.equity_curve.append(float(self.equity))

    def get_summary(self) -> dict:
        """获取回测总结 - 修复 Infinity 问题"""
        if len(self.trades) == 0:
            return {"message": "无交易记录"}

        closed_trades = [t for t in self.trades if t.pnl != 0]
        
        if len(closed_trades) == 0:
            return {"message": "无成交交易"}

        total_pnl = sum(t.pnl for t in closed_trades)
        total_commission = sum(t.commission for t in self.trades)
        net_pnl = total_pnl - total_commission

        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl < 0]

        # ===== 最大回撤 =====
        if len(self.equity_curve) > 0:
            peak = self.equity_curve[0]
            max_dd = 0.0
            for eq in self.equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0.0

        # ===== 盈亏比计算（修复 Infinity） =====
        total_trades = len(closed_trades)
        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
        avg_loss = abs(sum(t.pnl for t in losing) / len(losing)) if losing else 0
        
        # 盈亏比 = 平均盈利 / 平均亏损（避免除以0）
        if avg_loss > 0:
            profit_factor = avg_win / avg_loss
        else:
            profit_factor = 0.0

        # 夏普比率
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * (252 ** 0.5)

        # ===== 安全转换函数：确保所有值都是有限数字 =====
        def safe_float(val):
            if isinstance(val, float) and (val == float('inf') or val == float('-inf')):
                return 0.0
            if isinstance(val, float) and (val != val):  # NaN
                return 0.0
            return val

        return {
            "initial_capital": safe_float(self.initial_capital),
            "final_equity": safe_float(self.equity_curve[-1] if self.equity_curve else self.capital),
            "net_pnl": safe_float(net_pnl),
            "return_rate": safe_float(net_pnl / self.initial_capital * 100 if self.initial_capital > 0 else 0),
            "total_trades": total_trades,
            "win_rate": safe_float(win_rate * 100),
            "profit_factor": safe_float(profit_factor),
            "max_drawdown": safe_float(max_dd * 100),
            "total_commission": safe_float(total_commission),
            "sharpe_ratio": safe_float(sharpe),
            "avg_win": safe_float(avg_win),
            "avg_loss": safe_float(avg_loss),
        }

    def print_summary(self):
        summary = self.get_summary()
        if "message" in summary:
            print(summary["message"])
            return
        print("=" * 62)
        print(f"  {'📊 回测总结':^56}")
        print("=" * 62)
        print(f"  初始资金:       {summary['initial_capital']:>12,.2f}")
        print(f"  最终权益:       {summary['final_equity']:>12,.2f}")
        print(f"  净盈亏:         {summary['net_pnl']:>+12,.2f}")
        print(f"  收益率:         {summary['return_rate']:>+10.2f}%")
        print(f"  夏普比率:       {summary['sharpe_ratio']:>10.2f}")
        print(f"  {'─' * 56}")
        print(f"  总交易次数:     {summary['total_trades']:>8d}")
        print(f"  胜率:           {summary['win_rate']:>10.2f}%")
        print(f"  盈亏比:         {summary['profit_factor']:>10.2f}")
        print(f"  平均盈利:       {summary['avg_win']:>+12,.2f}")
        print(f"  平均亏损:       {summary['avg_loss']:>+12,.2f}")
        print(f"  {'─' * 56}")
        print(f"  最大回撤:       {summary['max_drawdown']:>10.2f}%")
        print(f"  总手续费:       {summary['total_commission']:>12,.2f}")
        print("=" * 62)