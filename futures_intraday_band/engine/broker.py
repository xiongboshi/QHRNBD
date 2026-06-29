"""
模拟经纪商 - 负责：
1. 订单执行与成交撮合
2. 手续费计算
3. 滑点模拟
4. 持仓与资金管理
5. 风控检查（日内限额、最大回撤、资金管理）
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import pandas as pd


# ==========================================================
# 枚举与数据结构
# ==========================================================

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"    # 市价单
    LIMIT = "limit"      # 限价单


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """订单对象"""
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
    reject_reason: str = ""  # 拒单原因

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_rejected(self) -> bool:
        return self.status == OrderStatus.CANCELLED


@dataclass
class Position:
    """持仓对象"""
    symbol: str
    volume: int = 0
    avg_price: float = 0.0
    pnl: float = 0.0
    pnl_ratio: float = 0.0

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
    """成交记录"""
    time: pd.Timestamp
    symbol: str
    side: str
    price: float
    volume: int
    commission: float
    pnl: float = 0.0
    before_pnl: float = 0.0
    tags: list = field(default_factory=list)  # 标签：如 "stop_loss", "take_profit"

    def __str__(self):
        action = "开多" if (self.side == "buy" and self.pnl == 0) else \
                 "平多" if (self.side == "sell" and self.pnl != 0) else \
                 "开空" if (self.side == "sell" and self.pnl == 0) else "平空"
        tag_str = f" [{','.join(self.tags)}]" if self.tags else ""
        return (f"[{self.time.strftime('%m-%d %H:%M')}] {self.symbol} {action} "
                f"{self.volume}手 @ {self.price:.1f} | "
                f"手续费:{self.commission:.2f} | 盈亏:{self.pnl:.2f}{tag_str}")


# ==========================================================
# 风控检查器
# ==========================================================

class RiskController:
    """风控检查器 - 独立的风控逻辑"""

    def __init__(self, config: dict = None):
        """
        Args:
            config: 风控配置字典，对应 settings.yaml 中的 risk 段
        """
        self.config = config or {}
        self._daily_pnl = 0.0          # 当日盈亏累计
        self._daily_trades = 0         # 当日交易次数
        self._max_drawdown_reached = False
        self._peak_capital = 0.0       # 最高资金峰值
        self._consecutive_losses = 0   # 连续亏损次数
        self._last_trade_pnl = 0.0

    def reset_daily(self):
        """每日重置"""
        self._daily_pnl = 0.0
        self._daily_trades = 0

    def on_trade(self, pnl: float, commission: float):
        """每次成交后更新状态"""
        net_pnl = pnl - commission
        self._daily_pnl += net_pnl
        self._daily_trades += 1

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._last_trade_pnl = pnl

    def check_order(self, side: OrderSide, volume: int, price: float,
                    current_capital: float, position: Position,
                    current_bar: pd.Series = None) -> tuple[bool, str]:
        """风控检查：是否允许下单

        Returns:
            (allowed: bool, reason: str)
        """
        risk = self.config

        # 1. 单笔手数限制
        max_size = risk.get("max_position_size", 20)
        new_total = abs(position.volume) + volume
        if new_total > max_size:
            return False, f"超过最大持仓手数限制 ({max_size})"

        # 2. 固定手数 vs 资金管理
        use_fixed = risk.get("use_fixed_size", True)
        if not use_fixed:
            # 资金管理模式：根据可用资金计算可开手数
            fixed_size = risk.get("fixed_contract_size", 1)
            if volume > fixed_size:
                return False, f"超过资金管理限制手数 ({fixed_size})"

        # 3. 日内最大亏损限制
        max_daily_loss = risk.get("max_daily_loss", 0)
        if max_daily_loss > 0 and self._daily_pnl <= -max_daily_loss:
            return False, f"触及日内最大亏损限制 ({max_daily_loss})"

        # 4. 连续亏损暂停
        max_consecutive_loss = risk.get("max_consecutive_loss", 0)
        if max_consecutive_loss > 0 and self._consecutive_losses >= max_consecutive_loss:
            return False, f"连续亏损 {self._consecutive_losses} 次，暂停交易"

        # 5. 最大回撤检查
        max_dd = risk.get("max_drawdown", 0.15)
        if self._peak_capital > 0:
            dd = (self._peak_capital - current_capital) / self._peak_capital
            if dd >= max_dd:
                self._max_drawdown_reached = True
                return False, f"触及最大回撤限制 ({dd*100:.1f}% >= {max_dd*100:.1f}%)"

        return True, ""

    def update_peak(self, capital: float):
        """更新资金峰值"""
        if capital > self._peak_capital:
            self._peak_capital = capital


# ==========================================================
# Broker - 增强版
# ==========================================================

class Broker:
    """模拟经纪商（增强版：加入风控、资金管理、动态滑点）"""

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

        # 持仓
        self.position = Position(symbol="")

        # 组件
        self.risk_controller = RiskController(risk_config or {})

        # 订单与成交
        self._order_id = 0
        self.trades: list[TradeRecord] = []
        self.all_orders: list[Order] = []

        # 状态
        self._current_bar: Optional[pd.Series] = None
        self.daily_trade_count = 0
        self.equity_curve: list[float] = [initial_capital]

        # 回调钩子
        self._on_trade_hooks: list[Callable] = []

    # ---------- 回调系统 ----------

    def add_trade_hook(self, hook: Callable):
        """添加成交回调"""
        self._on_trade_hooks.append(hook)

    def _fire_trade_hooks(self, trade: TradeRecord):
        for hook in self._on_trade_hooks:
            try:
                hook(trade)
            except Exception as e:
                print(f"[Broker] 回调执行失败: {e}")

    # ---------- 核心方法 ----------

    def set_current_bar(self, bar: pd.Series):
        """更新当前K线"""
        self._current_bar = bar

    @property
    def equity(self) -> float:
        """当前总权益 = 现金 + 浮动盈亏"""
        if self.position.is_flat or self._current_bar is None:
            return self.capital
        close = self._current_bar.get("close", 0)
        if self.position.volume > 0:
            unrealized = self.position.volume * (close - self.position.avg_price)
        else:
            unrealized = abs(self.position.volume) * (self.position.avg_price - close)
        return self.capital + unrealized

    @property
    def available_capital(self) -> float:
        """可用资金（扣除保证金后）"""
        if self.position.is_flat:
            return self.capital
        margin = self.get_margin(self.position.avg_price, abs(self.position.volume), self.position.symbol)
        return self.capital - margin + (self.equity - self.capital)  # 浮动盈利可追加保证金

    def get_margin(self, price: float, volume: int, symbol: str) -> float:
        """计算保证金"""
        contract = self.contract_config.get(symbol, {})
        multiplier = contract.get("multiplier", 10)
        margin_rate = contract.get("margin_rate", 0.08)
        return price * multiplier * volume * margin_rate

    def get_contract_value(self, price: float, volume: int, symbol: str) -> float:
        """计算合约价值"""
        contract = self.contract_config.get(symbol, {})
        multiplier = contract.get("multiplier", 10)
        return price * multiplier * volume

    def calculate_commission(self, price: float, volume: int, side: str, symbol: str) -> float:
        """计算手续费（支持平今/平昨差异化）"""
        comm = self.commission_config.get(symbol, {})
        multiplier = self.contract_config.get(symbol, {}).get("multiplier", 10)
        turnover = price * multiplier * abs(volume)

        if side == "buy":
            ratio = comm.get("open_ratio", 0.0001)
        else:
            # 判断平今仓（简化：日内平仓视为平今）
            ratio = comm.get("close_today_ratio", comm.get("close_ratio", 0.0001))

        return max(turnover * ratio, 0.01)

    def _get_slippage_price(self, price: float, side: OrderSide, volume: int = 1) -> float:
        """计算动态滑点后的执行价格

        滑点随波动率和成交量动态变化
        """
        slippage_cfg = self.slippage_config
        contract = self.contract_config.get(self.position.symbol, {})
        min_tick = contract.get("min_tick", 1)

        if slippage_cfg.get("mode") == "fixed":
            base_ticks = slippage_cfg.get("fixed_ticks", 1)
            # 大单量时滑点更大
            volume_factor = 1.0 + (abs(volume) - 1) * 0.5 if abs(volume) > 1 else 1.0
            ticks = base_ticks * volume_factor
            slippage = ticks * min_tick
        else:
            ratio = slippage_cfg.get("ratio", 0.0001)
            # 比例模式下滑点也随量增大
            slippage = price * ratio * (1 + abs(volume) * 0.01)

        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage

    def _next_order_id(self) -> int:
        self._order_id += 1
        return self._order_id

    def place_order(
        self,
        side: OrderSide,
        volume: int,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        symbol: str = "",
        tags: list = None,
    ) -> Optional[Order]:
        """下单（带风控检查）

        Args:
            side: 买卖方向
            volume: 手数（正数）
            price: 价格（市价单为None）
            order_type: 订单类型
            symbol: 品种代码
            tags: 订单标签

        Returns:
            成交后的Order对象，被拒则返回带 reject_reason 的Order
        """
        if volume <= 0:
            return None

        symbol = symbol or self.position.symbol

        # ---------- 风控检查 ----------
        fill_price = self._get_current_price(side) if self._current_bar is not None else (price or 0)
        allowed, reason = self.risk_controller.check_order(
            side=side,
            volume=volume,
            price=fill_price,
            current_capital=self.capital,
            position=self.position,
            current_bar=self._current_bar,
        )
        if not allowed:
            reject_order = Order(
                id=self._next_order_id(),
                time=self._current_bar.name if self._current_bar is not None else pd.Timestamp.now(),
                symbol=symbol,
                side=side,
                price=fill_price,
                volume=volume,
                order_type=order_type,
                status=OrderStatus.CANCELLED,
                reject_reason=reason,
            )
            self.all_orders.append(reject_order)
            return reject_order

        # 执行成交
        if order_type == OrderType.MARKET:
            if self._current_bar is None:
                return None
            price_to_use = self._get_current_price(side)
        else:
            price_to_use = price or self._get_current_price(side)

        return self._execute_order(side, volume, price_to_use, symbol, tags)

    def _get_current_price(self, side: OrderSide) -> float:
        """获取当前成交价（含滑点）"""
        if self._current_bar is None:
            return 0.0

        if side == OrderSide.BUY:
            # 买使用 high + 滑点（更保守）
            base_price = max(self._current_bar["close"], self._current_bar["open"])
        else:
            base_price = min(self._current_bar["close"], self._current_bar["open"])

        return self._get_slippage_price(base_price, side)

    def _execute_order(self, side: OrderSide, volume: int, price: float, symbol: str, tags: list = None) -> Order:
        """执行订单成交"""
        order = Order(
            id=self._next_order_id(),
            time=self._current_bar.name if self._current_bar is not None else pd.Timestamp.now(),
            symbol=symbol,
            side=side,
            price=price,
            volume=volume,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            fill_price=price,
            fill_time=self._current_bar.name if self._current_bar is not None else pd.Timestamp.now(),
        )
        self.all_orders.append(order)

        commission = self.calculate_commission(price, volume, side.value, symbol)
        order.commission = commission

        turnover = self.get_contract_value(price, volume, symbol)
        margin = self.get_margin(price, volume, symbol)

        # 判断是开仓还是平仓
        is_open = (side == OrderSide.BUY and self.position.volume >= 0) or \
                  (side == OrderSide.SELL and self.position.volume <= 0)

        if is_open:
            # ===== 开仓 =====
            pnl_record = TradeRecord(
                time=order.fill_time, symbol=symbol, side=side.value,
                price=price, volume=volume, commission=commission,
                tags=tags or [],
            )
            self.trades.append(pnl_record)
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

            self.capital -= turnover
            self.capital += (turnover - margin)  # 保证金以外的资金可自由使用

        else:
            # ===== 平仓 =====
            if self.position.volume > 0:
                close_pnl = volume * (price - self.position.avg_price)
                self.position.volume -= volume
            else:
                close_pnl = volume * (self.position.avg_price - price)
                self.position.volume += volume

            margin = self.get_margin(price, volume, symbol)
            self.capital += margin
            self.capital -= commission
            self.capital += close_pnl

            pnl_record = TradeRecord(
                time=order.fill_time, symbol=symbol, side=side.value,
                price=price, volume=volume, commission=commission,
                pnl=close_pnl, tags=tags or [],
            )
            self.trades.append(pnl_record)

            # 触发成交回调
            self._fire_trade_hooks(pnl_record)

            # 更新风控状态
            self.risk_controller.on_trade(close_pnl, commission)

        # 更新日内计数
        self.daily_trade_count += 1

        return order

    # ---------- 快捷方法 ----------

    def close_position(self, volume: int = 0, tags: list = None) -> Optional[Order]:
        """平仓"""
        if self.position.is_flat:
            return None

        close_vol = abs(self.position.volume) if volume == 0 else min(volume, abs(self.position.volume))
        if close_vol <= 0:
            return None

        side = OrderSide.SELL if self.position.volume > 0 else OrderSide.BUY
        return self.place_order(side, close_vol, tags=tags or ["close"])

    def close_today_positions(self) -> bool:
        """收盘前强制平仓"""
        if not self.position.is_flat:
            order = self.close_position(tags=["market_close"])
            return order is not None and order.is_filled
        return False

    def reset_daily_count(self):
        """重置日内计数"""
        self.daily_trade_count = 0
        self.risk_controller.reset_daily()

    def record_equity(self):
        """记录当前权益并更新风控峰值"""
        current_equity = self.equity
        self.equity_curve.append(current_equity)
        self.risk_controller.update_peak(current_equity)

    # ---------- 报告 ----------

    def get_summary(self) -> dict:
        """获取回测总结"""
        if len(self.trades) == 0:
            return {"message": "无交易记录"}

        closed_trades = [t for t in self.trades if t.pnl != 0]
        cancel_orders = [o for o in self.all_orders if o.is_rejected]

        total_pnl = sum(t.pnl for t in closed_trades)
        total_commission = sum(t.commission for t in self.trades)
        net_pnl = total_pnl - total_commission

        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl < 0]

        # 最大回撤
        peak = self.equity_curve[0]
        max_dd = 0.0
        max_dd_start = max_dd_end = 0
        for i, eq in enumerate(self.equity_curve):
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_end = i

        # 盈亏比
        if losing:
            win_rate = len(winning) / len(closed_trades) if closed_trades else 0
            avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
            avg_loss = abs(sum(t.pnl for t in losing) / len(losing)) if losing else 1
            profit_factor = sum(t.pnl for t in winning) / abs(sum(t.pnl for t in losing)) if sum(t.pnl for t in losing) != 0 else float("inf")
        else:
            win_rate = 1.0 if closed_trades else 0
            profit_factor = float("inf")

        # 夏普比率（简化：年化）
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * (252 ** 0.5)

        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.equity_curve[-1] if self.equity_curve else self.capital,
            "total_return": net_pnl,
            "return_rate": net_pnl / self.initial_capital * 100,
            "total_trades": len(closed_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd * 100,
            "total_commission": total_commission,
            "net_pnl": net_pnl,
            "sharpe_ratio": sharpe,
            "rejected_orders": len(cancel_orders),
            "avg_win": sum(t.pnl for t in winning) / len(winning) if winning else 0,
            "avg_loss": sum(t.pnl for t in losing) / len(losing) if losing else 0,
        }

    def print_summary(self):
        """打印回测总结"""
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
        print(f"  年化夏普比率:   {summary['sharpe_ratio']:>10.2f}")
        print(f"  {'─' * 56}")
        print(f"  总交易次数:     {summary['total_trades']:>8d}")
        print(f"  胜率:           {summary['win_rate']:>10.2f}%")
        print(f"  盈亏比:         {summary['profit_factor']:>10.2f}")
        print(f"  平均盈利:       {summary['avg_win']:>+12,.2f}")
        print(f"  平均亏损:       {summary['avg_loss']:>+12,.2f}")
        print(f"  {'─' * 56}")
        print(f"  最大回撤:       {summary['max_drawdown']:>10.2f}%")
        print(f"  总手续费:       {summary['total_commission']:>12,.2f}")
        print(f"  被拒订单数:     {summary['rejected_orders']:>8d}")
        print("=" * 62)
