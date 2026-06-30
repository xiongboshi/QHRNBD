"""
回测引擎主循环
"""
from pathlib import Path
from typing import Type, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

from engine.broker import Broker
from engine.data_feed import DataFeed
from strategies.base_strategy import BaseStrategy
from utils.logger import setup_logger


rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class Backtester:
    """回测引擎"""

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        config: dict,
        data_feed: DataFeed = None,
        logger=None,
    ):
        self.config = config
        self.strategy_class = strategy_class
        self.logger = logger or setup_logger("backtester")

        self.data_feed = data_feed or DataFeed(
            data_dir=config.get("data_dir", "data"),
            symbol=config["backtest"]["symbol"],
            freq=config["backtest"].get("data_freq", "1min"),
            data_source=config.get("data_source", "local"),
            tq_username=config.get("tqsdk", {}).get("username", ""),
            tq_password=config.get("tqsdk", {}).get("password", ""),
        )

        self.broker = Broker(
            initial_capital=config["backtest"]["initial_capital"],
            commission_config=config.get("commission", {}),
            contract_config=config.get("contract", {}),
            slippage_config=config.get("slippage", {}),
            risk_config=config.get("risk", {}),
        )

        self.strategy: Optional[BaseStrategy] = None
        self._data: Optional[pd.DataFrame] = None
        self._current_date = None
        self.daily_records: list[dict] = []
        self.daily_direction_records: list[dict] = []

    def run(self, data: pd.DataFrame = None, split_ratio: float = None) -> Broker:
        if data is not None:
            self._data = data
        else:
            self._data = self.data_feed.load()

        self.logger.info(f"数据加载完成: {len(self._data)} 根K线, "
                         f"{self._data.index[0]} ~ {self._data.index[-1]}")

        start_date = self.config["backtest"].get("start_date")
        end_date = self.config["backtest"].get("end_date")
        if start_date and isinstance(self._data.index, pd.DatetimeIndex):
            self._data = self._data.loc[start_date:]
        if end_date and isinstance(self._data.index, pd.DatetimeIndex):
            self._data = self._data.loc[:end_date]

        if len(self._data) == 0:
            raise RuntimeError("回测数据为空！")

        self.logger.info(f"回测时段: {self._data.index[0]} ~ {self._data.index[-1]}")

        self.strategy = self.strategy_class(self.broker)
        self.strategy._symbol = self.config["backtest"]["symbol"]

        strategy_params = self.config.get("strategy_params", {})
        if strategy_params:
            self.strategy.params.update(strategy_params)
            self.logger.info(f"策略参数: {strategy_params}")

        self.strategy.prepare(self._data)
        self.logger.info(f"策略初始化: {self.strategy}")
        self.strategy.on_start()

        self._run_loop()

        self.strategy.on_end()
        self.logger.info("回测完成")
        
        # ===== 提取支撑线数据 =====
        self.support_lines = []
        if hasattr(self.strategy, 'get_support_lines'):
            try:
                self.support_lines = self.strategy.get_support_lines()
            except:
                self.support_lines = []
        
        return self.broker


    def _run_loop(self):
        """主循环 - 每根K线记录权益"""
        daily_config = self.config.get("intraday", {})
        close_time_str = daily_config.get("close_at_market", "14:55")

        bar_count = 0
        last_date = None

        for idx, (timestamp, bar) in enumerate(self._data.iterrows()):
            self.broker.set_current_bar(bar)
            self.strategy.current_bar = bar
            self.strategy.current_index = idx

            current_date = timestamp.date()

            if current_date != last_date:
                self.broker.reset_daily_count()
                self._record_daily_stats(last_date)
                self._record_daily_direction(last_date, current_date, idx)
                last_date = current_date
                self._current_date = current_date

            # 策略信号
            self.strategy.on_bar(bar)

            # 每根K线记录权益
            self.broker.record_equity()

            # 收盘前强制平仓
            current_time = timestamp.strftime("%H:%M")
            if current_time == close_time_str and not self.broker.position.is_flat:
                self.broker.close_today_positions()
                self.logger.info(f"[{timestamp}] 收盘平仓")

            bar_count += 1

        # 最终记录
        self.broker.record_equity()
        self._record_daily_stats(last_date)
        self.logger.info(f"共处理 {bar_count} 根K线")

    def _record_daily_stats(self, date):
        if date is None:
            return
        self.daily_records.append({
            "date": date,
            "equity": self.broker.equity,
            "position": self.broker.position.volume,
            "trades_today": self.broker.daily_trade_count,
        })

    def _record_daily_direction(self, prev_date, current_date, idx):
        try:
            if self.strategy is None:
                return
            if hasattr(self.strategy, '_daily_trend'):
                trend = self.strategy._daily_trend
                direction_map = {1: "多头", -1: "空头", 0: "震荡"}
                daily_close = None
                if hasattr(self.strategy, '_daily_data') and self.strategy._daily_data is not None:
                    daily_data = self.strategy._daily_data
                    try:
                        if current_date in daily_data.index:
                            daily_close = daily_data.loc[current_date, "close"]
                    except:
                        pass
                self.daily_direction_records.append({
                    "date": current_date,
                    "direction": direction_map.get(trend, "未知"),
                    "trend_code": trend,
                    "daily_close": daily_close,
                })
        except Exception as e:
            self.logger.debug(f"记录日线方向失败: {e}")

    def report(self, save_dir: str = "results"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.broker.print_summary()
        self._save_trades(save_dir)
        self._save_daily_stats(save_dir)
        self._save_daily_direction(save_dir)
        self._plot_equity_curve(save_dir)
        self._save_summary_text(save_dir)

    def _save_trades(self, save_dir: Path):
        trades = self.broker.trades
        if not trades:
            return
        records = []
        for t in trades:
            records.append({
                "time": t.time,
                "symbol": t.symbol,
                "side": t.side,
                "price": t.price,
                "volume": t.volume,
                "commission": t.commission,
                "pnl": t.pnl,
                "tags": ",".join(t.tags) if t.tags else "",
            })
        trades_df = pd.DataFrame(records)
        trades_df.to_csv(save_dir / "report.csv", index=False)
        self.logger.info(f"交易明细已保存: {save_dir / 'report.csv'}")

    def _save_daily_stats(self, save_dir: Path):
        if self.daily_records:
            daily_df = pd.DataFrame(self.daily_records)
            daily_df.to_csv(save_dir / "daily_stats.csv", index=False)

    def _save_daily_direction(self, save_dir: Path):
        if not self.daily_direction_records:
            return
        df = pd.DataFrame(self.daily_direction_records)
        df.to_csv(save_dir / "daily_direction.csv", index=False)
        self.logger.info(f"日线方向记录已保存: {save_dir / 'daily_direction.csv'}")

    def _save_summary_text(self, save_dir: Path):
        summary = self.broker.get_summary()
        if "message" in summary:
            return
        lines = [
            "=" * 60,
            "期货日内波段回测报告",
            "=" * 60,
            f"策略: {self.strategy}",
            f"初始资金: {summary['initial_capital']:,.2f}",
            f"最终权益: {summary['final_equity']:,.2f}",
            f"净盈亏: {summary['net_pnl']:+,.2f}",
            f"收益率: {summary['return_rate']:+.2f}%",
            f"夏普比率: {summary['sharpe_ratio']:.2f}",
            "",
            f"总交易: {summary['total_trades']}",
            f"胜率: {summary['win_rate']:.2f}%",
            f"盈亏比: {summary['profit_factor']:.2f}",
            f"最大回撤: {summary['max_drawdown']:.2f}%",
            f"总手续费: {summary['total_commission']:,.2f}",
            "=" * 60,
        ]
        with open(save_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _plot_equity_curve(self, save_dir: Path):
        """绘制权益曲线"""
        equity = self.broker.equity_curve
        if len(equity) < 2:
            self.logger.warning("权益曲线数据不足")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        # 绘制权益曲线
        ax.plot(equity, linewidth=1.5, color="#1f77b4", label="权益")

        # 初始资金线
        ax.axhline(y=self.broker.initial_capital, color="gray", linestyle="--",
                   alpha=0.7, label=f"初始资金 {self.broker.initial_capital:,.0f}")

        # 填充区域
        ax.fill_between(range(len(equity)), self.broker.initial_capital, equity,
                        where=(np.array(equity) >= self.broker.initial_capital),
                        color="green", alpha=0.15, label="盈利区域")
        ax.fill_between(range(len(equity)), self.broker.initial_capital, equity,
                        where=(np.array(equity) < self.broker.initial_capital),
                        color="red", alpha=0.15, label="亏损区域")

        ax.set_title("权益曲线", fontsize=14, fontweight="bold")
        ax.set_xlabel("交易序号")
        ax.set_ylabel("权益 (¥)")

        def y_fmt(x, p):
            return f"{x:,.0f}"
        ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))

        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

        # 统计信息
        summary = self.broker.get_summary()
        if "net_pnl" in summary:
            text = (
                f"收益率: {summary['return_rate']:+.2f}%  |  "
                f"总交易: {summary['total_trades']}  |  "
                f"胜率: {summary['win_rate']:.2f}%  |  "
                f"最大回撤: {summary['max_drawdown']:.2f}%"
            )
            ax.text(0.02, 0.95, text, transform=ax.transAxes,
                    fontsize=11, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        save_path = save_dir / "equity_curve.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"权益曲线已保存: {save_path}")