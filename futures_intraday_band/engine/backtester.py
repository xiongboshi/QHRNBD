"""
回测引擎主循环
负责：
1. 串联 DataFeed → Broker → Strategy
2. 按 K线时间序列推进回测
3. 交易时段管理（日盘/夜盘）+ 收盘前强制平仓
4. 动态止损追踪
5. 多维度回测报告与图表
6. 每日PBX日线方向记录
"""
from pathlib import Path
from typing import Type, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams

from engine.broker import Broker
from engine.data_feed import DataFeed
from strategies.base_strategy import BaseStrategy
from utils.logger import setup_logger


# 图表中文支持
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class Backtester:
    """回测引擎（增强版：加入动态止损、分段回测、多维度分析）"""

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

        risk_config = config.get("risk", {})
        self.broker = Broker(
            initial_capital=config["backtest"]["initial_capital"],
            commission_config=config.get("commission", {}),
            contract_config=config.get("contract", {}),
            slippage_config=config.get("slippage", {}),
            risk_config=risk_config,
        )

        self.strategy: Optional[BaseStrategy] = None
        self._data: Optional[pd.DataFrame] = None
        self._current_date = None

        # 每日统计追踪
        self.daily_records: list[dict] = []
        
        # ===== 新增：每日方向记录 =====
        self.daily_direction_records: list[dict] = []

    def run(self, data: pd.DataFrame = None, split_ratio: float = None) -> Broker:
        """执行回测

        Args:
            data: 外部传入数据
            split_ratio: 训练/测试集分割比例（如 0.7 = 前70%训练，后30%测试）

        Returns:
            Broker 实例
        """
        if data is not None:
            self._data = data
        else:
            self._data = self.data_feed.load()

        self.logger.info(f"数据加载完成: {len(self._data)} 根K线, "
                         f"{self._data.index[0]} ~ {self._data.index[-1]}")

        # 按时间段过滤（仅在设置了日期且索引是DatetimeIndex时有效）
        start_date = self.config["backtest"].get("start_date")
        end_date = self.config["backtest"].get("end_date")
        if start_date and isinstance(self._data.index, pd.DatetimeIndex):
            self._data = self._data.loc[start_date:]
        if end_date and isinstance(self._data.index, pd.DatetimeIndex):
            self._data = self._data.loc[:end_date]

        if len(self._data) == 0:
            raise RuntimeError("回测数据为空！请检查 start_date/end_date 配置或数据文件")

        self.logger.info(f"回测时段: {self._data.index[0]} ~ {self._data.index[-1]}")

        # 分割训练/测试集
        if split_ratio is not None:
            split_idx = int(len(self._data) * split_ratio)
            train_data = self._data.iloc[:split_idx]
            test_data = self._data.iloc[split_idx:]
            self.logger.info(f"数据分割: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
            self._split_point = self._data.index[split_idx]

        # 策略初始化
        self.strategy = self.strategy_class(self.broker)
        self.strategy._symbol = self.config["backtest"]["symbol"]

        # 策略参数加载
        strategy_params = self.config.get("strategy_params", {})
        if strategy_params:
            self.strategy.params.update(strategy_params)
            self.logger.info(f"策略参数: {strategy_params}")

        self.strategy.prepare(self._data)
        self.logger.info(f"策略初始化: {self.strategy}")
        self.strategy.on_start()

        # 主循环
        self._run_loop()

        self.strategy.on_end()
        self.logger.info("回测完成")

        return self.broker

    def _run_loop(self):
        """主循环 - 逐根K线推进"""
        daily_config = self.config.get("intraday", {})
        close_time_str = daily_config.get("close_at_market", "14:55")

        # 动态止损配置
        trailing_stop_config = self.config.get("trailing_stop", {})
        use_trailing = trailing_stop_config.get("enabled", False)
        trailing_activation = trailing_stop_config.get("activation_pct", 0.005)
        trailing_distance = trailing_stop_config.get("trailing_distance", 0.003)

        bar_count = 0
        last_date = None
        trailing_high = 0.0
        trailing_low = float("inf")
        trailing_activated = False

        for idx, (timestamp, bar) in enumerate(self._data.iterrows()):
            self.broker.set_current_bar(bar)
            self.strategy.current_bar = bar
            self.strategy.current_index = idx

            current_date = timestamp.date()

            # 新的一天
            if current_date != last_date:
                self.broker.reset_daily_count()
                self._record_daily_stats(last_date)
                
                # ===== 新增：记录每日PBX日线方向 =====
                self._record_daily_direction(last_date, current_date, idx)
                
                last_date = current_date
                self._current_date = current_date
                trailing_activated = False
                trailing_high = 0.0
                trailing_low = float("inf")

            # ========================
            # 动态止损追踪
            # ========================
            if use_trailing and not self.broker.position.is_flat:
                is_long = self.broker.position.volume > 0
                entry_price = self.broker.position.avg_price
                current_close = bar["close"]
                high, low = bar["high"], bar["low"]

                if is_long:
                    if not trailing_activated:
                        if (current_close - entry_price) / entry_price >= trailing_activation:
                            trailing_activated = True
                            trailing_high = max(high, entry_price * (1 + trailing_activation))
                    else:
                        trailing_high = max(trailing_high, high)
                        stop_price = trailing_high * (1 - trailing_distance)
                        if low <= stop_price:
                            self.logger.info(f"[{timestamp}] 动态止损触发 (多头) @ {stop_price:.1f}")
                            self.broker.close_position(tags=["trailing_stop"])
                            trailing_activated = False
                else:
                    if not trailing_activated:
                        if (entry_price - current_close) / entry_price >= trailing_activation:
                            trailing_activated = True
                            trailing_low = min(low, entry_price * (1 - trailing_activation))
                    else:
                        trailing_low = min(trailing_low, low)
                        stop_price = trailing_low * (1 + trailing_distance)
                        if high >= stop_price:
                            self.logger.info(f"[{timestamp}] 动态止损触发 (空头) @ {stop_price:.1f}")
                            self.broker.close_position(tags=["trailing_stop"])
                            trailing_activated = False

            # 策略信号
            self.strategy.on_bar(bar)

            # 收盘前强制平仓
            current_time = timestamp.strftime("%H:%M")
            if current_time == close_time_str and not self.broker.position.is_flat:
                self.broker.close_today_positions()
                self.logger.info(f"[{timestamp}] 收盘平仓")

            # 记录权益
            is_market_close = (timestamp.hour == 15 and timestamp.minute == 0) or \
                              (idx == len(self._data) - 1) or \
                              (current_time >= "14:55" and current_time <= "15:00")
            if is_market_close and current_time >= "14:55":
                self.broker.record_equity()

            bar_count += 1

        # 最终记录
        if not self.broker.equity_curve or self.broker.equity_curve[-1] != self.broker.equity:
            self.broker.record_equity()
        self._record_daily_stats(last_date)

        self.logger.info(f"共处理 {bar_count} 根K线")

    def _record_daily_stats(self, date):
        """记录每日统计"""
        if date is None:
            return
        self.daily_records.append({
            "date": date,
            "equity": self.broker.equity,
            "position": self.broker.position.volume,
            "trades_today": self.broker.daily_trade_count,
        })

    # ==========================================================
    # 新增：每日PBX日线方向记录
    # ==========================================================
    def _record_daily_direction(self, prev_date, current_date, idx):
        """记录每日PBX日线方向"""
        try:
            if self.strategy is None:
                return
            
            # 从策略中获取日线方向
            if hasattr(self.strategy, '_daily_trend'):
                trend = self.strategy._daily_trend
                direction_map = {1: "多头", -1: "空头", 0: "震荡"}
                
                # 获取当日日线收盘价（用于参考）
                daily_close = None
                if hasattr(self.strategy, '_daily_data') and self.strategy._daily_data is not None:
                    daily_data = self.strategy._daily_data
                    # 查找当前日期对应的日线数据
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
                    "comment": f"PBX日线定方向: {direction_map.get(trend, '未知')}"
                })
        except Exception as e:
            self.logger.debug(f"记录日线方向失败: {e}")

    def report(self, save_dir: str = "results"):
        """生成多维度回测报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 打印总结
        self.broker.print_summary()

        # 保存交易明细
        self._save_trades(save_dir)

        # 保存每日统计
        self._save_daily_stats(save_dir)

        # ===== 新增：保存每日PBX日线方向 =====
        self._save_daily_direction(save_dir)

        # 绘制多维度图表
        self._plot_multi_chart(save_dir)

        # 保存摘要到文本
        self._save_summary_text(save_dir)

    def _save_trades(self, save_dir: Path):
        """保存交易明细"""
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

        # 也保存被拒订单
        rejected = [o for o in self.broker.all_orders if o.is_rejected]
        if rejected:
            reject_df = pd.DataFrame([{
                "time": o.time, "side": o.side.value,
                "volume": o.volume, "price": o.price,
                "reason": o.reject_reason,
            } for o in rejected])
            reject_df.to_csv(save_dir / "rejected_orders.csv", index=False)
            self.logger.info(f"被拒订单已保存: {save_dir / 'rejected_orders.csv'}")

    def _save_daily_stats(self, save_dir: Path):
        """保存每日统计"""
        if self.daily_records:
            daily_df = pd.DataFrame(self.daily_records)
            daily_df.to_csv(save_dir / "daily_stats.csv", index=False)

    # ==========================================================
    # 新增：保存每日PBX日线方向
    # ==========================================================
    def _save_daily_direction(self, save_dir: Path):
        """保存每日PBX日线方向记录"""
        if not self.daily_direction_records:
            self.logger.info("无日线方向记录")
            return
        
        df = pd.DataFrame(self.daily_direction_records)
        
        # 添加统计信息
        direction_counts = df["direction"].value_counts()
        
        # 保存CSV
        csv_path = save_dir / "daily_direction.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"日线方向记录已保存: {csv_path}")
        
        # 打印统计
        self.logger.info(f"日线方向统计: 多头={direction_counts.get('多头', 0)}天, "
                         f"空头={direction_counts.get('空头', 0)}天, "
                         f"震荡={direction_counts.get('震荡', 0)}天")

    def _save_summary_text(self, save_dir: Path):
        """保存摘要文本（含日线方向统计）"""
        summary = self.broker.get_summary()
        if "message" in summary:
            return
        
        # 获取日线方向统计
        direction_stats = ""
        if self.daily_direction_records:
            df = pd.DataFrame(self.daily_direction_records)
            direction_counts = df["direction"].value_counts()
            direction_stats = (
                f"\n日线方向统计:\n"
                f"  多头: {direction_counts.get('多头', 0)} 天\n"
                f"  空头: {direction_counts.get('空头', 0)} 天\n"
                f"  震荡: {direction_counts.get('震荡', 0)} 天"
            )
        
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
            direction_stats,
            "=" * 60,
        ]
        with open(save_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _plot_multi_chart(self, save_dir: Path):
        """绘制多维度图表（4合1）"""
        equity = self.broker.equity_curve
        if len(equity) < 2:
            self.logger.warning("权益曲线数据不足，无法绘制")
            return

        eq_array = np.array(equity)
        equity_series = pd.Series(eq_array)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 2, 2], hspace=0.3, wspace=0.25)

        # 1. 权益曲线 + 回撤填充
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity_series, linewidth=1.5, color="#1f77b4", label="Equity")
        ax1.axhline(y=self.broker.initial_capital, color="gray", linestyle="--", alpha=0.5)
        ax1.fill_between(range(len(eq_array)), self.broker.initial_capital, eq_array,
                          where=(eq_array >= self.broker.initial_capital),
                          color="green", alpha=0.08)
        ax1.fill_between(range(len(eq_array)), self.broker.initial_capital, eq_array,
                          where=(eq_array < self.broker.initial_capital),
                          color="red", alpha=0.08)
        ax1.set_title("权益曲线", fontsize=14, fontweight="bold")
        ax1.set_ylabel("权益 (¥)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="upper left")

        summary = self.broker.get_summary()
        if "net_pnl" in summary:
            stats_text = (
                f"收益率: {summary['return_rate']:+.2f}%  |  "
                f"夏普: {summary['sharpe_ratio']:.2f}  |  "
                f"最大回撤: {summary['max_drawdown']:.2f}%"
            )
            ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

        # 2. 回撤曲线
        ax2 = fig.add_subplot(gs[1, 0])
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        ax2.fill_between(range(len(drawdown)), 0, drawdown, color="crimson", alpha=0.2)
        ax2.plot(drawdown, color="crimson", linewidth=1)
        ax2.axhline(y=-summary.get("max_drawdown", 0), color="red", linestyle="--", alpha=0.5)
        ax2.set_title("回撤曲线", fontsize=12)
        ax2.set_ylabel("回撤 (%)")
        ax2.grid(True, alpha=0.25)
        ax2.set_xlim(0, len(drawdown))

        # 3. 每日收益率柱状图
        ax3 = fig.add_subplot(gs[1, 1])
        if self.daily_records:
            daily_df = pd.DataFrame(self.daily_records)
            if "equity" in daily_df.columns and len(daily_df) > 1:
                daily_df["return"] = daily_df["equity"].pct_change() * 100
                daily_df = daily_df.dropna()
                if not daily_df.empty:
                    colors = ["green" if r >= 0 else "red" for r in daily_df["return"]]
                    ax3.bar(range(len(daily_df)), daily_df["return"], color=colors, alpha=0.6, width=0.8)
                    ax3.axhline(y=0, color="black", linewidth=0.5)
                    ax3.set_title("每日收益率", fontsize=12)
                    ax3.set_ylabel("收益率 (%)")
                    ax3.grid(True, alpha=0.25)
                    ax3.set_xlabel("交易天数")

        # 4. 盈亏分布直方图
        ax4 = fig.add_subplot(gs[2, 0])
        closed_trades = [t for t in self.broker.trades if t.pnl != 0]
        if closed_trades:
            pnls = [t.pnl for t in closed_trades]
            ax4.hist(pnls, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
            ax4.axvline(x=0, color="red", linestyle="--", alpha=0.7)
            ax4.axvline(x=np.mean(pnls), color="green", linestyle="--", alpha=0.7, label=f"均值: {np.mean(pnls):.1f}")
            ax4.set_title("盈亏分布", fontsize=12)
            ax4.set_xlabel("盈亏 (¥)")
            ax4.set_ylabel("频次")
            ax4.legend()
            ax4.grid(True, alpha=0.25)

        # 5. 成交价格散点图
        ax5 = fig.add_subplot(gs[2, 1])
        if closed_trades:
            buy_trades = [t for t in closed_trades if t.side == "buy"]
            sell_trades = [t for t in closed_trades if t.side == "sell"]
            if buy_trades:
                ax5.scatter([t.time for t in buy_trades], [t.price for t in buy_trades],
                           color="green", alpha=0.5, s=20, label="开多/平空")
            if sell_trades:
                ax5.scatter([t.time for t in sell_trades], [t.price for t in sell_trades],
                           color="red", alpha=0.5, s=20, label="开空/平多")
            ax5.set_title("成交价格分布", fontsize=12)
            ax5.set_ylabel("价格")
            ax5.legend()
            ax5.grid(True, alpha=0.25)
            plt.setp(ax5.get_xticklabels(), rotation=30, ha="right")

        # 修复 tight_layout 警告：用 subplots_adjust 替代
        plt.subplots_adjust(left=0.08, right=0.95, bottom=0.06, top=0.94, 
                            hspace=0.35, wspace=0.25)
        
        save_path = save_dir / "backtest_report.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"回测报告已保存: {save_path}")