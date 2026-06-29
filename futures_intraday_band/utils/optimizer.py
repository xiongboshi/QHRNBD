"""
参数优化器 - 网格搜索最佳参数组合

用法：
    from utils.optimizer import ParameterOptimizer

    opt = ParameterOptimizer(
        strategy_class=PbxUShapeStrategy,
        config=config,
        data_feed=data_feed,
    )

    # 定义搜索空间
    param_grid = {
        "lookback": [8, 10, 12, 15],
        "recovery_ratio": [0.3, 0.4, 0.5],
        "stop_loss_atr": [1.0, 1.5, 2.0],
        "take_profit_atr": [2.0, 2.5, 3.0],
    }

    # 执行优化
    results = opt.grid_search(param_grid, metric="sharpe_ratio", top_n=10)

    # 输出最佳参数
    opt.print_top_results()
"""
from itertools import product
from copy import deepcopy
from typing import Type
import pandas as pd
import numpy as np
from datetime import datetime
import json

from engine.backtester import Backtester
from engine.data_feed import DataFeed
from engine.broker import Broker
from strategies.base_strategy import BaseStrategy
from utils.logger import setup_logger


class ParameterOptimizer:
    """参数优化器（网格搜索 + 防止过拟合的交叉验证）"""

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        config: dict,
        data_feed: DataFeed = None,
        n_jobs: int = 1,
    ):
        """
        Args:
            strategy_class: 策略类
            config: 配置字典
            data_feed: 数据馈送器
            n_jobs: 并行数（当前为串行，可扩展）
        """
        self.strategy_class = strategy_class
        self.config = deepcopy(config)
        self.data_feed = data_feed
        self.n_jobs = n_jobs
        self.logger = setup_logger("optimizer")
        self.results: list[dict] = []

    def grid_search(
        self,
        param_grid: dict,
        metric: str = "sharpe_ratio",
        top_n: int = 10,
        walk_forward: bool = False,
        n_splits: int = 3,
    ) -> list[dict]:
        """网格搜索最优参数

        Args:
            param_grid: 参数网格，如 {"lookback": [8, 10, 12], "recovery_ratio": [0.3, 0.5]}
            metric: 优化指标 (sharpe_ratio, return_rate, profit_factor, win_rate, calmar_ratio)
            top_n: 返回前 n 个结果
            walk_forward: 是否使用滚动窗口验证（防止过拟合）
            n_splits: 滚动窗口数量

        Returns:
            按 metric 排序的结果列表
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # 计算总组合数
        total = np.prod([len(v) for v in param_values])
        self.logger.info(f"参数搜索空间: {param_names}")
        self.logger.info(f"总组合数: {total}")

        if total > 200:
            self.logger.warning(f"组合数过多 ({total})，建议先用较粗网格筛选再细化")

        # 加载数据
        if self.data_feed:
            self._data = self.data_feed.load()
        else:
            self._data = None

        # 遍历所有组合
        count = 0
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            count += 1

            if count % 10 == 0 or count == 1 or count == total:
                self.logger.info(f"[{count}/{total}] 测试参数: {params}")

            try:
                if walk_forward and self._data is not None:
                    score = self._evaluate_walk_forward(params, n_splits, metric)
                else:
                    score = self._evaluate_single(params, metric)

                self.results.append({**params, **score})

            except Exception as e:
                self.logger.error(f"参数 {params} 评估失败: {e}")
                continue

        # 按指标排序
        if metric == "max_drawdown":
            # 回撤越小越好
            self.results.sort(key=lambda x: x.get(metric, float("inf")))
        else:
            self.results.sort(key=lambda x: x.get(metric, -float("inf")), reverse=True)

        top_results = self.results[:top_n]
        self.logger.info(f"优化完成，前 {top_n} 个结果已保存")

        # 保存结果
        self._save_results(param_grid, top_results)

        return top_results

    def _evaluate_single(self, params: dict, metric: str) -> dict:
        """单组参数评估"""
        config = deepcopy(self.config)
        config["strategy_params"] = params

        backtester = Backtester(
            strategy_class=self.strategy_class,
            config=config,
            data_feed=None,
            logger=self.logger,
        )

        broker = backtester.run(data=self._data)
        summary = broker.get_summary()

        return {
            "initial_capital": summary.get("initial_capital", 0),
            "final_equity": summary.get("final_equity", 0),
            "return_rate": summary.get("return_rate", 0),
            "total_trades": summary.get("total_trades", 0),
            "win_rate": summary.get("win_rate", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "max_drawdown": summary.get("max_drawdown", 0),
            "sharpe_ratio": summary.get("sharpe_ratio", 0),
            "net_pnl": summary.get("net_pnl", 0),
        }

    def _evaluate_walk_forward(self, params: dict, n_splits: int, metric: str) -> dict:
        """滚动窗口交叉验证（防止过拟合）

        将数据分为 n_splits 段，每段前80%训练，后20%测试
        """
        if self._data is None or len(self._data) < n_splits * 100:
            return self._evaluate_single(params, metric)

        chunk_size = len(self._data) // n_splits
        oos_scores = []

        for i in range(n_splits):
            train_end = (i + 1) * chunk_size
            test_start = train_end
            test_end = min(test_start + chunk_size, len(self._data))

            if test_start >= test_end:
                continue

            train_data = self._data.iloc[:train_end]
            test_data = self._data.iloc[test_start:test_end]

            if len(test_data) < 50:
                continue

            config = deepcopy(self.config)
            config["strategy_params"] = params

            # 在训练集上回测（仅用于确认参数有效性，不参与评分）
            backtester = Backtester(
                strategy_class=self.strategy_class,
                config=config,
                logger=self.logger,
            )
            backtester.run(data=train_data)
            train_summary = backtester.broker.get_summary()

            # 在测试集上回测（用于评分）
            backtester2 = Backtester(
                strategy_class=self.strategy_class,
                config=config,
                logger=self.logger,
            )
            backtester2.run(data=test_data)
            oos_summary = backtester2.broker.get_summary()

            # 收集测试集指标
            metric_value = oos_summary.get(metric, 0) if metric != "max_drawdown" else -oos_summary.get(metric, 0)
            oos_scores.append(metric_value)

        if not oos_scores:
            return self._evaluate_single(params, metric)

        # 返回平均和稳定性
        avg_score = np.mean(oos_scores)
        std_score = np.std(oos_scores) if len(oos_scores) > 1 else 0

        single_result = self._evaluate_single(params, metric)
        single_result[f"{metric}_walk_avg"] = avg_score
        single_result[f"{metric}_walk_std"] = std_score

        return single_result

    def print_top_results(self, n: int = 10):
        """打印前 n 个结果"""
        if not self.results:
            print("无结果可显示")
            return

        print("=" * 100)
        print(f"{'排名':>4} | {'参数':^50} | {'收益率':>8} | {'胜率':>6} | {'盈亏比':>8} | {'夏普':>6} | {'回撤':>6} | {'交易':>5}")
        print("-" * 100)

        for i, r in enumerate(self.results[:n]):
            # 提取参数
            param_str = ", ".join(f"{k}={v}" for k, v in r.items()
                                   if k not in ["initial_capital", "final_equity",
                                                 "return_rate", "total_trades",
                                                 "win_rate", "profit_factor",
                                                 "max_drawdown", "sharpe_ratio",
                                                 "net_pnl"])
            if len(param_str) > 48:
                param_str = param_str[:45] + "..."

            print(f"{i+1:>4} | {param_str:^50} | "
                  f"{r.get('return_rate', 0):>+7.2f}% | "
                  f"{r.get('win_rate', 0):>5.1f}% | "
                  f"{r.get('profit_factor', 0):>7.2f} | "
                  f"{r.get('sharpe_ratio', 0):>5.2f} | "
                  f"{r.get('max_drawdown', 0):>5.2f}% | "
                  f"{r.get('total_trades', 0):>5d}")

        print("=" * 100)

    def _save_results(self, param_grid: dict, top_results: list):
        """保存优化结果到文件"""
        import json
        from datetime import datetime

        output = {
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy_class.__name__,
            "param_grid": param_grid,
            "top_results": top_results,
        }

        # 保存为 JSON
        filepath = "results/optimization_results.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"优化结果已保存: {filepath}")

        # 也保存 CSV
        if top_results:
            df = pd.DataFrame(top_results)
            df.to_csv("results/optimization_results.csv", index=False)
