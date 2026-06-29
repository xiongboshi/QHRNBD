#!/usr/bin/env python3
"""
期货日内波段回测系统 - 主入口

用法：
    python run.py                                    # PBX+U形策略（默认天勤数据）
    python run.py -s ma_cross                        # 均线交叉策略
    python run.py -s pbx_u_shape --source local      # 使用本地数据
    python run.py -s pbx_u_shape --optimize          # 参数优化
    python run.py --generate-data                    # 仅生成模拟数据
    python run.py --list-strategies                  # 列出所有策略
"""
import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from engine.backtester import Backtester
from engine.data_feed import DataFeed
from utils.logger import setup_logger


# ==========================================================
# 策略注册表
# ==========================================================
STRATEGY_REGISTRY = {
    "pbx_u_shape": {
        "module": "strategies.pbx_u_shape",
        "class": "PbxUShapeStrategy",
        "desc": "PBX瀑布线 + U形底部形态策略（增强版）",
    },
    "ma_cross": {
        "module": "strategies.ma_cross",
        "class": "MaCrossStrategy",
        "desc": "均线交叉策略（示例）",
    },
}


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """加载 YAML 配置文件"""
    config_path = Path(__file__).parent / config_path
    if not config_path.exists():
        print(f"[警告] 配置文件不存在: {config_path}，使用默认配置")
        return {
            "backtest": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100_000,
                "symbol": "rb888",
                "data_freq": "1min",
            },
            "default_data_source": "tqsdk",  # 默认使用天勤
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 如果配置文件中没有 default_data_source，设置默认值
    if "default_data_source" not in config:
        config["default_data_source"] = "tqsdk"
    
    print(f"[配置] 已加载: {config_path}")
    return config


def resolve_strategy(strategy_name: str):
    """动态导入策略类"""
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"未知策略: '{strategy_name}'。可用策略: {available}")

    info = STRATEGY_REGISTRY[strategy_name]
    import importlib
    module = importlib.import_module(info["module"])
    strategy_class = getattr(module, info["class"])
    return strategy_class, info["desc"]


def main():
    parser = argparse.ArgumentParser(
        description="期货日内波段回测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    示例:
    python run.py                                    # 默认 PBX+U形（天勤数据）
    python run.py -s ma_cross                        # 均线交叉
    python run.py --source local                     # 使用本地数据
    python run.py --source sample                    # 使用模拟数据
    python run.py --optimize                         # 参数优化
    python run.py -s pbx_u_shape --freq 5min --symbol i888  # 铁矿石5分钟
        """,
    )
    parser.add_argument("--strategy", "-s", default="pbx_u_shape",
                        help=f"策略名称 ({', '.join(STRATEGY_REGISTRY.keys())})")
    parser.add_argument("--generate-data", action="store_true", help="仅生成模拟数据")
    parser.add_argument("--data-file", type=str, default=None, help="数据文件路径")
    parser.add_argument("--freq", type=str, default="15min",
                        choices=["1min", "5min", "15min", "30min", "60min", "1h", "1d", "day"], 
                        help="K线频率: 1min/5min/15min/30min/60min/1h/1d")
    parser.add_argument("--symbol", type=str, default=None, help="品种代码")
    # ===== 修改点：默认数据源改为 tqsdk =====
    parser.add_argument("--source", type=str, default=None,  # None表示从配置文件读取
                        choices=["local", "tqsdk", "sample"], help="数据来源")
    parser.add_argument("--username", type=str, default="", help="天勤账号（覆盖配置文件）")
    parser.add_argument("--password", type=str, default="", help="天勤密码（覆盖配置文件）")
    parser.add_argument("--optimize", action="store_true", help="执行参数优化")
    parser.add_argument("--list-strategies", action="store_true", help="列出所有策略")
    parser.add_argument("--walk-forward", action="store_true", help="滚动窗口验证（防过拟合）")

    args = parser.parse_args()

    # 列出策略
    if args.list_strategies:
        print("\n可用策略:")
        print("=" * 60)
        for name, info in STRATEGY_REGISTRY.items():
            print(f"  {name:<20} - {info['desc']}")
        print("=" * 60)
        return

    # 加载配置
    config = load_config()

    # 命令行覆盖配置
    if args.symbol:
        config["backtest"]["symbol"] = args.symbol
    if args.freq:
        config["backtest"]["data_freq"] = args.freq
    
    # ===== 修改点：数据源优先级：命令行 > 配置文件 > 默认tqsdk =====
    if args.source is None:
        # 从配置文件读取默认数据源
        args.source = config.get("default_data_source", "tqsdk")
    config["data_source"] = args.source
    
    if args.username:
        config.setdefault("tqsdk", {})["username"] = args.username
    if args.password:
        config.setdefault("tqsdk", {})["password"] = args.password

    # 创建日志器
    logger = setup_logger("backtest", log_file="results/backtest.log")

    # 天勤账号：优先从配置文件读取，命令行参数可覆盖
    tq_username = args.username or config.get("tqsdk", {}).get("username", "")
    print(f"[天勤账号] {tq_username if tq_username else '未设置'}")
    tq_password = args.password or config.get("tqsdk", {}).get("password", "")
    print(f"[天勤密码] {'已设置' if tq_password else '未设置'}")

    # 打印当前数据源
    print(f"[数据源] {args.source}")
    
    data_feed = DataFeed(
        data_dir=str(Path(__file__).parent / "data"),
        symbol=config["backtest"]["symbol"],
        freq=config["backtest"]["data_freq"],
        data_source=args.source,
        tq_username=tq_username,
        tq_password=tq_password,
        start_date=config["backtest"].get("start_date", "2024-01-01"),
        end_date=config["backtest"].get("end_date", "2024-12-31"),
    )

    # 如果指定了数据文件
    if args.data_file:
        data_feed.load(args.data_file)
    elif args.generate_data:
        data_path = Path(__file__).parent / "data" / "processed" / f"{config['backtest']['symbol']}.csv"
        DataFeed.generate_sample_data(
            n_bars=5000, freq=args.freq,
            symbol=config["backtest"]["symbol"],
            save_path=str(data_path),
        )
        print(f"[数据] 模拟数据已生成: {data_path}")
        return

    # 解析策略
    strategy_class, desc = resolve_strategy(args.strategy)
    print(f"[策略] {desc}")

    # ======================================
    # 参数优化模式
    # ======================================
    if args.optimize:
        from utils.optimizer import ParameterOptimizer

        param_grid = {
            "lookback": [8, 10, 12, 15],
            "recovery_ratio": [0.3, 0.4, 0.5],
            "stop_loss_atr": [1.0, 1.5, 2.0],
            "take_profit_atr": [2.0, 2.5, 3.0, 4.0],
        }

        if args.freq == "5min":
            param_grid["lookback"] = [6, 8, 10, 12]
            param_grid["recovery_ratio"] = [0.25, 0.35, 0.45]

        optimizer = ParameterOptimizer(
            strategy_class=strategy_class,
            config=config,
            data_feed=data_feed,
        )

        results = optimizer.grid_search(
            param_grid=param_grid,
            metric="sharpe_ratio",
            top_n=15,
            walk_forward=args.walk_forward,
        )

        optimizer.print_top_results(15)
        return

    # ======================================
    # 标准回测模式
    # ======================================
    backtester = Backtester(
        strategy_class=strategy_class,
        config=config,
        data_feed=data_feed,
        logger=logger,
    )

    broker = backtester.run()
    backtester.report(save_dir="results")

    # 显示最近交易
    closed = [t for t in broker.trades if t.pnl != 0]
    if closed:
        print(f"\n最近交易记录（显示前{min(10, len(closed))}笔）:")
        print("-" * 80)
        for t in closed[:10]:
            print(t)
        if len(closed) > 10:
            print(f"  ... 还有 {len(closed) - 10} 笔交易")


if __name__ == "__main__":
    main()