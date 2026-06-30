#!/usr/bin/env python3
"""
Web操作界面 - 期货日内波段回测系统
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd

from engine.backtester import Backtester
from engine.data_feed import DataFeed
from utils.logger import setup_logger

app = Flask(__name__)
app.secret_key = 'futures_backtest_secret_key_2026'
CORS(app)

STRATEGY_REGISTRY = {
    "pbx_u_shape": {
        "module": "strategies.pbx_u_shape",
        "class": "PbxUShapeStrategy",
        "desc": "PBX瀑布线 + U形底部形态（三层框架）",
        "params": {
            "pbx_periods": [4, 6, 9, 13, 18, 24],
            "pbx_trend_bars": 100,
            "lookback": 8,
            "recovery_ratio": 0.3,
            "min_drop_pct": 0.002,
            "u_shape_smooth": 3,
            "use_hour_filter": True,
            "hour_trend_bars": 100,
            "stop_loss_atr": 1.5,
            "take_profit_atr": 3.0,
            "fixed_contracts": 1,
            "max_positions_per_day": 3,
            "min_holding_bars": 2,
        }
    },
    "ma_cross": {
        "module": "strategies.ma_cross",
        "class": "MaCrossStrategy",
        "desc": "均线交叉策略",
        "params": {
            "fast_ma": 5,
            "slow_ma": 20,
            "stop_loss_atr": 1.5,
            "take_profit_atr": 3.0,
        }
    }
}


def resample_kline(df, target_freq):
    """重采样K线数据"""
    if target_freq == '15min':
        return df
    elif target_freq == '1h':
        return df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    elif target_freq == '1d':
        return df.resample('1d').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    else:
        return df


def run_backtest(strategy_name: str, symbol: str, freq: str,
                 start_date: str, end_date: str, source: str = "tqsdk"):
    """执行回测并返回结果"""
    from run import load_config
    config = load_config()

    config["backtest"]["symbol"] = symbol
    config["backtest"]["data_freq"] = freq
    config["backtest"]["start_date"] = start_date
    config["backtest"]["end_date"] = end_date
    config["data_source"] = source

    import importlib
    strategy_info = STRATEGY_REGISTRY.get(strategy_name)
    if not strategy_info:
        raise ValueError(f"未知策略: {strategy_name}")

    module = importlib.import_module(strategy_info["module"])
    strategy_class = getattr(module, strategy_info["class"])

    if "params" in strategy_info:
        config["strategy_params"] = strategy_info["params"]

    data_feed = DataFeed(
        data_dir=str(PROJECT_ROOT / "data"),
        symbol=config["backtest"]["symbol"],
        freq=config["backtest"]["data_freq"],
        data_source=config["data_source"],
        tq_username=config.get("tqsdk", {}).get("username", ""),
        tq_password=config.get("tqsdk", {}).get("password", ""),
        start_date=config["backtest"].get("start_date", ""),
        end_date=config["backtest"].get("end_date", ""),
    )

    logger = setup_logger("web_backtest")
    backtester = Backtester(
        strategy_class=strategy_class,
        config=config,
        data_feed=data_feed,
        logger=logger,
    )

    broker = backtester.run()
    summary = broker.get_summary()

    # 交易数据
    trades = []
    for t in broker.trades:
        trades.append({
            "time": str(t.time),
            "side": t.side,
            "price": t.price,
            "volume": t.volume,
            "pnl": t.pnl,
            "commission": t.commission,
        })

    # 获取原始数据（用于多周期展示）
    raw_data = None
    if backtester._data is not None:
        raw_data = backtester._data.copy()
    
    # ===== K线数据（多周期） =====
    kline_data = {}
    if raw_data is not None:
        # 15分钟
        df_15min = raw_data.iloc[-5000:]
        kline_data['15min'] = []
        for dt, row in df_15min.iterrows():
            kline_data['15min'].append({
                "time": str(dt),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
        
        # 1小时
        df_1h = resample_kline(raw_data, '1h').iloc[-2000:]
        kline_data['1h'] = []
        for dt, row in df_1h.iterrows():
            kline_data['1h'].append({
                "time": str(dt),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
        
        # 日线
        df_1d = resample_kline(raw_data, '1d').iloc[-500:]
        kline_data['1d'] = []
        for dt, row in df_1d.iterrows():
            kline_data['1d'].append({
                "time": str(dt),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })

    # ===== 权益曲线 =====
    equity_curve = []
    if hasattr(broker, 'equity_curve') and broker.equity_curve:
        for i, eq_value in enumerate(broker.equity_curve):
            equity_curve.append({
                "time": str(i),
                "equity": float(eq_value)
            })

    if not equity_curve:
        equity_curve = [
            {"time": "0", "equity": float(broker.initial_capital)}
        ]

    # ===== 提取支撑线数据 =====
    support_lines = []
    if hasattr(backtester, 'support_lines') and backtester.support_lines:
        for line in backtester.support_lines:
            support_lines.append({
                "type": line.get("type", "horizontal"),
                "price": float(line.get("price", 0)),
                "start_idx": int(line.get("start_idx", 0)),
                "time": line.get("time"),  # ✅ 新增：支撑线时间
            })
        print(f"[调试] 支撑线数量: {len(support_lines)}")

    return {
        "summary": summary,
        "trades": trades,
        "kline_data": kline_data,
        "equity_curve": equity_curve,
        "support_lines": support_lines,
        "total_trades": len(trades),
    }


@app.route("/")
def index():
    return render_template("index.html", strategies=STRATEGY_REGISTRY)


@app.route("/api/strategies")
def get_strategies():
    return jsonify({
        "strategies": [
            {"name": k, "desc": v["desc"]}
            for k, v in STRATEGY_REGISTRY.items()
        ]
    })


@app.route("/api/run", methods=["POST"])
def run_backtest_api():
    try:
        data = request.get_json()
        strategy_name = data.get("strategy", "pbx_u_shape")
        symbol = data.get("symbol", "rb888")
        freq = data.get("freq", "15min")
        start_date = data.get("start_date", "2025-07-01")
        end_date = data.get("end_date", "2026-06-29")
        source = data.get("source", "tqsdk")

        result = run_backtest(strategy_name, symbol, freq, start_date, end_date, source)
        return jsonify({"success": True, "data": result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    template_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "css").mkdir(parents=True, exist_ok=True)
    (static_dir / "js").mkdir(parents=True, exist_ok=True)

    app.run(debug=True, host="0.0.0.0", port=5000)