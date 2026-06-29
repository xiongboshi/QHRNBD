#!/usr/bin/env python3
"""
Web操作界面 - 期货日内波段回测系统
使用 lightweight-charts 实现类文华财经K线图
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, jsonify, request, send_file, session
from flask_cors import CORS
import pandas as pd
import json
import base64
import io
import matplotlib.pyplot as plt
from datetime import datetime

from engine.backtester import Backtester
from engine.data_feed import DataFeed
from engine.broker import Broker
from utils.logger import setup_logger

app = Flask(__name__)
app.secret_key = 'futures_backtest_secret_key_2026'
CORS(app)

# ==========================================================
# 策略注册表
# ==========================================================
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


# ==========================================================
# 回测执行函数
# ==========================================================
def run_backtest(strategy_name: str, symbol: str, freq: str,
                 start_date: str, end_date: str, source: str = "tqsdk"):
    """执行回测并返回结果"""
    
    # 加载配置
    from run import load_config
    config = load_config()
    
    # 更新配置
    config["backtest"]["symbol"] = symbol
    config["backtest"]["data_freq"] = freq
    config["backtest"]["start_date"] = start_date
    config["backtest"]["end_date"] = end_date
    config["data_source"] = source
    
    # 获取策略
    import importlib
    strategy_info = STRATEGY_REGISTRY.get(strategy_name)
    if not strategy_info:
        raise ValueError(f"未知策略: {strategy_name}")
    
    module = importlib.import_module(strategy_info["module"])
    strategy_class = getattr(module, strategy_info["class"])
    
    # 设置策略参数
    if "params" in strategy_info:
        config["strategy_params"] = strategy_info["params"]
    
    # 数据加载
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
    
    # 执行回测
    logger = setup_logger("web_backtest")
    backtester = Backtester(
        strategy_class=strategy_class,
        config=config,
        data_feed=data_feed,
        logger=logger,
    )
    
    broker = backtester.run()
    
    # 生成报告
    summary = broker.get_summary()
    
    # 获取交易数据（用于K线图）
    trades = []
    for t in broker.trades:
        trades.append({
            "time": str(t.time),
            "side": t.side,
            "price": t.price,
            "volume": t.volume,
            "pnl": t.pnl,
            "commission": t.commission,
            "entry": t.side == "buy",  # 用于标记
        })
    
    # 获取K线数据（全部数据，不采样）
    kline_data = []
    for idx, (dt, row) in enumerate(backtester._data.iterrows()):
        kline_data.append({
            "time": str(dt),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0)),
        })
    
    # 获取权益曲线
    equity_curve = [
        {"time": str(t[0]), "equity": float(t[1])}
        for t in broker.equity_curve
    ] if broker.equity_curve else []
    
    return {
        "summary": summary,
        "trades": trades,
        "kline_data": kline_data,
        "equity_curve": equity_curve,
        "daily_direction": backtester.daily_direction_records if hasattr(backtester, 'daily_direction_records') else [],
        "total_trades": len(trades),
    }


# ==========================================================
# 生成K线图HTML（使用lightweight-charts）
# ==========================================================
def generate_kline_html(kline_data, trades, symbol="rb888"):
    """生成K线图HTML代码（含进出场标记）"""
    
    if not kline_data:
        return "<div style='text-align:center;padding:50px;color:#6a7a8e;'>无K线数据</div>"
    
    # 准备数据
    import json as json_lib
    
    # 限制数据量（lightweight-charts最多显示8000根）
    if len(kline_data) > 6000:
        kline_data = kline_data[-6000:]
    
    # 准备交易标记
    markers = []
    for t in trades:
        markers.append({
            "time": t["time"],
            "price": t["price"],
            "type": "buy" if t["side"] == "buy" else "sell",
            "text": "多" if t["side"] == "buy" else "空",
        })
    
    # 生成HTML
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K线图</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            background: #0a0e17; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh;
            font-family: 'Microsoft YaHei', Arial, sans-serif;
        }}
        #chart {{
            width: 100%;
            height: 100vh;
        }}
        .chart-toolbar {{
            position: absolute;
            top: 10px;
            right: 20px;
            z-index: 100;
            display: flex;
            gap: 8px;
            background: rgba(10, 14, 23, 0.85);
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid #1a2332;
        }}
        .chart-toolbar button {{
            background: #1a2332;
            border: none;
            color: #e0e0e0;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }}
        .chart-toolbar button:hover {{
            background: #24344a;
        }}
        .chart-toolbar .info {{
            color: #6a7a8e;
            font-size: 12px;
            padding: 4px 8px;
        }}
    </style>
</head>
<body>
    <div id="chart"></div>
    <div class="chart-toolbar">
        <span class="info">{symbol} | {len(kline_data)}根K线 | {len(trades)}笔交易</span>
        <button onclick="fitContent()">📐 自适应</button>
    </div>

    <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        // K线数据
        const klineData = {json_lib.dumps(kline_data)};
        const trades = {json_lib.dumps(trades)};
        const markers = {json_lib.dumps(markers)};
        
        // 创建图表
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
            width: window.innerWidth,
            height: window.innerHeight,
            layout: {{
                background: {{ color: '#0a0e17' }},
                textColor: '#6a7a8e',
            }},
            grid: {{
                vertLines: {{ color: '#0d1421' }},
                horzLines: {{ color: '#0d1421' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {{
                    color: '#1a2332',
                    width: 1,
                    style: LightweightCharts.LineStyle.Solid,
                }},
                horzLine: {{
                    color: '#1a2332',
                    width: 1,
                    style: LightweightCharts.LineStyle.Solid,
                }},
            }},
            rightPriceScale: {{
                borderColor: '#1a2332',
                scaleMargins: {{
                    top: 0.05,
                    bottom: 0.25,
                }},
            }},
            timeScale: {{
                borderColor: '#1a2332',
                timeVisible: true,
                secondsVisible: false,
                tickMarkFormatter: function(time, tickMarkType) {{
                    const date = new Date(time * 1000);
                    const month = (date.getMonth() + 1).toString().padStart(2, '0');
                    const day = date.getDate().toString().padStart(2, '0');
                    const hours = date.getHours().toString().padStart(2, '0');
                    const mins = date.getMinutes().toString().padStart(2, '0');
                    return month + '-' + day + ' ' + hours + ':' + mins;
                }},
            }},
        }});
        
        // 主K线图
        const mainSeries = chart.addCandlestickSeries({{
            upColor: '#e74c3c',
            downColor: '#2ecc71',
            borderUpColor: '#e74c3c',
            borderDownColor: '#2ecc71',
            wickUpColor: '#e74c3c',
            wickDownColor: '#2ecc71',
        }});
        
        // 设置K线数据
        const formattedData = klineData.map(item => ({{
            time: new Date(item.time).getTime() / 1000,
            open: item.open,
            high: item.high,
            low: item.low,
            close: item.close,
        }}));
        mainSeries.setData(formattedData);
        
        // 添加进出场标记
        const markersData = markers.map(m => {{
            return {{
                time: new Date(m.time).getTime() / 1000,
                position: m.type === 'buy' ? 'aboveBar' : 'belowBar',
                color: m.type === 'buy' ? '#e74c3c' : '#2ecc71',
                shape: m.type === 'buy' ? 'arrowUp' : 'arrowDown',
                text: m.text,
                size: 2,
            }};
        }});
        mainSeries.setMarkers(markersData);
        
        // 添加成交量图（副图）
        const volumeSeries = chart.addHistogramSeries({{
            color: '#1a2332',
            priceFormat: {{
                type: 'volume',
            }},
            priceScaleId: 'volume',
            scaleMargins: {{
                top: 0.85,
                bottom: 0,
            }},
        }});
        
        const volumeData = formattedData.map((item, idx) => {{
            const vol = klineData[idx]?.volume || 0;
            const isUp = item.close >= item.open;
            return {{
                time: item.time,
                value: vol,
                color: isUp ? 'rgba(231, 76, 60, 0.5)' : 'rgba(46, 204, 113, 0.5)',
            }};
        }});
        volumeSeries.setData(volumeData);
        
        // 添加移动平均线（MA5, MA10, MA20）
        function calculateMA(data, period) {{
            const result = [];
            for (let i = period - 1; i < data.length; i++) {{
                let sum = 0;
                for (let j = 0; j < period; j++) {{
                    sum += data[i - j].close;
                }}
                result.push({{
                    time: data[i].time,
                    value: sum / period,
                }});
            }}
            return result;
        }}
        
        const ma5Data = calculateMA(formattedData, 5);
        const ma10Data = calculateMA(formattedData, 10);
        const ma20Data = calculateMA(formattedData, 20);
        
        const ma5Series = chart.addLineSeries({{
            color: '#f1c40f',
            lineWidth: 1,
            priceScaleId: 'right',
        }});
        ma5Series.setData(ma5Data);
        
        const ma10Series = chart.addLineSeries({{
            color: '#3498db',
            lineWidth: 1,
            priceScaleId: 'right',
        }});
        ma10Series.setData(ma10Data);
        
        const ma20Series = chart.addLineSeries({{
            color: '#9b59b6',
            lineWidth: 1,
            priceScaleId: 'right',
        }});
        ma20Series.setData(ma20Data);
        
        // 窗口自适应
        function fitContent() {{
            chart.timeScale().fitContent();
        }}
        
        window.addEventListener('resize', function() {{
            chart.resize(window.innerWidth, window.innerHeight);
        }});
        
        // 初始自适应
        setTimeout(fitContent, 500);
        
        // 键盘快捷键
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'r' || e.key === 'R') {{
                fitContent();
            }}
        }});
    </script>
</body>
</html>
    '''
    return html


# ==========================================================
# 路由
# ==========================================================

@app.route("/")
def index():
    """首页"""
    return render_template("index.html", strategies=STRATEGY_REGISTRY)


@app.route("/api/strategies")
def get_strategies():
    """获取策略列表"""
    return jsonify({
        "strategies": [
            {"name": k, "desc": v["desc"]}
            for k, v in STRATEGY_REGISTRY.items()
        ]
    })


@app.route("/api/run", methods=["POST"])
def run_backtest_api():
    """执行回测"""
    try:
        data = request.get_json()
        strategy_name = data.get("strategy", "pbx_u_shape")
        symbol = data.get("symbol", "rb888")
        freq = data.get("freq", "15min")
        start_date = data.get("start_date", "2025-07-01")
        end_date = data.get("end_date", "2026-06-29")
        source = data.get("source", "tqsdk")
        
        result = run_backtest(strategy_name, symbol, freq, start_date, end_date, source)
        
        # 存储K线数据到session供图表使用
        session['kline_data'] = result['kline_data']
        session['trades'] = result['trades']
        session['symbol'] = symbol
        
        return jsonify({"success": True, "data": result})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/kline_chart")
def get_kline_chart():
    """获取K线图HTML"""
    kline_data = session.get('kline_data', [])
    trades = session.get('trades', [])
    symbol = session.get('symbol', 'rb888')
    
    if not kline_data:
        return jsonify({"html": "<div style='text-align:center;padding:50px;color:#6a7a8e;'>请先执行回测</div>"})
    
    html = generate_kline_html(kline_data, trades, symbol)
    return jsonify({"html": html})


# ==========================================================
# 主程序
# ==========================================================

if __name__ == "__main__":
    # 确保模板和静态文件目录存在
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    template_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "css").mkdir(parents=True, exist_ok=True)
    (static_dir / "js").mkdir(parents=True, exist_ok=True)
    
    app.run(debug=True, host="0.0.0.0", port=5000)