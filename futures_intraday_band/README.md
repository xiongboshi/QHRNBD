# 期货日内波段回测系统

一个轻量级、模块化的期货日内波段回测框架，支持快速添加新策略。

## 目录结构

```
futures_intraday_band/
├── data/                     # 数据存储
│   ├── raw/                  # 原始行情
│   └── processed/            # 清洗后数据
├── engine/                   # 回测引擎核心
│   ├── backtester.py         # 主循环
│   ├── broker.py             # 模拟成交、风控
│   └── data_feed.py          # 数据加载与K线合成
├── strategies/               # 策略模块
│   ├── base_strategy.py      # 策略基类
│   ├── pbx_u_shape.py        # PBX+U形策略
│   └── ma_cross.py           # 均线交叉策略（示例）
├── utils/                    # 工具函数
│   ├── indicators.py         # 技术指标计算
│   └── logger.py             # 日志工具
├── config/
│   └── settings.yaml         # 配置文件
├── results/                  # 回测结果输出
├── run.py                    # 主入口
└── requirements.txt
```

## 快速开始

```bash
cd futures_intraday_band

# 安装依赖
pip install -r requirements.txt

# 运行 PBX+U形策略（自动生成模拟数据）
python run.py

# 运行均线交叉策略
python run.py -s ma_cross

# 仅生成模拟数据
python run.py --generate-data

# 指定品种和K线频率
python run.py -s pbx_u_shape --symbol i888 --freq 5min
```

## 添加新策略

1. 在 `strategies/` 下创建新文件，继承 `BaseStrategy`
2. 在 `run.py` 的 `STRATEGY_REGISTRY` 中注册
3. 运行 `python run.py -s your_strategy_name`

### 策略模板

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, broker, params=None):
        super().__init__(broker, params)

    def prepare(self, df):
        # 预计算指标
        pass

    def on_bar(self, bar):
        # 策略逻辑
        if not self.has_position and signal_buy:
            self.buy(1)
        elif self.has_position and signal_sell:
            self.sell(1)
```

## 配置说明

编辑 `config/settings.yaml` 可配置：
- 回测时间段、品种、K线频率
- 手续费、保证金、滑点
- 日内交易限制（次数、收盘平仓时间）
- 风险控制参数
