"""
数据加载模块
"""
from pathlib import Path
import pandas as pd


class DataFeed:
    """数据加载类 - 不继承任何类"""

    def __init__(
        self,
        data_dir: str = "data",
        symbol: str = "rb888",
        freq: str = "1min",
        data_source: str = "local",
        tq_username: str = "",
        tq_password: str = "",
        start_date: str = "",
        end_date: str = "",
    ):
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        self.freq = freq
        self.data_source = data_source
        self.tq_username = tq_username
        self.tq_password = tq_password
        self.start_date = start_date
        self.end_date = end_date
        self._data = None

    def load(self, filepath=None):
        """加载数据"""
        if self.data_source == "tqsdk":
            return self._load_from_tqsdk()
        elif self.data_source == "sample":
            return self._load_sample_data()
        else:
            return self._load_from_local(filepath)

    def _load_from_tqsdk(self):
        """从天勤加载数据"""
        try:
            from tqsdk import TqApi, TqAuth
        except ImportError:
            print("[DataFeed] 请安装 tqsdk: pip install tqsdk")
            return self._load_sample_data()

        try:
            username = self.tq_username
            password = self.tq_password
            if not username or not password:
                print("[DataFeed] 天勤账号未配置，使用模拟数据")
                return self._load_sample_data()

            api = TqApi(auth=TqAuth(username, password))

            # 合约代码转换
            symbol = self.symbol.upper()
            base = symbol.rstrip("0123456789")
            exchange_map = {
                "RB": "SHFE.rb", "I": "DCE.i", "MA": "CZCE.MA",
                "CU": "SHFE.cu", "AU": "SHFE.au", "AG": "SHFE.ag",
                "J": "DCE.j", "JM": "DCE.jm", "FG": "CZCE.FG",
            }
            tq_symbol = f"KQ.m@{exchange_map[base]}" if base in exchange_map else symbol

            # 周期映射
            freq_map = {
                "1min": 60, "5min": 300, "15min": 900,
                "30min": 1800, "60min": 3600, "1h": 3600,
                "1d": 86400, "day": 86400,
            }
            seconds = freq_map.get(self.freq, 900)

            print(f"[DataFeed] 获取K线: {tq_symbol}, {self.freq}")

            klines = api.get_kline_serial(tq_symbol, seconds, data_length=5000)

            if klines is None or len(klines) == 0:
                print("[DataFeed] 天勤返回空数据")
                api.close()
                return self._load_sample_data()

            df = klines.copy()
            api.close()

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"] / 1_000_000_000, unit="s")
                df.set_index("datetime", inplace=True)

            keep = ["open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            df = df.dropna()

            print(f"[DataFeed] 天勤加载成功: {len(df)} 条K线")
            return df

        except Exception as e:
            print(f"[DataFeed] 天勤加载失败: {e}")
            return self._load_sample_data()

    def _load_from_local(self, filepath=None):
        """从本地加载"""
        if filepath is None:
            for sub in ["processed", "raw"]:
                p = self.data_dir / sub / f"{self.symbol}.csv"
                if p.exists():
                    filepath = str(p)
                    break

        if filepath is None:
            print("[DataFeed] 未找到本地数据，使用模拟数据")
            return self._load_sample_data()

        filepath = Path(filepath)
        if not filepath.exists():
            print(f"[DataFeed] 文件不存在: {filepath}")
            return self._load_sample_data()

        df = pd.read_csv(filepath)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

        print(f"[DataFeed] 本地加载成功: {len(df)} 条K线")
        return df

    def _load_sample_data(self):
        """生成模拟数据"""
        print("[DataFeed] 生成模拟数据...")
        import numpy as np
        from datetime import datetime, timedelta

        n_bars = 5000
        base = datetime(2024, 1, 2, 9, 0, 0)
        timestamps = []
        step = 15
        t = base
        for _ in range(n_bars):
            timestamps.append(t)
            t += timedelta(minutes=step)
            if t.hour >= 15 and t.minute > 0:
                t = t.replace(hour=9, minute=0) + timedelta(days=1)

        np.random.seed(42)
        price = 4000 * np.exp(np.cumsum(np.random.normal(0, 0.001, n_bars)))

        df = pd.DataFrame({
            "datetime": timestamps,
            "open": price * (1 + np.random.normal(0, 0.0005, n_bars)),
            "high": price * (1 + abs(np.random.normal(0, 0.001, n_bars))),
            "low": price * (1 - abs(np.random.normal(0, 0.001, n_bars))),
            "close": price,
            "volume": np.random.randint(1000, 50000, n_bars),
        })
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.dropna()

        print(f"[DataFeed] 模拟数据生成: {len(df)} 条")
        return df

    @property
    def data(self):
        return self._data