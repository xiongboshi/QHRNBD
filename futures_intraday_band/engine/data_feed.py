"""
数据加载与K线合成模块
负责：
1. 从 CSV/Parquet/Feather 加载本地数据
2. 从 天勤 TqSdk API 获取历史/实时数据
3. K线时间周期重采样（如1分钟→5分钟）
4. 统一数据格式输出
"""
from pathlib import Path
from typing import Optional, Literal

import pandas as pd


# 标准K线列名
OHLCV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume", "open_interest"]


class TqApiError(Exception):
    """天勤API连接异常"""
    pass


class DataFeed:
    """数据馈送器 - 加载并预处理行情数据

    支持三种数据来源（优先级从高到低）：
    1. local    : 本地文件（CSV/Parquet）
    2. tqsdk    : 天勤 TqSdk API
    3. sample   : 模拟数据（fallback）
    """

    def __init__(
        self,
        data_dir: str = "data",
        symbol: str = "rb888",
        freq: str = "1min",
        data_source: Literal["local", "tqsdk", "sample"] = "local",
        tq_username: str = "",
        tq_password: str = "",
        start_date: str = "",  # ✅ 新增
        end_date: str = "",    # ✅ 新增
    ):
        """
        Args:
            data_dir: 本地数据目录
            symbol: 品种代码（天勤格式如 KQ.m@SHFE.rb）
            freq: K线频率
            data_source: 数据来源
            tq_username: 天勤账号（data_source='tqsdk'时需要）
            tq_password: 天勤密码
        """
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        self.freq = freq
        self.data_source = data_source
        self._tq_account = (tq_username, tq_password)
        self._data: Optional[pd.DataFrame] = None
        self._tq_api = None  # 天勤API实例（懒加载）
        self.start_date = start_date  # 新增
        self.end_date = end_date      # 新增

    @property
    def data(self) -> pd.DataFrame:
        """获取加载后的数据"""
        if self._data is None:
            raise RuntimeError("数据尚未加载，请先调用 load() 方法")
        return self._data

    def load(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """加载数据

        自动根据 data_source 选择加载方式

        Args:
            filepath: 数据文件路径（仅 local 模式）
        """
        if self.data_source == "tqsdk":
            df = self._load_from_tqsdk()
        elif self.data_source == "sample":
            df = self._load_sample_data()
        else:
            df = self._load_from_local(filepath)

        # 按需重采样
        if self.freq != df.index.dtype:
            # 检查当前频率，如果不是目标频率则重采样
            pass

        if self.freq != "1min":
            df = self._resample(df, self.freq)

        self._data = df
        print(f"[DataFeed] 加载完成: {len(df)} 根K线, {df.index[0]} ~ {df.index[-1]}, 来源={self.data_source}")
        return df

    # ==========================================================
    # 方式一：天勤 TqSdk API
    # ==========================================================

    def _ensure_tq_api(self):
        """懒加载天勤API（新版SDK兼容）"""
        if self._tq_api is not None:
            return self._tq_api
        
        try:
            from tqsdk import TqApi, TqAuth
            from tqsdk import TqKq  # 新增导入
        except ImportError:
            raise ImportError(
                "使用天勤数据源需要安装 tqsdk:\n"
                "  pip install tqsdk\n\n"
                "天勤官网: https://www.shinnytech.com/tqsdk/"
            )

        username, password = self._tq_account
        if username and password:
            auth = TqAuth(username, password)
            self._tq_api = TqApi(auth=auth)
        else:
            print("[DataFeed] 天勤使用模拟账号模式，如需实盘数据请传入 username/password")
            self._tq_api = TqApi()

        return self._tq_api


    def _load_from_tqsdk(self) -> pd.DataFrame:
        """从天勤API获取K线数据"""
        try:
            from tqsdk import TqApi, TqAuth
        except ImportError:
            print("[DataFeed] 未安装 tqsdk，使用模拟数据")
            return self._load_sample_data()
        
        # 创建API实例（如果还没有）
        if self._tq_api is None:
            username, password = self._tq_account
            if username and password:
                self._tq_api = TqApi(auth=TqAuth(username, password))
            else:
                print("[DataFeed] 天勤账号未配置，使用模拟数据")
                return self._load_sample_data()
        
        api = self._tq_api

        try:
            tq_symbol = self._to_tq_symbol(self.symbol)
            
            # K线周期映射
            freq_map = {
                "1min": 60,
                "5min": 300,
                "15min": 900,
                "30min": 1800,
                "60min": 3600,
                "1h": 3600,
                "1H": 3600,
                "1d": 86400,
                "1D": 86400,
                "day": 86400,
            }
            seconds = freq_map.get(self.freq, 60)

            print(f"[DataFeed] 从天勤获取数据: {tq_symbol}, {self.freq}...")

            # 获取K线序列
            klines = api.get_kline_serial(tq_symbol, seconds, data_length=5000)
            
            # 等待数据就绪（只需一次）
            api.wait_update()
            
            # 提取数据
            df = klines.iloc[:].copy()
            
            if df is None or len(df) == 0:
                raise TqApiError(f"天勤未返回数据: {tq_symbol}")

            # ==========================================================
            # 时间处理：datetime是纳秒级时间戳
            # ==========================================================
            if "datetime" in df.columns:
                df["datetime"] = pd.to_numeric(df["datetime"])
                df["datetime"] = pd.to_datetime(df["datetime"] / 1_000_000_000, unit="s")
                df.set_index("datetime", inplace=True)
            
            df.sort_index(inplace=True)

            # 只保留OHLCV列
            keep_cols = ["open", "high", "low", "close", "volume"]
            if "open_oi" in df.columns:
                keep_cols.append("open_oi")
            if "close_oi" in df.columns:
                keep_cols.append("close_oi")
            
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df[keep_cols]

            api.close()
            print(f"[DataFeed] 天勤数据加载成功: {len(df)} 条, {df.index[0]} ~ {df.index[-1]}")
            return df

        except Exception as e:
            print(f"[DataFeed] 天勤数据获取失败: {e}")
            return self._load_sample_data()



    @staticmethod
    def _to_tq_symbol(symbol: str) -> str:
        """将简写合约代码转为天勤格式

        Examples:
            rb888 -> KQ.m@SHFE.rb
            i888  -> KQ.m@DCE.i
            MA888 -> KQ.m@CZCE.MA
        """
        symbol = symbol.upper()
        # 去掉数字后缀
        base = symbol.rstrip("0123456789")

        exchange_map = {
            "RB": "SHFE.rb", "CU": "SHFE.cu", "AL": "SHFE.al",
            "AU": "SHFE.au", "AG": "SHFE.ag", "ZN": "SHFE.zn",
            "I": "DCE.i", "JM": "DCE.jm", "J": "DCE.j",
            "MA": "CZCE.MA", "TA": "CZCE.TA", "FG": "CZCE.FG",
            "SC": "INE.sc", "LU": "INE.lu",
        }

        if base in exchange_map:
            return f"KQ.m@{exchange_map[base]}"

        # 如果是具体合约如 rb2405，直接返回
        if any(c.isdigit() for c in symbol):
            return symbol

        # 无法识别，保持原样
        print(f"[DataFeed] 警告: 无法映射合约 {symbol} 到天勤格式，直接使用")
        return symbol

    # ==========================================================
    # 方式二：本地文件（原始实现）
    # ==========================================================

    def _load_from_local(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """从本地文件加载数据"""
        if filepath is None:
            filepath = self._auto_find_file(use_processed=True)

        # 空路径或文件不存在 -> fallback 到 sample
        if not filepath:
            print("[DataFeed] 未找到数据文件，使用模拟数据")
            return self._load_sample_data()

        filepath = Path(filepath)
        if not filepath.exists():
            print(f"[DataFeed] 本地文件不存在: {filepath}，使用模拟数据")
            return self._load_sample_data()

        # 根据扩展名选择读取方式
        suffix = filepath.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(filepath)
        elif suffix == ".parquet":
            df = pd.read_parquet(filepath)
        elif suffix == ".feather":
            df = pd.read_feather(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {suffix} (文件: {filepath})")

        # 标准化列名
        df = self._standardize_columns(df)

        # 解析时间
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        # 去重
        df = df[~df.index.duplicated(keep="first")]

        return df

    # ==========================================================
    # 方式三：模拟数据（Fallback）
    # ==========================================================

    def _load_sample_data(self) -> pd.DataFrame:
        """生成并加载模拟数据"""
        print("[DataFeed] 生成模拟数据用于测试...")
        from datetime import datetime
        # 模拟2024年的数据
        year = 2024
        df = self.generate_sample_data(
            n_bars=5000,
            freq=self.freq,
            symbol=self.symbol,
            year=year,
        )
        # 确保 datetime 是索引
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)
        return df

    def get_slice(self, start: str, end: str) -> pd.DataFrame:
        """获取时间切片"""
        return self.data.loc[start:end].copy()

    def _auto_find_file(self, use_processed: bool) -> str:
        """自动查找数据文件"""
        if use_processed:
            proc_dir = self.data_dir / "processed"
            if proc_dir.exists():
                files = list(proc_dir.glob(f"{self.symbol}.*"))
                if files:
                    return str(files[0])

        raw_dir = self.data_dir / "raw"
        if raw_dir.exists():
            files = list(raw_dir.glob(f"{self.symbol}.*"))
            if files:
                return str(files[0])

        return None  # 返回 None，让上层 fallback 到 sample

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """将常见列名映射为标准命名"""
        col_mapping = {
            "date": "datetime", "time": "datetime", "timestamp": "datetime",
            "trading_date": "datetime", "candle_begin_time": "datetime",
            "opening": "open", "o": "open",
            "high_price": "high", "highest": "high", "h": "high",
            "low_price": "low", "lowest": "low", "l": "low",
            "closing": "close", "c": "close",
            "vol": "volume", "volumn": "volume", "v": "volume",
            "oi": "open_interest", "openint": "open_interest",
        }
        rename_map = {k: v for k, v in col_mapping.items()
                      if k in df.columns and v not in df.columns or k != v}
        df.rename(columns=rename_map, inplace=True)

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"数据缺少关键列: {col}")

        return df

    def _resample(self, df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """K线重采样"""
        ohlc_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        if "open_interest" in df.columns:
            ohlc_dict["open_interest"] = "last"

        resampled = df.resample(target_freq).agg(ohlc_dict)
        resampled.dropna(subset=["open", "close"], inplace=True)
        return resampled

    @staticmethod
    def generate_sample_data(
        n_bars: int = 10000,
        freq: str = "1min",
        start_price: float = 3800.0,
        volatility: float = 0.0003,
        symbol: str = "rb888",
        save_path: Optional[str] = None,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """生成模拟K线数据用于测试"""
        import numpy as np
        from datetime import datetime, timedelta

        if year:
            base = datetime(year, 1, 2, 9, 0, 0)  # 指定年份的1月2日
        else:
            now = datetime.now()
            base = now.replace(hour=9, minute=0, second=0, microsecond=0)
        while base.weekday() >= 5:
            base -= timedelta(days=1)
        timestamps = []
        t = base
        bar_minutes = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "60min": 60}
        step = bar_minutes.get(freq, 1)

        for _ in range(n_bars):
            timestamps.append(t)
            t += timedelta(minutes=step)
            if t.hour == 11 and t.minute > 30:
                t = t.replace(hour=13, minute=30)
            elif t.hour >= 15 and t.minute > 0:
                t += timedelta(hours=16)
                t = t.replace(hour=9, minute=0)
                while t.weekday() >= 5:
                    t += timedelta(days=1)

        np.random.seed(42)
        returns = np.random.normal(0, volatility, n_bars)
        price = start_price * np.exp(np.cumsum(returns))
        price = np.maximum(price, start_price * 0.8)

        highs = price * (1 + np.abs(np.random.normal(0, volatility * 0.5, n_bars)))
        lows = price * (1 - np.abs(np.random.normal(0, volatility * 0.5, n_bars)))
        opens = price * (1 + np.random.normal(0, volatility * 0.3, n_bars))
        volumes = np.random.randint(1000, 50000, n_bars)

        df = pd.DataFrame({
            "datetime": timestamps[:n_bars],
            "open": np.round(opens, 1),
            "high": np.round(np.maximum(highs, opens), 1),
            "low": np.round(np.minimum(lows, price), 1),
            "close": np.round(price, 1),
            "volume": volumes,
        })
        df["open_interest"] = np.random.randint(100000, 200000, n_bars)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix == ".csv":
                df.to_csv(save_path, index=False)
            elif save_path.suffix == ".parquet":
                df.to_parquet(save_path, index=False)
            print(f"[DataFeed] 模拟数据已保存至: {save_path}")

        return df


    def has_data(self) -> bool:
        """检查是否已加载数据"""
        return self._data is not None and len(self._data) > 0