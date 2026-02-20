###main####
# -*- coding: utf-8 -*-
import os
import sys
import multiprocessing

# [환경 설정] 윈도우 및 오라클 클라우드 환경에서 라이브러리 간의 병렬 처리 충돌을 방지하기 위한 강제 설정
os.environ["TA_CORES"] = "0"
os.environ["PANDAS_TA_CORES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "1"

import asyncio
import ccxt.async_support as ccxt
from ccxt.base.errors import RateLimitExceeded, NetworkError, ExchangeNotAvailable, RequestTimeout, BadSymbol, InsufficientFunds

import pandas as pd
import numpy as np
import aiohttp
import json
import sqlite3
import traceback
import random
import time
import math
import functools
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# [봇 이벤트 클라이언트 (선택 사항)]
try:
    from bot_event_client import BotEventClient
except ImportError:
    BotEventClient = None

# [파일 경로 설정] 모든 로그와 데이터베이스는 절대 경로를 사용하여 데이터 유실 및 경로 오류 방지
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DB_FILE = os.environ.get("DASH_DB_FILE", os.path.join(BASE_DIR, "trading_log.db"))
ERROR_LOG_FILE = os.path.join(BASE_DIR, "bot_error.log")
CRASH_LOG_FILE = os.path.join(BASE_DIR, "bot_crash.log")

# [개선사항 A] 전역 스레드 풀 실행기 (CPU 연산용)
# 코어 수에 맞춰 워커 수 조정 (기본값: 코어수 * 2)
CPU_EXECUTOR = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 2))

# [공통] 로그 기록 함수 - 파일 시스템에 실시간 Flush 수행으로 데이터 쓰기 보장
def write_log(file_path, message, include_traceback=False):
    try:
        # 디렉토리 없으면 생성
        d = os.path.dirname(file_path)
        if d:
            os.makedirs(d, exist_ok=True)

        now_str = datetime.now().isoformat(sep=' ', timespec='seconds')
        log_msg = f"[{now_str}] {message}\n"
        
        if include_traceback:
            log_msg += traceback.format_exc() + "\n"

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(log_msg)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        # 절대 침묵하지 말고 최소 STDERR로 남김
        try:
            sys.stderr.write(f"[write_log_fail] path={file_path} err={e} msg={str(message)[:200]}\n")
            sys.stderr.flush()
        except Exception:
            pass

# -----------------------------------------------------------
# [모듈 0] 설정 매니저 - 전역 파라미터 및 리스크 관리 설정
# -----------------------------------------------------------
class Config:
    EXCHANGE_ID = "bybit"
    API_KEY = ""
    SECRET = ""

    IS_TEST_MODE = False  # 테스트 모드 여부 (True일 경우 바이비트 테스트넷 접속)
    TEST_SYMBOL = "BTC/USDT:USDT"
    QUOTE_CURRENCY = "USDT"

    BATCH_SIZE = 10  # 병렬 처리 단위

    # [메인 루프] 주기/배치/레버리지 상한
    MAIN_LOOP_SEC = 10
    MAIN_BATCH_SIZE = 10
    MAX_LEVERAGE = 3.0


    # [TF 구성]
    # [2단계 fetch용] 공통필터 TF / 엔트리 TF
    COMMON_TFS = ["4h", "1h", "30m", "15m"]
    ENTRY_TFS = ["1m", "5m"]

    # 전체 필요 TF 목록
    TIMEFRAMES = list(dict.fromkeys(COMMON_TFS + ENTRY_TFS))




    MAIN_TIMEFRAME = "30m"
    ENTRY_TIMEFRAME = "1m"

    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

    # [자산 관리] 총 시드 대비 목표 진입 비중 (10%)
    MAX_POSITION_PERCENT = 0.10

    # [추가 매수 기준]
    # - 보유증거금(current_m)이 목표 진입시드(target_m)의 80% 미만이면 추가매수(Top-up) 허용
    PYRAMIDING_THRESHOLD = 0.82

    # [리스크 관리] 1회 거래당 총 자산 대비 최대 허용 손실
    RISK_PER_TRADE_PERCENT = 0.009  ##백분으로 변환 기재

    # [시스템 보호] 총 자산 대비 실시간 손실 한도 도달 시 즉시 전량 청산
    HARD_STOP_LOSS_PERCENT = -0.009 ##0.9% 그대로 기재


    LEVERAGE_SAFETY_BUFFER = 1.00

    # [거래소 기능] 진입 시점 가격 대비 5% 트레일링 스탑 설정
    EXCHANGE_TRAILING_STOP_GAP = 0.05


    # [NEW] Trailing Stop (Bybit) - ATR% 기반 자동 산출
    TS_ATR_TF = "15m"          # ATR 계산 TF
    TS_ATR_PERIOD = 14
    TS_ATR_MULT = 2.0          # trail_pct = ATR% * MULT
    TS_MIN_PCT = 0.02          # 최소 2%
    TS_MAX_PCT = 0.15          # 최대 15% (요청값)
    TS_ACTIVATION_ATR_MULT = 0.8  # activation_pct = ATR% * 0.8 (min/max clamp)
    TS_ACTIVATION_MIN_PCT = 0.01  # 최소 1% 유리하게 진행 후 활성화
    TS_ACTIVATION_MAX_PCT = 0.05  # 최대 5% 유리하게 진행 후 활성화

    # [NEW] Time Stop - 진입 후 N봉 동안 진행(유리방향 이동)이 없으면 청산
    TIME_STOP_TF = "5m"
    TIME_STOP_BARS = 8                 # 8봉 = 40분 (요청값)
    TIME_STOP_PROGRESS_ATR_MULT = 0.5  # 진행 기준 = (trail_pct * 0.5) 이상

    # [NEW] SHORT 사용 여부
    ENABLE_SHORT = True

    EMA_FAST = 20  # EMA20
    EMA_SLOW = 60  # EMA60
    EMA_200 = 200  # EMA200
    EMA_TRAILING = 25  # EMA25: 기존 청산/추적용

    # [추가 - 1분봉 공통]
    EMA_SL_RESET_1M = 30  # 1분봉 스탑로스 재설정 기준선 EMA30
    EMA_EXIT_FAST_1M = 25  # 1분봉 전체매도 조건용 EMA25
    ATR_SL_1M_K = 0.20  # 1m 최초 SL: EMA20 아래로 ATR * 0.20

    # [추가 - 5분봉 전략]
    ATR_SL_5M_K = 0.20  # 5m 최초 SL: EMA20 아래로 ATR * 0.20 (원하면 0.10~0.30 튜닝)
    ATR_SL_15M_K = 0.20  # 15m SL 재설정: EMA20 아래로 ATR * 0.20 (기존과 동일)

    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    BB_LENGTH = 20
    BB_STD = 2.0

    ATR_PERIOD = 14
    ATR_TOUCH_K = 0.2  # 터치 판단 ATR 오차 범위
    ATR_SL_K = 0.1      # (기존) 손절가 산출 ATR 이격 범위(남겨둠)

    # [추가 - SuperTrend(5m 엔트리/SL 기준)]
    # SuperTrend (Wilder ATR 기반)
    ST_PERIOD = 10
    ST_MULTIPLIER = 2.0
    ST_COL = 'st_10_2'
    ST_DIR_COL = 'st_dir_10_2'

    # [추가 - 공통전략 필터]
    ADX_PERIOD = 14
    ADX_MIN_TREND = 20  # 15m ADX >=20 + 이전봉 대비 상승
    ADX_STRONG = 35     # 또는 15m ADX >=35

    # 공통 필터에서 BB 확장(30m/1h)을 '필수'로 강제할지 여부
    # - 기본 False: (기존 동작 유지) BB 확장은 참고용 메시지로만 사용
    # - True: 공통필터 통과에 BB 확장도 필요
    REQUIRE_COMMON_BB_EXPAND = False

    ZIGZAG_LEFT_BARS = 5
    ZIGZAG_RIGHT_BARS = 2

    ORDER_BOOK_DEPTH_STEP = 5  # 주문 시 참고할 호가 깊이
    HARD_STOP_DEPTH_STEP = 10  # 전량 매도시 참고할 호가 깊이

    EXECUTION_TIMEOUT = 1  # 지정가 대기 시간
    SCAN_DELAY = 0.5  # 루프 배치 간 딜레이
    ORDER_TIMEOUT_SECONDS = 60  # 미체결 주문 취소 기준

    # [API 보호] async 동시 호출 제한 + 재시도
    API_MAX_RETRIES = 4  # 재시도 횟수
    API_BACKOFF_BASE = 0.4  # 첫 백오프(초)
    API_BACKOFF_MAX = 6.0   # 최대 대기(초)

    # [EXIT 루프] 포지션 감시 주기
    EXIT_LOOP_SEC = 5.0

    # [EXIT 주문] 청산(매도/부분익절)은 빠르게 시장가로 처리
    EXIT_USE_MARKET = True

    # [EXIT 신호용] 5m 데이터 최소만(Full_Exit 판단/EMA trailing)
    EXIT_OHLCV_LIMIT = 120

    TOP_COIN_LIMIT = 300       # 전체 감시 Universe (활성도 점수로 확정된 200)
    CANDIDATE_LIMIT = 100      # Universe(200)에서 다시 후보군 100
    CANDIDATE_REFRESH_SEC = 30  # 후보군 갱신 주기(초)

    # [Universe Refresh] 유니버스 자체를 주기적으로 재선정
    UNIVERSE_REFRESH_SEC = 300   # 5분마다 유니버스 재선정(추천: 300~900)
    UNIVERSE_CACHE_SEC = 300     # _build_universe_by_activity 내부 캐시 버킷(기본 5분)



    # 스캔/청산 API 동시성 분리(청산이 밀리지 않게)
    API_CONCURRENCY_SCAN = 4
    API_CONCURRENCY_EXIT = 2

    # 진입 주문 속도 옵션
    ENTRY_TIME_IN_FORCE = "IOC"
    ENTRY_STATUS_WAIT_SEC = 0.8
    ENTRY_MARKET_FALLBACK = True

    # [1m MACD 개선 필터 기본값]
    MACD_NEAR_ATR = 0.05
    MACD_CAP_ATR = 0.20
    MACD_REQUIRE_ZERO_CROSS = False
    MACD_REQUIRE_POSITIVE = True

    # [PATCH] Universe 활성도 점수 산출용 (기존 흐름 유지하면서 추가)
    UNIVERSE_POOL_MULT = 3             # quoteVolume 상위 TOP_COIN_LIMIT*3을 풀로 잡고, 그 안에서 활성도 점수로 200 확정
    UNIVERSE_ACTIVITY_TFS = ["30m", "1h", "4h"]
    UNIVERSE_OHLCV_LIMIT = 30          # MA20 계산 가능한 최소치 확보
    UNIVERSE_WEIGHT_1H = 0.45
    UNIVERSE_WEIGHT_30M = 0.35
    UNIVERSE_WEIGHT_4H = 0.20

    UNIVERSE_POOL_N = 300

    @classmethod
    def load_config(cls, file_path=CONFIG_FILE):
        try:
            if not os.path.exists(file_path):
                print("[시스템] 설정 파일이 없습니다. 기본값을 사용합니다.")
                return "Config file missing"
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls.EXCHANGE_ID = data.get("exchange", cls.EXCHANGE_ID)
            cls.API_KEY = data.get("api_key", cls.API_KEY)
            cls.SECRET = data.get("secret", cls.SECRET)
            cls.TELEGRAM_BOT_TOKEN = data.get("telegram_bot_token", cls.TELEGRAM_BOT_TOKEN)
            cls.TELEGRAM_CHAT_ID = data.get("telegram_chat_id", cls.TELEGRAM_CHAT_ID)
            cls.IS_TEST_MODE = data.get("is_test_mode", cls.IS_TEST_MODE)
            cls.TEST_SYMBOL = data.get("test_symbol", cls.TEST_SYMBOL)

            # (선택) config.json에 있으면 오버라이드
            cls.ATR_SL_5M_K = float(data.get("atr_sl_5m_k", cls.ATR_SL_5M_K))
            cls.MACD_NEAR_ATR = float(data.get("macd_near_atr", cls.MACD_NEAR_ATR))
            cls.MACD_CAP_ATR = float(data.get("macd_cap_atr", cls.MACD_CAP_ATR))
            cls.MACD_REQUIRE_ZERO_CROSS = bool(data.get("macd_require_zero_cross", cls.MACD_REQUIRE_ZERO_CROSS))
            cls.MACD_REQUIRE_POSITIVE = bool(data.get("macd_require_positive", cls.MACD_REQUIRE_POSITIVE))

            # [PATCH] Universe 활성도 파라미터(있으면 오버라이드)
            cls.UNIVERSE_POOL_MULT = int(data.get("universe_pool_mult", cls.UNIVERSE_POOL_MULT))
            cls.UNIVERSE_OHLCV_LIMIT = int(data.get("universe_ohlcv_limit", cls.UNIVERSE_OHLCV_LIMIT))
            cls.UNIVERSE_WEIGHT_1H = float(data.get("universe_weight_1h", cls.UNIVERSE_WEIGHT_1H))
            cls.UNIVERSE_WEIGHT_30M = float(data.get("universe_weight_30m", cls.UNIVERSE_WEIGHT_30M))
            cls.UNIVERSE_WEIGHT_4H = float(data.get("universe_weight_4h", cls.UNIVERSE_WEIGHT_4H))
            tfs = data.get("universe_activity_tfs", None)
            if isinstance(tfs, list) and tfs:
                cls.UNIVERSE_ACTIVITY_TFS = [str(x) for x in tfs if x]

            return "Config 로드 완료"
        except Exception as e:
            return f"Config 로드 에러: {e}"


# -----------------------------------------------------------
# [모듈 1] 데이터베이스 매니저 - 거래 기록 저장 및 무결성 관리
# -----------------------------------------------------------
class DatabaseManager:
    def __init__(self, db_name=DB_FILE):
        self.db_name = db_name
        self.conn = None
        self.init_msg = ""
        self.setup_db()

    def setup_db(self):
        try:
            self.conn = sqlite3.connect(self.db_name, timeout=30, check_same_thread=False)
            cursor = self.conn.cursor()

            # 락 내성 강화
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=30000;")  # 30초 대기

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    quantity REAL,
                    leverage INTEGER,
                    strategy_note TEXT
                )
                """
            )
            self.conn.commit()
            self.init_msg = "DB 연결 및 스키마 확인 성공"
        except Exception as e:
            self.init_msg = f"DB 초기화 에러: {e}"
            write_log(ERROR_LOG_FILE, self.init_msg, include_traceback=True)

    def log_trade(self, symbol, side, price, quantity, leverage, note=""):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            now_iso = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO trades (timestamp, symbol, side, price, quantity, leverage, strategy_note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now_iso, symbol, side, price, quantity, leverage, note),
            )
            self.conn.commit()

            # save_to_csv가 있을 때만 호출
            if hasattr(self, "save_to_csv"):
                self.save_to_csv(now_iso, symbol, side, price, quantity, leverage, note)

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"DB 기록 에러: {e}", include_traceback=True)

    def close(self):
        if self.conn:
            self.conn.close()


# -----------------------------------------------------------
# [모듈 2] 텔레그램 매니저 - 실시간 상황 브리핑
# -----------------------------------------------------------
class TelegramManager:
    def __init__(self, bot_token: str = None, chat_id: str = None, *, timeout_sec: int = 15):
        # (1) 인자 받기: 여기 없어서 TypeError 났던 거
        self.bot_token = bot_token
        self.chat_id = str(chat_id) if chat_id is not None else None

        # (2) 기존 구조 유지용 필드
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=timeout_sec)

        # (3) 텔레그램 URL 생성
        self.base_url = (
            f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            if self.bot_token else None
        )

    async def start(self):
        # 기존 send_message_direct가 self.start()를 호출하니까 반드시 있어야 함
        if not self.base_url or not self.chat_id:
            return
        if self.session and not self.session.closed:
            return
        self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None

    async def send_message_direct(self, text):
        # 설정이 없으면 조용히 스킵 (봇 전체가 죽지 않게)
        if not self.base_url or not self.chat_id:
            return

        if not self.session or self.session.closed:
            await self.start()

        try:
            s = str(text).replace("\x00", "")  # 혹시 모를 NULL 문자 제거
            # 텔레그램 안전 길이(여유): "길이 기준"으로 자르는 게 제일 안전함
            chunks = [s[i:i+3500] for i in range(0, len(s), 3500)] or [""]

            for ch in chunks:
                payload = {"chat_id": self.chat_id, "text": ch}
                async with self.session.post(self.base_url, json=payload) as response:
                    if response.status != 200:
                        body = await response.text()
                        write_log(ERROR_LOG_FILE, f"텔레그램 전송 에러 code={response.status} body={body[:300]}")
        except Exception as e:
            write_log(ERROR_LOG_FILE, f"텔레그램 연결 장애: {e}")


# -----------------------------------------------------------
# [모듈 3] 데이터 매니저 - 거래소 연동 및 데이터 수집
# -----------------------------------------------------------
class DataManager:
    def __init__(self, exchange, telegram, bot):
        self.exchange = exchange
        self.telegram = telegram
        self.bot = bot



    async def fetch_timeframes(self, symbol, timeframes, limit=300):
        """
        지정한 timeframes만 fetch (심볼 내부 병렬화).
        - 하나라도 실패하면 None 반환(1단계 필터 정확성 우선)
        - DF 정렬/중복 제거/타임스탬프 이상치 제거 강화
        """
        try:
            tfs = list(timeframes)

            tasks = [
                self.bot._api_call(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    tf,
                    None,
                    int(limit),
                    tag=f"fetch_ohlcv:{symbol}:{tf}",
                )
                for tf in tfs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            symbol_ohlcvs = {}
            for tf, res in zip(tfs, results):
                if isinstance(res, Exception):
                    # 여기서 조용히 죽으면 원인 추적이 힘듦 -> 파일 로그 남김
                    write_log(ERROR_LOG_FILE, f"[DATA_ERROR] {symbol} fetch_ohlcv({tf}) EXC: {res}")
                    return None
                if not res:
                    write_log(ERROR_LOG_FILE, f"[DATA_ERROR] {symbol} fetch_ohlcv({tf}) EMPTY")
                    return None

                df = pd.DataFrame(res, columns=["timestamp", "open", "high", "low", "close", "volume"])
                if df.empty:
                    write_log(ERROR_LOG_FILE, f"[DATA_ERROR] {symbol} {tf} DF empty")
                    return None

                # timestamp 정제
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
                df = df.dropna(subset=["timestamp"])

                # 수치 정제
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["open", "high", "low", "close", "volume"])

                # 시간 정렬 + 중복 제거(거래소 가끔 중복/역전 데이터 줌)
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

                # 최소 길이 방어
                if len(df) < 30:
                    write_log(ERROR_LOG_FILE, f"[DATA_ERROR] {symbol} {tf} too short len={len(df)}")
                    return None

                symbol_ohlcvs[tf] = df

            return symbol_ohlcvs

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[DATA_ERROR] {symbol} fetch_timeframes FAIL: {e}", include_traceback=True)
            return None


    async def fetch_common_data(self, symbol, limit=300):
        return await self.fetch_timeframes(symbol, Config.COMMON_TFS, limit=limit)

    async def fetch_entry_data(self, symbol, limit=300):
        return await self.fetch_timeframes(symbol, Config.ENTRY_TFS, limit=limit)

    async def fetch_all_data(self, symbol, limit=300):
        """호환용(미사용): Config.TIMEFRAMES 전체 fetch"""
        return await self.fetch_timeframes(symbol, Config.TIMEFRAMES, limit=limit)

    async def get_target_price_by_orderbook(self, symbol, side, depth_step=5):
        """오더북 기반 목표가 산출 (동시성 제한 적용)"""
        try:
            limit_count = int(depth_step) + 10
            orderbook = await self.bot._api_call(
                self.exchange.fetch_order_book,
                symbol,
                limit=limit_count,
                tag=f"fetch_order_book:{symbol}",
            )
            if not orderbook:
                return None

            target_index = max(0, int(depth_step) - 1)
            asks = orderbook.get("asks", []) or []
            bids = orderbook.get("bids", []) or []

            if side == "buy":
                if len(asks) > target_index:
                    return asks[target_index][0]
                if asks:
                    return asks[0][0]

            if side == "sell":
                if len(bids) > target_index:
                    return bids[target_index][0]
                if bids:
                    return bids[0][0]

            return None
        except Exception:
            return None


# -----------------------------------------------------------
# [모듈 4] 기술적 분석기 - 수치 계산 및 매매 신호 판별
# -----------------------------------------------------------
class TechnicalAnalyzer:
    """
    [개선사항 A] Pandas 연산을 메인 스레드에서 분리하여 ThreadPoolExecutor에서 실행
    """

    @staticmethod
    def calculate_ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_ichimoku(high, low, close):
        twenty_six_h = high.rolling(window=26).max()
        twenty_six_l = low.rolling(window=26).min()
        kijun = (twenty_six_h + twenty_six_l) / 2
        return kijun

    @staticmethod
    def calculate_bb(series, length=20, std=2.0):
        s = series.astype(float)
        ma = s.rolling(window=length, min_periods=length).mean()
        sd = s.rolling(window=length, min_periods=length).std(ddof=1)  # 핵심: ddof=0
        upper = ma + (sd * float(std))
        lower = ma - (sd * float(std))
        return upper, lower


    @staticmethod
    def calculate_atr(df, period=14):
        h, l, c, pc = df["high"], df["low"], df["close"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_vwap(df):
        hlc3 = (df["high"] + df["low"] + df["close"]) / 3
        pv = hlc3 * df["volume"]
        dates = df["timestamp"].dt.date
        date_changed = dates != dates.shift(1)
        vwap = []
        cum_pv, cum_vol = 0, 0
        for i in range(len(df)):
            if date_changed.iloc[i]:
                cum_pv, cum_vol = pv.iloc[i], df["volume"].iloc[i]
            else:
                cum_pv += pv.iloc[i]
                cum_vol += df["volume"].iloc[i]
            vwap.append(cum_pv / (cum_vol + 1e-10))
        return pd.Series(vwap, index=df.index)

    @staticmethod
    def calculate_volume_ma(series, length=20):
        return series.rolling(window=length).mean()

    @staticmethod
    def calculate_adx(df, period=14):
        """
        ADX (Wilder)
        df: columns ['high','low','close'] 필요
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        up = high.diff()
        down = -low.diff()

        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = (
            100
            * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
            / (atr + 1e-10)
        )
        minus_di = (
            100
            * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
            / (atr + 1e-10)
        )

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx


    @staticmethod
    def calculate_supertrend(df, period=10, multiplier=3.0):
        """
        SuperTrend line + direction
        - df: columns ['high','low','close']
        - returns: (st_line: pd.Series, st_dir: pd.Series(+1/-1))
        """
        try:
            if df is None or df.empty or len(df) < (int(period) + 5):
                return None, None

            import numpy as np
            high = df['high'].astype(float).to_numpy()
            low = df['low'].astype(float).to_numpy()
            close = df['close'].astype(float).to_numpy()
            n = len(close)
            hl2 = (high + low) / 2.0

            # TR/ATR (Wilder RMA)
            tr = np.zeros(n, dtype=float)
            tr[0] = high[0] - low[0]
            prev_close = close[:-1]
            tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))

            atr = np.full(n, np.nan, dtype=float)
            p = int(period)
            if n >= p:
                atr[p - 1] = np.nanmean(tr[:p])
                for i in range(p, n):
                    atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p

            basic_ub = hl2 + float(multiplier) * atr
            basic_lb = hl2 - float(multiplier) * atr

            final_ub = np.full(n, np.nan, dtype=float)
            final_lb = np.full(n, np.nan, dtype=float)

            start = None
            for i in range(n):
                if not np.isnan(basic_ub[i]) and not np.isnan(basic_lb[i]):
                    final_ub[i] = basic_ub[i]
                    final_lb[i] = basic_lb[i]
                    start = i
                    break
            if start is None:
                return None, None

            for i in range(start + 1, n):
                if (basic_ub[i] < final_ub[i - 1]) or (close[i - 1] > final_ub[i - 1]):
                    final_ub[i] = basic_ub[i]
                else:
                    final_ub[i] = final_ub[i - 1]

                if (basic_lb[i] > final_lb[i - 1]) or (close[i - 1] < final_lb[i - 1]):
                    final_lb[i] = basic_lb[i]
                else:
                    final_lb[i] = final_lb[i - 1]

            st = np.full(n, np.nan, dtype=float)
            st_dir = np.full(n, 0, dtype=int)

            st[start] = final_lb[start] if close[start] >= final_lb[start] else final_ub[start]
            st_dir[start] = 1 if st[start] == final_lb[start] else -1

            for i in range(start + 1, n):
                prev_st = st[i - 1]
                if prev_st == final_ub[i - 1]:
                    st[i] = final_lb[i] if close[i] > final_ub[i] else final_ub[i]
                else:
                    st[i] = final_ub[i] if close[i] < final_lb[i] else final_lb[i]
                st_dir[i] = 1 if st[i] == final_lb[i] else -1

            import pandas as pd
            st_s = pd.Series(st, index=df.index)
            dir_s = pd.Series(st_dir, index=df.index)
            return st_s, dir_s
        except Exception:
            return None, None

    @staticmethod
    def _add_indicators_for_tfs_sync(ohlcvs_dict, tfs=None):
        """
        ohlcvs_dict에 지표를 추가 (동기 방식, 워커 스레드에서 실행됨)
        """
        if not ohlcvs_dict:
            return None

        target_tfs = list(ohlcvs_dict.keys()) if (tfs is None) else list(tfs)
        
        # 딕셔너리 원본 훼손 방지를 위해 필요한 경우 copy 사용 고려
        # 여기서는 DataFrame 자체를 수정하므로 얕은 복사로 충분치 않을 수 있으나
        # 보통 fetch된 DF는 독립적이므로 바로 수정
        
        for tf in target_tfs:
            df = ohlcvs_dict.get(tf)
            if df is None or df.empty:
                continue
            if len(df) < 60:
                continue

            df["ema_fast"] = TechnicalAnalyzer.calculate_ema(df["close"], Config.EMA_FAST)
            df["ema_slow"] = TechnicalAnalyzer.calculate_ema(df["close"], Config.EMA_SLOW)
            df["ema_200"] = TechnicalAnalyzer.calculate_ema(df["close"], Config.EMA_200)
            df["ema_trailing"] = TechnicalAnalyzer.calculate_ema(df["close"], Config.EMA_TRAILING)


            # [추가] 엔트리/청산용 EMA20
            df["ema20"] = TechnicalAnalyzer.calculate_ema(df["close"], 20)
            # 1m SL/Exit용 EMA30/EMA25(컬럼은 모든 TF에 만들어도 무방)
            df["ema_30"] = TechnicalAnalyzer.calculate_ema(df["close"], Config.EMA_SL_RESET_1M)
            df["ema_25"] = TechnicalAnalyzer.calculate_ema(df["close"], Config.EMA_EXIT_FAST_1M)

            df["rsi"] = TechnicalAnalyzer.calculate_rsi(df["close"], Config.RSI_PERIOD)

            m, s = TechnicalAnalyzer.calculate_macd(df["close"], Config.MACD_FAST, Config.MACD_SLOW, Config.MACD_SIGNAL)
            df["macd"], df["macd_signal"] = m, s
            df["macd_hist"] = m - s

            df["bb_upper"], df["bb_lower"] = TechnicalAnalyzer.calculate_bb(df["close"], Config.BB_LENGTH, Config.BB_STD)

            # [추가] BB 폭(upper-lower)
            try:
                df["bb_width"] = (df["bb_upper"].astype(float) - df["bb_lower"].astype(float)).abs()
            except Exception:
                pass

            df["atr"] = TechnicalAnalyzer.calculate_atr(df, Config.ATR_PERIOD)
            df["vwap"] = TechnicalAnalyzer.calculate_vwap(df)

            # [추가] SuperTrend (기본: 10,2)
            try:
                st_line, st_dir = TechnicalAnalyzer.calculate_supertrend(
                    df[["high","low","close"]].copy(),
                    period=getattr(Config, 'ST_PERIOD', 10),
                    multiplier=getattr(Config, 'ST_MULTIPLIER', 2.0)
                )
                if st_line is not None and st_dir is not None:
                    df[getattr(Config, 'ST_COL', 'st_10_2')] = st_line
                    df[getattr(Config, 'ST_DIR_COL', 'st_dir_10_2')] = st_dir
                    # [호환] 별칭 컬럼
                    df["st"] = st_line
                    df["st_dir"] = st_dir
            except Exception:
                pass

            if "volume" in df.columns:
                df["vol_ma20"] = TechnicalAnalyzer.calculate_volume_ma(df["volume"], 20)

            try:
                df["IKS_26"] = TechnicalAnalyzer.calculate_ichimoku(df["high"], df["low"], df["close"])
            except Exception:
                pass

            try:
                df["adx"] = TechnicalAnalyzer.calculate_adx(df, Config.ADX_PERIOD)
            except Exception:
                pass

            ohlcvs_dict[tf] = df

        return ohlcvs_dict

    @staticmethod
    async def add_indicators_for_tfs(ohlcvs_dict, tfs=None):
        """
        [비동기 래퍼] 지표 계산을 스레드 풀로 위임
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                CPU_EXECUTOR, 
                TechnicalAnalyzer._add_indicators_for_tfs_sync, 
                ohlcvs_dict, 
                tfs
            )
        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[TA_CALC_ERR] {e}", include_traceback=True)
            return None

    @staticmethod
    def add_indicators(ohlcvs_dict):
        # 호환성 유지용 (동기)
        return TechnicalAnalyzer._add_indicators_for_tfs_sync(ohlcvs_dict, tfs=None)


    @staticmethod
    def _bb_expand_2bars(df):
        """
        '진짜 조개 확장' (strict) 2연속 체크:
        - upper는 이전봉 대비 상승
        - lower는 이전봉 대비 하락
        이 조건이 2번 연속 발생해야 True
        """
        try:
            if df is None or df.empty:
                return False, ""
            if ("bb_upper" not in df.columns) or ("bb_lower" not in df.columns):
                return False, ""
            if len(df) < 4:
                return False, ""

            u0 = float(df["bb_upper"].iloc[-1])
            l0 = float(df["bb_lower"].iloc[-1])
            u1 = float(df["bb_upper"].iloc[-2])
            l1 = float(df["bb_lower"].iloc[-2])
            u2 = float(df["bb_upper"].iloc[-3])
            l2 = float(df["bb_lower"].iloc[-3])

            exp_prev = (u1 > u2) and (l1 < l2)
            exp_now = (u0 > u1) and (l0 < l1)

            du_prev = u1 - u2
            dl_prev = l1 - l2
            du_now = u0 - u1
            dl_now = l0 - l1

            if exp_prev and exp_now:
                return True, f"EXP2 strict | du:{du_prev:.6f}->{du_now:.6f}, dl:{dl_prev:.6f}->{dl_now:.6f}"
            else:
                return False, f"NO strict | prev(du={du_prev:.6f},dl={dl_prev:.6f}) now(du={du_now:.6f},dl={dl_now:.6f})"
        except Exception:
            return False, ""


    @staticmethod
    def _bb_shell_expand_1bar(df):
        """
        1봉 '진짜 조개 확장' 체크 (현재봉 기준):
        - bb_upper[-1] > bb_upper[-2]
        - bb_lower[-1] < bb_lower[-2]
        """
        try:
            if df is None or df.empty:
                return False, ""
            if ("bb_upper" not in df.columns) or ("bb_lower" not in df.columns):
                return False, ""
            if len(df) < 3:
                return False, ""

            upper_now = float(df["bb_upper"].iloc[-1])
            upper_prev = float(df["bb_upper"].iloc[-2])
            lower_now = float(df["bb_lower"].iloc[-1])
            lower_prev = float(df["bb_lower"].iloc[-2])

            ok = (upper_now > upper_prev) and (lower_now < lower_prev)
            msg = f"BB_SHELL_1bar UP:{upper_prev:.6f}->{upper_now:.6f}, DN:{lower_prev:.6f}->{lower_now:.6f}"
            return ok, msg
        except Exception:
            return False, ""




    @staticmethod
    def _check_common_conditions_sync_long(dfs_common):
        """
        [LONG 공통조건]
        - 기존 check_common_conditions 로직을 '롱 전용'으로 분리
        - 반환: (ok: bool, msg: str)
        """
        try:
            df4 = dfs_common.get("4h")
            df1 = dfs_common.get("1h")
            df30 = dfs_common.get("30m")
            df15 = dfs_common.get("15m")

            if df4 is None or df1 is None or df30 is None or df15 is None:
                return False, ""

            # 마지막 값 추출
            r4 = float(df4["rsi"].iloc[-1])
            r1 = float(df1["rsi"].iloc[-1])
            r30 = float(df30["rsi"].iloc[-1])
            r15 = float(df15["rsi"].iloc[-1])

            m30_h = float(df30["macd_hist"].iloc[-1])
            m1_h = float(df1["macd_hist"].iloc[-1])

            # 볼린저/조개(4h)
            bb4_upper = df4.get("bb_upper")
            bb4_lower = df4.get("bb_lower")
            bb4_width = None
            if bb4_upper is not None and bb4_lower is not None:
                try:
                    bb4_width = float((bb4_upper - bb4_lower).iloc[-1])
                except Exception:
                    bb4_width = None

            bb4_shell = False
            try:
                # "조개" 조건: 최근 N봉에서 폭이 축소(정체) 후 한봉 확장 직전/직후 느낌
                # (기존 로직과 최대한 유사하게 유지)
                if bb4_width is not None:
                    w = df4["bb_width"].astype(float) if "bb_width" in df4.columns else None
                    if w is not None and len(w) >= 25:
                        w_now = float(w.iloc[-1])
                        w_min = float(w.iloc[-20:].min())
                        bb4_shell = (w_now <= (w_min * 1.25))
            except Exception:
                bb4_shell = False

            # RSI 기반 강세: bb4_shell 의존성 제거 (BB수축 없어도 진입 가능)
            # bb4_shell=True이면 기준 완화, False이면 더 높은 RSI 요구
            if bb4_shell:
                cond_r4_strong = (r4 >= 63) or (r30 >= 65 and r1 >= 65 and r4 >= 58)
                cond_rsi_strong = cond_r4_strong and (r1 >= 55) and (r30 >= 55) and (r15 >= 50)
            else:
                cond_r4_strong = (r4 >= 67) or (r30 >= 68 and r1 >= 68 and r4 >= 62)
                cond_rsi_strong = cond_r4_strong and (r1 >= 58) and (r30 >= 58) and (r15 >= 53)

            # MACD 히스토그램 양수(모멘텀)
            cond_macd = (m30_h > 0.0) or (m1_h > 0.0)

            # 15m 거래량: 현재봉 > 20MA
            v15 = df15["volume"].astype(float)
            v15_ma20 = v15.rolling(20).mean()
            cond_vol = False
            if len(v15) >= 25:
                cond_vol = float(v15.iloc[-1]) > float(v15_ma20.iloc[-1])

            # ADX 트렌드 필터(15m)
            cond_adx = False
            try:
                adx = df15.get("adx")
                if adx is not None and len(adx) >= 3:
                    adx_now = float(adx.iloc[-1])
                    adx_prev = float(adx.iloc[-2])
                    cond_adx = (adx_now >= float(Config.ADX_STRONG)) or (
                        (adx_now >= float(Config.ADX_MIN_TREND)) and (adx_now >= adx_prev)
                    )
            except Exception:
                cond_adx = False

            # BB 확장(30m 또는 1h) - 선택적 강제
            cond_bb_expand = False
            bb_msg = ""
            try:
                ok30, msg30 = TechnicalAnalyzer._bb_shell_expand_1bar(df30)
                if ok30:
                    cond_bb_expand = True
                    bb_msg = "BB_Exp_30m"
                else:
                    ok1, msg1 = TechnicalAnalyzer._bb_shell_expand_1bar(df1)
                    if ok1:
                        cond_bb_expand = True
                        bb_msg = "BB_Exp_1h"
            except Exception:
                cond_bb_expand = False

            if getattr(Config, "REQUIRE_COMMON_BB_EXPAND", False):
                if not cond_bb_expand:
                    return False, ""

            ok = cond_rsi_strong and cond_macd and cond_vol and cond_adx

            if ok:
                msg = (
                    f"[COMMON_LONG] RSI 4h/1h/30m/15m={r4:.2f}/{r1:.2f}/{r30:.2f}/{r15:.2f} | "
                    f"MACD_hist(30m/1h)={m30_h:.4f}/{m1_h:.4f} | "
                    f"VOL15>{'MA20' if cond_vol else 'x'} | "
                    f"ADX15={'OK' if cond_adx else 'x'}"
                )
                if cond_bb_expand:
                    msg += f" | {bb_msg}"
                return True, msg

            return False, ""

        except Exception:
            return False, ""

    @staticmethod
    def _check_common_conditions_sync_short(dfs_common):
        """
        [SHORT 공통조건]
        - LONG 공통의 '반대 축'을 기본으로 하되, BB확장/ADX/거래량은 동일하게 유지
        - 반환: (ok: bool, msg: str)
        """
        try:
            df4 = dfs_common.get("4h")
            df1 = dfs_common.get("1h")
            df30 = dfs_common.get("30m")
            df15 = dfs_common.get("15m")

            if df4 is None or df1 is None or df30 is None or df15 is None:
                return False, ""

            r4 = float(df4["rsi"].iloc[-1])
            r1 = float(df1["rsi"].iloc[-1])
            r30 = float(df30["rsi"].iloc[-1])
            r15 = float(df15["rsi"].iloc[-1])

            m30_h = float(df30["macd_hist"].iloc[-1])
            m1_h = float(df1["macd_hist"].iloc[-1])

            # 4h BB 조개(수축)
            bb4_shell = False
            try:
                w = df4["bb_width"].astype(float) if "bb_width" in df4.columns else None
                if w is not None and len(w) >= 25:
                    w_now = float(w.iloc[-1])
                    w_min = float(w.iloc[-20:].min())
                    bb4_shell = (w_now <= (w_min * 1.25))
            except Exception:
                bb4_shell = False

            # RSI 기반 약세(숏): bb4_shell 의존성 제거 (BB수축 없어도 진입 가능)
            # bb4_shell=True이면 기준 완화, False이면 더 낮은 RSI 요구
            if bb4_shell:
                cond_r4_weak = (r4 <= 37) or (r30 <= 35 and r1 <= 35 and r4 <= 42)
                cond_rsi_weak = cond_r4_weak and (r1 <= 45) and (r30 <= 45) and (r15 <= 50)
            else:
                cond_r4_weak = (r4 <= 33) or (r30 <= 32 and r1 <= 32 and r4 <= 38)
                cond_rsi_weak = cond_r4_weak and (r1 <= 42) and (r30 <= 42) and (r15 <= 47)

            # MACD 히스토그램 음수(모멘텀)
            cond_macd = (m30_h < 0.0) or (m1_h < 0.0)

            # 15m 거래량: 현재봉 > 20MA (변동성/체결활성)
            v15 = df15["volume"].astype(float)
            v15_ma20 = v15.rolling(20).mean()
            cond_vol = False
            if len(v15) >= 25:
                cond_vol = float(v15.iloc[-1]) > float(v15_ma20.iloc[-1])

            # ADX 트렌드 필터(15m): 방향 무관(강한 추세면 OK)
            cond_adx = False
            try:
                adx = df15.get("adx")
                if adx is not None and len(adx) >= 3:
                    adx_now = float(adx.iloc[-1])
                    adx_prev = float(adx.iloc[-2])
                    cond_adx = (adx_now >= float(Config.ADX_STRONG)) or (
                        (adx_now >= float(Config.ADX_MIN_TREND)) and (adx_now >= adx_prev)
                    )
            except Exception:
                cond_adx = False

            # BB 확장(30m 또는 1h)
            cond_bb_expand = False
            bb_msg = ""
            try:
                ok30, msg30 = TechnicalAnalyzer._bb_shell_expand_1bar(df30)
                if ok30:
                    cond_bb_expand = True
                    bb_msg = "BB_Exp_30m"
                else:
                    ok1, msg1 = TechnicalAnalyzer._bb_shell_expand_1bar(df1)
                    if ok1:
                        cond_bb_expand = True
                        bb_msg = "BB_Exp_1h"
            except Exception:
                cond_bb_expand = False

            if getattr(Config, "REQUIRE_COMMON_BB_EXPAND", False):
                if not cond_bb_expand:
                    return False, ""

            ok = cond_rsi_weak and cond_macd and cond_vol and cond_adx

            if ok:
                msg = (
                    f"[COMMON_SHORT] RSI 4h/1h/30m/15m={r4:.2f}/{r1:.2f}/{r30:.2f}/{r15:.2f} | "
                    f"MACD_hist(30m/1h)={m30_h:.4f}/{m1_h:.4f} | "
                    f"VOL15>{'MA20' if cond_vol else 'x'} | "
                    f"ADX15={'OK' if cond_adx else 'x'}"
                )
                if cond_bb_expand:
                    msg += f" | {bb_msg}"
                return True, msg

            return False, ""

        except Exception:
            return False, ""

    @staticmethod
    async def check_common_conditions_sides(dfs_common):
        """
        LONG/SHORT 공통조건을 동시에 평가.
        return: (long_ok, short_ok, msg_long, msg_short)
        """
        loop = asyncio.get_running_loop()
        long_ok, msg_long = await loop.run_in_executor(
            CPU_EXECUTOR, TechnicalAnalyzer._check_common_conditions_sync_long, dfs_common
        )
        short_ok, msg_short = await loop.run_in_executor(
            CPU_EXECUTOR, TechnicalAnalyzer._check_common_conditions_sync_short, dfs_common
        )
        return bool(long_ok), bool(short_ok), str(msg_long or ""), str(msg_short or "")

    @staticmethod
    async def check_common_conditions(dfs_common):
        """
        (호환) 기존 호출부 유지용: LONG 공통조건만 반환.
        """
        loop = asyncio.get_running_loop()
        ok, msg = await loop.run_in_executor(
            CPU_EXECUTOR, TechnicalAnalyzer._check_common_conditions_sync_long, dfs_common
        )
        return bool(ok), str(msg or "")

    @staticmethod
    def _check_signals_sync(ohlcvs_dict, side: str = "long"):
        """
        엔트리 시그널 (5m)
        - side: "long" or "short"
        return:
          (sig: bool, full_exit_sig: bool, entry_sl: float, strategy_name: str, sl_source: str, common_msg: str, details: dict)
        """
        try:
            side = str(side or "long").lower().strip()
            df1 = ohlcvs_dict.get("1m")
            df5 = ohlcvs_dict.get("5m")
            df15 = ohlcvs_dict.get("15m")  # 선택적: REQUIRE_ENTRY_BB_EXPAND 시에만 필수

            # 1m/5m은 필수, 15m은 BB게이트 옵션이 켜진 경우만 필수
            if df1 is None or df5 is None:
                return False, False, 0.0, "", "", "", {}

            # 5m 지표
            if "ema20" not in df5.columns or "atr" not in df5.columns or "st" not in df5.columns or "st_dir" not in df5.columns:
                return False, False, 0.0, "", "", "", {}

            # 현재가/최근캔들
            c_now = float(df5["close"].iloc[-1])
            o_now = float(df5["open"].iloc[-1])
            h_now = float(df5["high"].iloc[-1])
            l_now = float(df5["low"].iloc[-1])

            c_prev = float(df5["close"].iloc[-2])
            o_prev = float(df5["open"].iloc[-2])
            h_prev = float(df5["high"].iloc[-2])
            l_prev = float(df5["low"].iloc[-2])

            atr_now = float(df5["atr"].iloc[-1] or 0.0)
            if atr_now <= 0:
                atr_now = 0.0

            st_now = float(df5["st"].iloc[-1] or 0.0)
            st_prev = float(df5["st"].iloc[-2] or 0.0)
            dir_now = float(df5["st_dir"].iloc[-1] or 0.0)
            dir_prev = float(df5["st_dir"].iloc[-2] or 0.0)

            # 터치 허용오차(ATR 기반)
            tol_k = float(getattr(Config, "ATR_TOUCH_K", 0.0) or 0.0)
            tol = (atr_now * tol_k) if (atr_now > 0 and tol_k > 0) else 0.0

            details = {
                "side": side,
                "c_now": c_now,
                "st_now": st_now,
                "st_prev": st_prev,
                "dir_now": dir_now,
                "dir_prev": dir_prev,
                "atr_now": atr_now,
                "tol": tol,
            }

            # 공통 msg는 상위에서 넣되, 여기선 비워둠
            common_msg = ""

            # full_exit 시그널은 포지션 보유 시 exit_loop에서 처리 (여기선 False 고정)
            full_exit = False

            # 15m BB 확장 게이트(둘 다 동일)
            # - 이미 공통에서 강제할 수 있으나, 엔트리에서도 한 번 더 게이트(옵션)
            if getattr(Config, "REQUIRE_ENTRY_BB_EXPAND", False):
                if df15 is None:
                    return False, full_exit, 0.0, "", "", "", details
                ok15, _ = TechnicalAnalyzer._bb_shell_expand_1bar(df15)
                if not ok15:
                    return False, full_exit, 0.0, "", "", "", details

            bull_prev = (c_prev > o_prev)
            bull_now = (c_now > o_now)
            bear_prev = (c_prev < o_prev)
            bear_now = (c_now < o_now)

            # -------------------------
            # LONG: 5m ST(10,2) 터치 + ST 상승 + 2연속 양봉
            # -------------------------
            if side == "long":
                touch_prev = (l_prev <= (st_prev + tol)) and (c_prev >= st_prev)
                st_rise = (st_now >= st_prev) and (dir_prev > 0) and (dir_now > 0)

                if touch_prev and st_rise and bull_prev and bull_now:
                    # SL: max(min(low_now, low_prev), st_now) (롱은 아래)
                    sl_raw = max(min(l_now, l_prev), st_now)
                    details["touch_prev"] = touch_prev
                    details["st_rise"] = st_rise
                    details["bull_prev"] = bull_prev
                    details["bull_now"] = bull_now
                    details["sl_raw"] = sl_raw

                    return True, full_exit, float(sl_raw), "Buy_5M_ST10_2_Touch", "ST_5m", common_msg, details

                return False, full_exit, 0.0, "", "", "", details

            # -------------------------
            # SHORT: 5m ST(10,2) 리테스트(위쪽 터치) + ST 하락 + 2연속 음봉
            # -------------------------
            if side == "short":
                # 숏: 슈퍼트렌드가 하방(-1)이고, 이전봉 고가가 ST 근처까지 닿았다가(리테스트) 종가가 ST 아래
                touch_prev = (h_prev >= (st_prev - tol)) and (c_prev <= st_prev)
                st_fall = (st_now <= st_prev) and (dir_prev < 0) and (dir_now < 0)

                if touch_prev and st_fall and bear_prev and bear_now:
                    # SL: min(max(high_now, high_prev), st_now) (숏은 위)
                    sl_raw = min(max(h_now, h_prev), st_now)
                    # SL이 현재가 위에 있어야 함
                    if sl_raw <= c_now:
                        sl_raw = max(h_now, h_prev, st_now)

                    details["touch_prev"] = touch_prev
                    details["st_fall"] = st_fall
                    details["bear_prev"] = bear_prev
                    details["bear_now"] = bear_now
                    details["sl_raw"] = sl_raw

                    return True, full_exit, float(sl_raw), "Sell_5M_ST10_2_Retest", "ST_5m", common_msg, details

                return False, full_exit, 0.0, "", "", "", details

            return False, full_exit, 0.0, "", "", "", details

        except Exception:
            return False, False, 0.0, "", "", "", {}


    @staticmethod
    async def check_signals(ohlcvs_dict, side: str = "long"):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            CPU_EXECUTOR,
            TechnicalAnalyzer._check_signals_sync,
            ohlcvs_dict,
            side,
        )


# -----------------------------------------------------------
# [모듈 5] 리스크 매니저 - 자산 비중 및 레버리지 계산
# -----------------------------------------------------------
from decimal import Decimal, ROUND_HALF_UP

# -----------------------------------------------------------
# [모듈 5] 리스크 매니저 - 자산 비중 및 레버리지 계산 (교체)
# -----------------------------------------------------------
class RiskManager:
    @staticmethod
    def _snap_to_step(value: float, step: float) -> float:
        """
        거래소가 허용하는 레버리지 step(예: 0.5, 0.1)에 맞춰 정규화.
        - step 외 값은 거래소가 거부할 수 있어 필수.
        - 특정 방향(내림/올림) 강제가 아니라, "가장 가까운 step"으로 맞춤.
        """
        v = float(value)
        s = float(step) if step and float(step) > 0 else 0.0
        if s <= 0:
            return v

        dv = Decimal(str(v))
        ds = Decimal(str(s))

        # 가장 가까운 스텝으로 스냅 (ROUND_HALF_UP)
        n = (dv / ds).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        out = n * ds
        return float(out)

    @staticmethod
    def calculate_entry_params(
        balance: float,
        price: float,
        sl: float,
        max_lev: float,
        current_m: float = 0.0,
        leverage_step: float = 0.1,
        min_leverage: float = 1.5,  # ★ 최소 레버리지 1.5배
    ):
        """
        return: (final_lev, qty, needed_m)

        - final_lev: float 유지
        - 최소레버리지: 1.5배 강제
        - 최대레버리지: 심볼 max_lev 이내
        - 스텝: 심볼 leverage_step(0.5/0.1 등)에 맞춤
        """
        if balance <= 0 or price <= 0 or sl <= 0:
            return float(min_leverage), 0.0, 0.0

        # 목표 증거금 (예: balance의 10%)
        target_m = float(balance) * float(Config.MAX_POSITION_PERCENT)

        # SL 거리 비율
        gap = abs(float(price) - float(sl)) / float(price)
        if gap < 0.001:
            gap = 0.001

        # 리스크 고정 설계 그대로
        calc_lev = float(Config.RISK_PER_TRADE_PERCENT) / (float(Config.MAX_POSITION_PERCENT) * float(gap))

        # 심볼 최대 레버리지 내로 clamp + 최소 1.5 강제 (여기서는 소숫점 유지)
        max_lev_f = float(max_lev) if max_lev and float(max_lev) > 0 else 1.0
        lev = max(float(min_leverage), min(max_lev_f, float(calc_lev)))

        # 심볼 step에 맞춰 정규화 (거래소 제약)
        step = float(leverage_step) if leverage_step and float(leverage_step) > 0 else 0.1
        lev = RiskManager._snap_to_step(lev, step)

        # 정규화 후 경계 재보정
        if lev > max_lev_f:
            lev = RiskManager._snap_to_step(max_lev_f, step)
        if lev < float(min_leverage):
            lev = RiskManager._snap_to_step(float(min_leverage), step)

        needed_m = max(0.0, float(target_m) - float(current_m or 0.0))
        qty = (float(needed_m) * float(lev)) / float(price)

        return float(lev), float(qty), float(needed_m)



# -----------------------------------------------------------
# [모듈 6] 봇 컨트롤러 - 통합 매매 루프 및 단계별 알림
# -----------------------------------------------------------
class AsyncTradingBot:
    def __init__(self):
        self.exchange = None
        self.data_manager = None
        self.telegram = None
        self.db_manager = None
        self.is_running = True
        self.last_status_time = datetime.min
        self.target_symbols = []
        self.last_rsi_alert_time = {}
        self.found_strong_symbols = {}
        self.notification_queue = None
        self.tp_status = {}
        self.initial_sl_map = {}  # key=(symbol,pos_idx) -> float

        self.api_sem = None

        self.exit_locks = {}  # (symbol,pos_idx) -> asyncio.Lock
        self.exit_inflight = set()
        self.last_exit_time = {}

        self.candidate_lock = asyncio.Lock()
        self.last_candidate_refresh = 0.0
        self.exit_sem = None

        self.universe_symbols = []
        self.candidate_symbols = []

        self.entry_strategy_map = {}  # key=(symbol,pos_idx) -> strategy_note
        self.bb_armed_map = {}  # key=(symbol,pos_idx) -> bool

        # [NEW] Position meta for time-stop / trailing-stop arming
        # key=(symbol,pos_idx) -> dict(entry_ts, entry_price, best_price, best_profit_pct)
        self.pos_meta_map = {}

        # [NEW] Trailing Stop state
        self.ts_armed_map = {}  # key=(symbol,pos_idx) -> bool
        self.ts_dist_map = {}   # key=(symbol,pos_idx) -> float(last applied)
        self.group_id_map = {}  # key=(symbol,pos_idx) -> group_id
        self.event_cli = None

        # [추가] 심볼 단위 진입 락(동일 심볼 중복 진입 방지)
        self.entry_locks = {}  # symbol -> asyncio.Lock

        # [PATCH] 활성도 점수 캐시 (Universe=200 확정용)
        self.activity_scores = {}       # symbol -> float
        self.activity_scores_ts = 0.0  # last update time



    def _calc_supertrend_dir(self, df, period=10, multiplier=3.0):
        """
        df 컬럼: h, l, c (float)
        반환: (st_line: np.ndarray, st_dir: np.ndarray)
        - st_dir: +1(상승), -1(하락)
        """
        try:
            import numpy as np

            if df is None or len(df) < (period + 5):
                return None, None

            high = df["h"].astype(float).to_numpy()
            low = df["l"].astype(float).to_numpy()
            close = df["c"].astype(float).to_numpy()

            n = len(close)
            hl2 = (high + low) / 2.0

            # TR/ATR
            tr = np.zeros(n, dtype=float)
            tr[0] = high[0] - low[0]
            prev_close = close[:-1]
            tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))

            atr = np.full(n, np.nan, dtype=float)
            # 초기 ATR: 단순 평균
            if n >= period:
                atr[period - 1] = np.nanmean(tr[:period])
                # 이후 ATR: RMA(= Wilder smoothing)
                for i in range(period, n):
                    atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

            basic_ub = hl2 + multiplier * atr
            basic_lb = hl2 - multiplier * atr

            final_ub = np.full(n, np.nan, dtype=float)
            final_lb = np.full(n, np.nan, dtype=float)

            # 초기화
            for i in range(n):
                if not np.isnan(basic_ub[i]) and not np.isnan(basic_lb[i]):
                    final_ub[i] = basic_ub[i]
                    final_lb[i] = basic_lb[i]
                    start = i
                    break
            else:
                return None, None

            # 밴드 보정
            for i in range(start + 1, n):
                # upper band
                if (basic_ub[i] < final_ub[i - 1]) or (close[i - 1] > final_ub[i - 1]):
                    final_ub[i] = basic_ub[i]
                else:
                    final_ub[i] = final_ub[i - 1]

                # lower band
                if (basic_lb[i] > final_lb[i - 1]) or (close[i - 1] < final_lb[i - 1]):
                    final_lb[i] = basic_lb[i]
                else:
                    final_lb[i] = final_lb[i - 1]

            st = np.full(n, np.nan, dtype=float)
            st_dir = np.full(n, 0.0, dtype=float)

            # supertrend 초기값
            st[start] = final_lb[start] if close[start] >= final_lb[start] else final_ub[start]
            st_dir[start] = 1.0 if st[start] == final_lb[start] else -1.0

            for i in range(start + 1, n):
                prev_st = st[i - 1]

                if prev_st == final_ub[i - 1]:
                    # 하락 추세 라인이던 상태
                    st[i] = final_lb[i] if close[i] > final_ub[i] else final_ub[i]
                else:
                    # 상승 추세 라인이던 상태
                    st[i] = final_ub[i] if close[i] < final_lb[i] else final_lb[i]

                st_dir[i] = 1.0 if st[i] == final_lb[i] else -1.0

            # 정수 형태로 리턴(+1/-1)
            st_dir = np.where(st_dir >= 0, 1, -1).astype(int)
            return st, st_dir

        except Exception:
            return None, None


    async def maybe_stflip_topup(self, symbol, pos, total_bal, curr_price):
        """
        [별도 추매 모듈]
        - 15m ST: 상승 유지(이전봉 기준)
        - 5m ST: Flip Up 이벤트(이전봉에서 Down->Up 전환 완성)
        - 현재 증거금 비중 < 15% 일 때만 발동
        - 목표 증거금 비중: 20%까지 채우기
        - 동일 5m flip(같은 캔들 ts)에서는 1회만 실행
        - 기존 피라미딩/추매 로직과 상태/예산 분리
        """
        try:
            if total_bal is None or float(total_bal) <= 0:
                return
            if curr_price is None or float(curr_price) <= 0:
                return
            if not pos:
                return

            # 롱만 (현재 봇이 롱 위주면 이게 안전)
            if self._get_pos_side(pos) != "long":
                return

            pos_idx = self._extract_position_idx(pos, default=0)


            # --- 별도 상태 저장소 (없으면 생성) ---
            if not hasattr(self, "stflip_topup_last_ts"):
                self.stflip_topup_last_ts = {}  # key=(symbol,pos_idx) -> last_5m_flip_candle_ts

            # --- 현재 증거금 비중 체크(15% 미만만) ---
            cur_m = float(self._estimate_position_margin(pos, curr_price=float(curr_price)) or 0.0)
            gate_m = float(total_bal) * 0.15
            target_m = float(total_bal) * 0.20

            # 이미 15% 이상이면 "잦은 추매 방지" 규칙에 의해 그냥 종료
            if cur_m >= gate_m:
                return

            # 이미 20% 이상이면 더 살 이유 없음
            if cur_m >= target_m:
                return

            # --- 15m/5m OHLCV fetch (exit_sem 사용) ---
            o15 = await self._api_call(self.exchange.fetch_ohlcv, symbol, "15m", None, 120, sem=self.exit_sem, tag=f"st15:{symbol}")
            o5  = await self._api_call(self.exchange.fetch_ohlcv, symbol, "5m",  None, 200, sem=self.exit_sem, tag=f"st5:{symbol}")
            if not o15 or not o5:
                return

            import pandas as pd

            df15 = pd.DataFrame(o15, columns=["ts", "o", "h", "l", "c", "v"])
            df5  = pd.DataFrame(o5,  columns=["ts", "o", "h", "l", "c", "v"])
            df15[["h", "l", "c"]] = df15[["h", "l", "c"]].apply(pd.to_numeric, errors="coerce")
            df5[["h", "l", "c"]]  = df5[["h", "l", "c"]].apply(pd.to_numeric, errors="coerce")

            # --- SuperTrend 계산 (필요시 period/multiplier 여기서 조정) ---
            st15, dir15 = self._calc_supertrend_dir(df15, period=10, multiplier=3.0)
            st5,  dir5  = self._calc_supertrend_dir(df5,  period=10, multiplier=3.0)
            if st15 is None or dir15 is None or st5 is None or dir5 is None:
                return
            if len(dir15) < 3 or len(dir5) < 4:
                return

            # "이전봉에서 완성" 규칙:
            # - fetch_ohlcv의 마지막은 진행중일 가능성이 커서
            # - 확정봉은 [-2], 그 직전 확정봉은 [-3]로 판단
            dir15_prev = int(dir15[-2])  # 15m 확정봉 방향
            dir5_prev2 = int(dir5[-3])   # 5m 확정봉(이전이전)
            dir5_prev  = int(dir5[-2])   # 5m 확정봉(이전)

            # 15m 상승 유지 필터
            if dir15_prev != 1:
                return

            # 5m Flip Up 이벤트(Down -> Up)
            flip_up = (dir5_prev2 == -1) and (dir5_prev == 1)
            if not flip_up:
                return

            # flip 캔들 ts(이전 확정봉 ts)를 키로 해서 "동일 이벤트 1회" 방지
            flip_ts = int(df5["ts"].iloc[-2])
            key = (symbol, int(pos_idx))
            if self.stflip_topup_last_ts.get(key) == flip_ts:
                return

            # --- 목표 20%까지 채우기 위한 필요 증거금 계산 ---
            needed_m = float(target_m - cur_m)
            if needed_m <= 0:
                return

            # 현재 레버리지로 수량 환산(레버리지/SL 재계산 등 기존 로직과 분리)
            lev = float(self._get_pos_num(pos, "leverage", default=1.0) or 1.0)
            if lev <= 0:
                lev = 1.0

            qty = (needed_m * lev) / float(curr_price)

            # 수량 정밀도/최소수량 처리
            qty = float(self.exchange.amount_to_precision(symbol, qty))
            min_q = float(self.get_symbol_min_amount(symbol) or 0.0)

            if qty <= 0:
                return
            if min_q > 0 and qty < min_q:
                # 여기서 억지로 minQty로 올리면 20% 초과될 수 있어,
                # "잦은 추매 방지" 관점에서 그냥 스킵이 더 안전함
                self.queue_notify(f"[ST_FLIP_TOPUP_SKIP] {symbol} qty<{min_q} (qty={qty}, need_m={needed_m:.4f})")
                self.stflip_topup_last_ts[key] = flip_ts
                return

            # 심볼 단위 진입 락으로 중복 주문 방지
            lock = self._get_entry_lock(symbol)
            async with lock:
                # 다시 한번 현재 증거금 확인(레이스 방지)
                pos2 = await self.get_safe_position(symbol)
                if not pos2:
                    return
                cur_m2 = float(self._estimate_position_margin(pos2, curr_price=float(curr_price)) or 0.0)
                if cur_m2 >= gate_m or cur_m2 >= target_m:
                    self.stflip_topup_last_ts[key] = flip_ts
                    return

                # 주문 실행(기존 추매 로직과 분리: note로 식별)
                self.queue_notify(
                    f"[ST_FLIP_TOPUP] {symbol} 5m FlipUp + 15m Up 유지 -> 추매\n"
                    f"- curr_m:{cur_m2:.4f} (<15%) -> target_m:{target_m:.4f}(20%)\n"
                    f"- add_qty:{qty} @ {float(curr_price):.6f} (lev={lev})"
                )

                await self.execute_aggressive_order(
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    note="Add_STFlip_5m",
                    total_bal=float(total_bal),
                    sl_p=None,   # SL/레버리지 재계산에 섞이지 않게(완전 분리)
                    depth=None
                )

                # 동일 flip 이벤트 1회 처리 완료 마킹
                self.stflip_topup_last_ts[key] = flip_ts

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[ST_FLIP_TOPUP_ERR] {symbol}: {e}", include_traceback=True)









    def _market_safe(self, symbol: str) -> dict:
        try:
            return self.exchange.market(symbol) or {}
        except Exception:
            return {}

    async def get_symbol_max_leverage(self, symbol: str) -> int:
        """
        [필수] max leverage가 없으면 주문/리스크 계산에서 터지거나 qty=0이 됩니다.
        Bybit/CCXT 시장정보에서 최대 레버리지 최대한 안전하게 추출.
        """
        try:
            m = self._market_safe(symbol)

            # 1) CCXT 표준 limits.leverage.max
            try:
                v = (((m.get("limits") or {}).get("leverage") or {}).get("max"))
                if v:
                    v = int(float(v))
                    if v > 0:
                        return v
            except Exception:
                pass

            info = m.get("info") or {}

            # 2) Bybit V5: leverageFilter.maxLeverage
            try:
                lf = info.get("leverageFilter") or info.get("leverage_filter") or {}
                v = lf.get("maxLeverage") or lf.get("max_leverage") or lf.get("max")
                if v:
                    v = int(float(v))
                    if v > 0:
                        return v
            except Exception:
                pass

            # 3) 다른 케이스: maxLeverage 키가 직접 존재
            for k in ("maxLeverage", "max_leverage", "leverageMax", "maxLev"):
                try:
                    v = info.get(k)
                    if v:
                        v = int(float(v))
                        if v > 0:
                            return v
                except Exception:
                    continue

        except Exception:
            pass

        # fallback (보수적으로 30)
        return 30


    async def get_symbol_leverage_step(self, symbol: str) -> float:
        """
        심볼별 레버리지 스텝(0.5/0.1 등) 추출.
        - Bybit V5에서 leverageFilter.leverageStep가 가장 흔함
        - 못 찾으면 0.1 fallback
        """
        try:
            m = self._market_safe(symbol)
            info = (m.get("info") or {})

            lf = info.get("leverageFilter") or info.get("leverage_filter") or {}
            for k in ("leverageStep", "leverage_step", "step", "minLeverageStep"):
                v = lf.get(k)
                if v is not None:
                    try:
                        step = float(v)
                        if step > 0:
                            return step
                    except Exception:
                        pass

            for k in ("leverageStep", "leverage_step", "leverage_step_size", "levStep"):
                v = info.get(k)
                if v is not None:
                    try:
                        step = float(v)
                        if step > 0:
                            return step
                    except Exception:
                        pass

        except Exception:
            pass

        return 0.1





    def get_symbol_min_amount(self, symbol: str) -> float:
        """
        최소 주문수량. qty가 여기보다 작으면 amount_to_precision에서 0으로 떨어져
        '주문 스킵'이 반복될 수 있음.
        """
        m = self._market_safe(symbol)

        # CCXT 표준
        try:
            v = (((m.get("limits") or {}).get("amount") or {}).get("min"))
            if v is not None:
                v = float(v)
                if v > 0:
                    return v
        except Exception:
            pass

        # Bybit info (lotSizeFilter.minOrderQty 등)
        info = m.get("info") or {}
        for path in (
            ("lotSizeFilter", "minOrderQty"),
            ("lot_size_filter", "min_order_qty"),
        ):
            try:
                d = info.get(path[0]) or {}
                v = d.get(path[1])
                if v is not None:
                    v = float(v)
                    if v > 0:
                        return v
            except Exception:
                continue

        return 0.0

    def _sanitize_long_sl(self, entry_price: float, sl_price: float, *, min_gap_pct: float = 0.002) -> float:
        """
        롱 포지션 기준:
        - SL이 entry 위/동일이면 거래소가 거부할 수 있어서 강제로 entry 아래로 내림
        """
        try:
            ep = float(entry_price)
            sp = float(sl_price)
            if ep <= 0 or sp <= 0:
                return sp
            # 최소 0.2% 아래
            max_ok = ep * (1.0 - float(min_gap_pct))
            if sp >= max_ok:
                return max_ok
            return sp
        except Exception:
            return sl_price












    def _sanitize_short_sl(self, entry_price: float, sl_price: float, *, min_gap_pct: float = 0.002) -> float:
        """
        숏 포지션 기준:
        - SL이 entry 아래/동일이면 거래소가 거부할 수 있어서 강제로 entry 위로 올림
        """
        try:
            ep = float(entry_price)
            sp = float(sl_price)
            if ep <= 0 or sp <= 0:
                return sp
            # 최소 0.2% 위
            min_ok = ep * (1.0 + float(min_gap_pct))
            if sp <= min_ok:
                return min_ok
            return sp
        except Exception:
            return sl_price

    def _get_entry_lock(self, symbol: str) -> asyncio.Lock:
        lk = self.entry_locks.get(symbol)
        if lk is None:
            lk = asyncio.Lock()
            self.entry_locks[symbol] = lk
        return lk

    def _extract_position_idx(self, pos, default=0) -> int:
        """Bybit hedge/one-way 모두 안전하게 positionIdx 추출"""
        try:
            if not pos:
                return default
            if pos.get("positionIdx") is not None:
                return int(pos.get("positionIdx"))
            info = pos.get("info") or {}
            if isinstance(info, dict) and info.get("positionIdx") is not None:
                return int(info.get("positionIdx"))
        except Exception:
            pass
        return default

    def _get_pos_num(self, pos, *keys, default=0.0) -> float:
        """pos / pos['info'] 어디에 있든 숫자값 안전하게 꺼내기"""
        try:
            if not pos:
                return float(default)

            for k in keys:
                v = pos.get(k)
                if v not in (None, "", "0"):
                    try:
                        return float(v)
                    except Exception:
                        pass

            info = pos.get("info") or {}
            if isinstance(info, dict):
                for k in keys:
                    v = info.get(k)
                    if v not in (None, "", "0"):
                        try:
                            return float(v)
                        except Exception:
                            pass
        except Exception:
            pass
        return float(default)

    def _get_exit_lock(self, key):
        """key: (symbol, pos_idx)"""
        lock = self.exit_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.exit_locks[key] = lock
        return lock

    def _purge_symbol_tuple_keys(self, d: dict, symbol: str):
        """dict 키가 (symbol, pos_idx) 형태일 때 해당 symbol 키 전부 제거"""
        try:
            for k in list(d.keys()):
                if isinstance(k, tuple) and len(k) >= 1 and k[0] == symbol:
                    d.pop(k, None)
        except Exception:
            pass

    def _make_group_id(self, symbol: str, side: str) -> str:
        direction = "LONG" if side == "buy" else "SHORT"
        return f"{symbol.replace('/','').replace(':','')}-{int(time.time()*1000)}-{direction}"

    def _get_group_id(self, symbol: str, pos_idx: int) -> str:
        return self.group_id_map.get((symbol, int(pos_idx)), "")

    def _purge_position_state(self, lock_key):
        """
        lock_key: (symbol, pos_idx)
        전량 청산 이후 상태 오염 방지용 purge
        """
        try:
            self.tp_status.pop(lock_key, None)
            self.initial_sl_map.pop(lock_key, None)
            self.entry_strategy_map.pop(lock_key, None)
            self.bb_armed_map.pop(lock_key, None)
            self.group_id_map.pop(lock_key, None)
        except Exception:
            pass

    def _extract_order_id(self, order) -> str:
        try:
            if not isinstance(order, dict):
                return ""

            if order.get("id"):
                return str(order["id"])

            info = order.get("info") or {}
            if isinstance(info, dict):
                # 1) 바로 밑 키들
                for k in ("orderId", "orderID", "order_id", "orderLinkId", "order_link_id", "orderLinkID"):
                    v = info.get(k)
                    if v:
                        return str(v)

                # 2) result 내부
                r = info.get("result")
                if isinstance(r, dict):
                    for k in ("orderId", "orderID", "order_id", "orderLinkId", "order_link_id", "orderLinkID"):
                        v = r.get(k)
                        if v:
                            return str(v)
        except Exception:
            pass

        return ""


    # -------------------------------
    # [PATCH] Universe 활성도 점수 계산 유틸
    # -------------------------------
    def _calc_quote_volume_from_ticker(self, ticker: dict) -> float:
        if not ticker:
            return 0.0
        try:
            qv = ticker.get("quoteVolume", None)
            if qv is not None:
                qv = float(qv)
                return qv if qv > 0 else 0.0
        except Exception:
            pass

        try:
            bv = float(ticker.get("baseVolume") or 0.0)
            last = float(ticker.get("last") or 0.0)
            v = bv * last
            return v if v > 0 else 0.0
        except Exception:
            return 0.0

    def _turnover_spike_from_ohlcv(self, ohlcv_rows):
        """
        ohlcv_rows: [[ts, o, h, l, c, v], ...]
        반환: (spike, level)
        - spike: (최근 '완료' 봉 turnover) / (직전 20개 완료봉 turnover 평균)
        - level: 최근 완료봉 turnover(절대치)
        """
        try:
            if not ohlcv_rows or len(ohlcv_rows) < 25:
                return 0.0, 0.0

            # 최근 완료봉 = -2 (마지막 봉은 진행중일 수 있음)
            r_now = ohlcv_rows[-2]
            c_now = float(r_now[4] or 0.0)
            v_now = float(r_now[5] or 0.0)
            now = c_now * v_now
            if now < 0:
                now = 0.0

            # 직전 20개 완료봉: (-22) ~ (-3) = 20개
            s = 0.0
            n = 0
            for r in ohlcv_rows[-22:-2]:
                c = float(r[4] or 0.0)
                v = float(r[5] or 0.0)
                tv = c * v
                if tv > 0:
                    s += tv
                    n += 1

            if n <= 0:
                return 0.0, float(now)

            ma = s / float(n)
            if ma <= 0:
                return 0.0, float(now)

            spike = now / (ma + 1e-10)

            # 과도한 꼬리 방지
            if spike > 20.0:
                spike = 20.0
            if spike < 0.0:
                spike = 0.0

            return float(spike), float(now)

        except Exception:
            return 0.0, 0.0



    async def _fetch_ohlcv_safe(self, symbol: str, tf: str, limit: int):
        import time
        import asyncio

        def _tf_to_sec(x: str) -> int:
            x = (x or "").strip().lower()
            if x.endswith("m"):
                return int(x[:-1]) * 60
            if x.endswith("h"):
                return int(x[:-1]) * 3600
            if x.endswith("d"):
                return int(x[:-1]) * 86400
            return 0

        try:
            limit = int(limit or 60)
            if limit < 30:
                limit = 30

            if not hasattr(self, "_univ_ohlcv_cache"):
                self._univ_ohlcv_cache = {}

            # 유니버스 전용 세마포어 (없으면 생성)
            if not hasattr(self, "universe_api_sem") or self.universe_api_sem is None:
                # 유니버스는 낮게(예: 4~8) 잡는게 안전
                n = int(getattr(Config, "UNIVERSE_API_CONCURRENCY", 6))
                if n <= 0:
                    n = 6
                self.universe_api_sem = asyncio.Semaphore(n)

            timeout_sec = float(getattr(Config, "UNIVERSE_OHLCV_TIMEOUT_SEC", 3.5))
            cache_max = int(getattr(Config, "UNIVERSE_OHLCV_CACHE_MAX", 5000))

            tf_sec = _tf_to_sec(tf)
            now = time.time()

            bucket = int(now // tf_sec) if tf_sec > 0 else int(now // 60)

            key = (str(symbol), str(tf), int(limit), int(bucket))
            hit = self._univ_ohlcv_cache.get(key)
            if hit:
                return hit

            coro = self._api_call(
                self.exchange.fetch_ohlcv,
                symbol,
                tf,
                None,
                limit,
                tag=f"universe_ohlcv:{symbol}:{tf}",
                # 여기만 유니버스 전용 sem으로!
                sem=self.universe_api_sem,
            )
            data = await asyncio.wait_for(coro, timeout=timeout_sec)

            if data:
                self._univ_ohlcv_cache[key] = data
                if len(self._univ_ohlcv_cache) > cache_max:
                    for k in list(self._univ_ohlcv_cache.keys())[: cache_max // 2]:
                        self._univ_ohlcv_cache.pop(k, None)

            return data

        except Exception:
            return None




    async def _compute_activity_score_one(self, symbol: str, ticker: dict):
        """
        활성도 점수:
        - 1h/30m/4h turnover spike(완료봉 기준) + 유동성(quoteVolume) 가중
        """
        try:
            qv = self._calc_quote_volume_from_ticker(ticker)
            if qv <= 0:
                return symbol, 0.0

            tfs = list(getattr(Config, "UNIVERSE_ACTIVITY_TFS", ["30m", "1h"]))
            limit = int(getattr(Config, "UNIVERSE_OHLCV_LIMIT", 60))

            tasks = [self._fetch_ohlcv_safe(symbol, tf, limit) for tf in tfs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            spike_1h = level_1h = 0.0
            spike_30m = level_30m = 0.0
            # spike_4h = level_4h = 0.0

            for tf, res in zip(tfs, results):
                if isinstance(res, Exception) or (not res):
                    continue
                sp, lv = self._turnover_spike_from_ohlcv(res)
                if tf == "1h":
                    spike_1h, level_1h = sp, lv
                elif tf == "30m":
                    spike_30m, level_30m = sp, lv
                # elif tf == "4h":
                #     spike_4h, level_4h = sp, lv

            w1h = float(getattr(Config, "UNIVERSE_WEIGHT_1H", 0.45))
            w30 = float(getattr(Config, "UNIVERSE_WEIGHT_30M", 0.35))
            # w4h = float(getattr(Config, "UNIVERSE_WEIGHT_4H", 0.20))

            base = (np.log1p(spike_1h) * w1h) + (np.log1p(spike_30m) * w30) #+ (np.log1p(spike_4h) * w4h)

            # 유동성 가중(quoteVolume 로그): 저유동/노이즈 종목 과대선정 방지
            liq = np.log10(qv + 1.0)

            score = float(base * liq)

            if not np.isfinite(score) or score < 0:
                score = 0.0

            return symbol, score

        except Exception:
            return symbol, 0.0
        

    async def _build_universe_by_activity(self, all_symbols, tickers: dict):
        import asyncio
        import math
        import time

        try:
            if not all_symbols:
                return [], {}

            # [PATCH] Config 기반 버킷 캐시 (기본 5분)
            cache_sec = int(getattr(Config, "UNIVERSE_CACHE_SEC", 1800))
            cache_sec = max(60, cache_sec)
            bucket = int(time.time() // cache_sec)

            last_bucket = getattr(self, "_universe_final_bucket", None)
            last_cache = getattr(self, "_universe_final_cache", None)
            if last_bucket == bucket and last_cache:
                return last_cache  # (universe, universe_scores)


            # ---- 이하 기존 로직 그대로 ----
            top_n = int(getattr(Config, "TOP_COIN_LIMIT", 300))
            pool_n = int(getattr(Config, "UNIVERSE_POOL_N", 300))
            if pool_n < top_n:
                pool_n = top_n

            ranked_qv = []
            qv_map = {}
            for s in all_symbols:
                t = (tickers or {}).get(s) or {}
                qv = float(self._calc_quote_volume_from_ticker(t) or 0.0)
                ranked_qv.append((s, qv))
                qv_map[str(s)] = qv

            ranked_qv.sort(key=lambda x: x[1], reverse=True)
            pool_n = min(pool_n, len(ranked_qv))
            pool_syms = [sym for sym, _qv in ranked_qv[:pool_n] if sym]
            if not pool_syms:
                return [], {}

            tfs = list(getattr(Config, "UNIVERSE_ACTIVITY_TFS", ["30m", "1h"]))
            limit = int(getattr(Config, "UNIVERSE_OHLCV_LIMIT", 30))
            if limit < 30:
                limit = 30

            w1h = float(getattr(Config, "UNIVERSE_WEIGHT_1H", 0.45))
            w30 = float(getattr(Config, "UNIVERSE_WEIGHT_30M", 0.35))

            workers_n = int(getattr(Config, "UNIVERSE_ACTIVITY_CONCURRENCY", 18))
            if workers_n <= 0:
                workers_n = 18

            spike_by_tf = {tf: {} for tf in tfs}
            q = asyncio.Queue()

            for sym in pool_syms:
                for tf in tfs:
                    q.put_nowait((sym, tf))

            async def _worker():
                while True:
                    item = await q.get()
                    if item is None:
                        q.task_done()
                        break
                    sym, tf = item
                    try:
                        rows = await self._fetch_ohlcv_safe(sym, tf, limit)
                        if rows:
                            sp, _lv = self._turnover_spike_from_ohlcv(rows)
                            spike_by_tf.get(tf, {})[str(sym)] = float(sp or 0.0)
                    except Exception:
                        pass
                    finally:
                        q.task_done()

            workers = [asyncio.create_task(_worker()) for _ in range(workers_n)]
            await q.join()
            for _ in workers:
                q.put_nowait(None)
            await asyncio.gather(*workers, return_exceptions=True)

            score_map = {}
            for sym in pool_syms:
                sym_s = str(sym)
                qv = float(qv_map.get(sym_s, 0.0))
                if qv <= 0:
                    score_map[sym_s] = 0.0
                    continue

                sp1h = float(spike_by_tf.get("1h", {}).get(sym_s, 0.0))
                sp30 = float(spike_by_tf.get("30m", {}).get(sym_s, 0.0))

                base = (math.log1p(sp1h) * w1h) + (math.log1p(sp30) * w30)
                liq = math.log10(qv + 1.0)

                score = float(base * liq)
                if (not math.isfinite(score)) or score < 0:
                    score = 0.0
                score_map[sym_s] = score

            def _score_key(sym):
                sym_s = str(sym)
                return (float(score_map.get(sym_s, 0.0)), float(qv_map.get(sym_s, 0.0)))

            pool_syms_sorted = sorted(pool_syms, key=_score_key, reverse=True)
            universe = pool_syms_sorted[:top_n]
            universe_scores = {str(s): float(score_map.get(str(s), 0.0)) for s in universe}

            # [PATCH] 최종 결과 캐시 저장(Config 버킷)
            self._universe_final_bucket = bucket
            self._universe_final_cache = (universe, universe_scores)


            return universe, universe_scores

        except Exception:
            return [], {}


        

    def _estimate_position_margin(self, pos, curr_price=None) -> float:
        """
        [PATCH] 보유 포지션의 '증거금(current_m)' 추정(추가매수 판단용)
        우선순위:
        1) pos/pos.info 내 initialMargin/positionIM 류
        2) notional(contracts*price) / leverage 로 근사
        """
        try:
            if not pos:
                return 0.0

            m = abs(self._get_pos_num(
                pos,
                "initialMargin", "positionIM", "positionIm", "posIM", "margin", "positionMargin",
                default=0.0
            ))
            if m > 0:
                return float(m)

            contracts = abs(self._get_pos_num(pos, "contracts", "size", default=0.0))
            if contracts <= 0:
                return 0.0

            px = 0.0
            if curr_price is not None:
                try:
                    px = float(curr_price)
                except Exception:
                    px = 0.0

            if px <= 0:
                px = float(self._get_pos_num(pos, "markPrice", "mark_price", "entryPrice", "avgPrice", "averagePrice", default=0.0))

            if px <= 0:
                return 0.0

            lev = float(self._get_pos_num(pos, "leverage", default=1.0))
            if lev <= 0:
                lev = 1.0

            notional = contracts * px
            m2 = notional / lev
            return float(m2 if m2 > 0 else 0.0)
        except Exception:
            return 0.0



    async def execute_exit_order(self, symbol, side, qty, note, total_bal, pos_idx=None):
        """
        Exit 전용: 시장가(reduceOnly)로 즉시 청산.
        - hedge 모드 대응: positionIdx 지정 가능
        - 모든 청산 주문은 exit_sem으로만 호출
        """
        try:
            qty = abs(float(qty))
            qty = float(self.exchange.amount_to_precision(symbol, qty))
            if qty <= 0:
                return

            params = {"reduceOnly": True}
            if pos_idx is not None:
                params["positionIdx"] = int(pos_idx)

            exit_order = await self._api_call(
                self.exchange.create_market_order,
                symbol,
                side,
                qty,
                params=params,
                tag=f"EXIT_market:{symbol}:{note}",
                sem=self.exit_sem,
            )

            exit_exchange_order_id = self._extract_order_id(exit_order)

            try:
                t = await self._api_call(
                    self.exchange.fetch_ticker,
                    symbol,
                    tag=f"exit_ticker:{symbol}",
                    sem=self.exit_sem
                )
                px = float(t.get("last") or 0.0)
            except Exception:
                px = 0.0

            try:
                pidx_for_event = int(pos_idx) if (pos_idx is not None) else 0
                gid = self._get_group_id(symbol, pidx_for_event)

                if not gid:
                    gid = self._make_group_id(symbol, "buy" if side == "sell" else "sell")
                    self.group_id_map[(symbol, pidx_for_event)] = gid

                await asyncio.sleep(0.2)

                pos_after_exit = await self.get_safe_position(symbol, pos_idx=pidx_for_event)
                remain = 0.0
                if pos_after_exit:
                    remain = abs(self._get_pos_num(pos_after_exit, "contracts", "size", default=0.0))

                event_type = "CLOSE_FULL" if remain <= 0 else "CLOSE_PART"
                pos_dir = "LONG" if side == "sell" else "SHORT"

                entry_note = (
                    self.entry_strategy_map.get((symbol, pidx_for_event))
                    or self.entry_strategy_map.get((symbol, 0))
                    or ""
                )
                if not entry_note:
                    entry_note = "UNKNOWN_ENTRY"

                if self.event_cli and gid:
                    await self.event_cli.send_trade_event(
                        {
                            "symbol": symbol,
                            "event_type": event_type,
                            "side": pos_dir,
                            "group_id": gid,
                            "order_link_id": gid,
                            "qty": float(qty),
                            "price": float(px),
                            "strategy_note": str(entry_note),
                            "extra_json": {
                                "exit_note": str(note),
                                "remain_contracts": float(remain),
                                "exchange_order_id": exit_exchange_order_id,
                            },
                        }
                    )

                if event_type == "CLOSE_FULL":
                    self.group_id_map.pop((symbol, pidx_for_event), None)

            except Exception:
                pass

            self.queue_notify(f"[EXIT_DONE] {note} {side.upper()} {symbol} qty={qty}")

        except Exception as e:
            self.queue_notify(f"[EXIT_ERROR] {symbol} {note} 실패: {e}")


    async def get_all_open_positions(self, sem=None):
        """
        Bybit linear 전체 포지션 중 '보유중인 것(contracts/size!=0)'만 반환.
        hedge 모드 포함: symbol 같아도 positionIdx로 구분됨.
        """
        use_sem = sem if sem is not None else self.exit_sem
        try:
            positions = await self._api_call(
                self.exchange.fetch_positions,
                None,
                params={"category": "linear"},
                tag="fetch_positions_all",
                sem=use_sem,
            )

            if not positions:
                return []

            open_pos = []
            for p in positions:
                try:
                    c = abs(self._get_pos_num(p, "contracts", "size", default=0.0))
                    if c > 0:
                        sym = p.get("symbol")
                        if sym:
                            open_pos.append(p)
                except Exception:
                    continue

            return open_pos

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[POS_ALL_ERR] {e}")
            return []

    def _get_pos_side(self, pos) -> str:
        """
        반환: "long" | "short" | "unknown"
        - CCXT/Bybit 포지션 구조 차이를 최대한 흡수
        """
        try:
            if not pos:
                return "unknown"
            side = pos.get("side")
            if side:
                s = str(side).lower()
                if "long" in s or s == "buy":
                    return "long"
                if "short" in s or s == "sell":
                    return "short"

            info = pos.get("info") or {}
            if isinstance(info, dict):
                side2 = info.get("side")
                if side2:
                    s = str(side2).lower()
                    if "long" in s or s == "buy":
                        return "long"
                    if "short" in s or s == "sell":
                        return "short"

                pidx = info.get("positionIdx")
                if pidx is not None:
                    pidx = int(pidx)
                    if pidx == 1:
                        return "long"
                    if pidx == 2:
                        return "short"
        except Exception:
            pass
        return "unknown"



    def _close_side_for_pos(self, pos) -> str:
        """
        포지션 청산 주문 side 반환: "sell"(롱 청산) / "buy"(숏 청산)
        """
        ps = self._get_pos_side(pos)
        if ps == "short":
            return "buy"
        return "sell"



    async def _api_call(self, fn, *args, tag="", sem=None, **kwargs):
        """
        동시성 제한 + 재시도 + 백오프
        - 재시도 의미 없는 에러(BadSymbol/InsufficientFunds)는 즉시 중단
        """
        use_sem = sem if sem is not None else self.api_sem
        if use_sem is None:
            return await fn(*args, **kwargs)

        last_err = None

        for attempt in range(1, Config.API_MAX_RETRIES + 1):
            try:
                async with use_sem:
                    return await fn(*args, **kwargs)

            except (BadSymbol, InsufficientFunds) as e:
                # 재시도해도 해결 안 되는 케이스
                write_log(ERROR_LOG_FILE, f"[API_FATAL] {tag} err={e}")
                raise

            except (RateLimitExceeded, NetworkError, ExchangeNotAvailable, RequestTimeout) as e:
                last_err = e
                backoff = min(Config.API_BACKOFF_MAX, Config.API_BACKOFF_BASE * (2 ** (attempt - 1)))
                backoff = backoff * (1.0 + random.random() * 0.2)
                write_log(ERROR_LOG_FILE, f"[API_RETRY] {tag} attempt={attempt}/{Config.API_MAX_RETRIES} err={e} sleep={backoff:.2f}s")
                await asyncio.sleep(backoff)
                continue

            except Exception as e:
                # 기타 에러는 즉시 노출(숨기면 디버깅 불가능)
                write_log(ERROR_LOG_FILE, f"[API_ERR] {tag} err={e}", include_traceback=True)
                raise

        # 재시도 소진
        raise last_err if last_err else RuntimeError("API call failed without exception")

    async def setup(self):
        # self.db_manager = DatabaseManager(DB_FILE)
        # write_log(ERROR_LOG_FILE, f"[DB] {self.db_manager.init_msg} (path={DB_FILE})")

        self.notification_queue = asyncio.Queue()
        write_log(ERROR_LOG_FILE, "봇 엔진 시동 중...")

        Config.load_config(CONFIG_FILE)

        self.api_sem = asyncio.Semaphore(getattr(Config, "API_CONCURRENCY_SCAN", 6))
        self.exit_sem = asyncio.Semaphore(getattr(Config, "API_CONCURRENCY_EXIT", 2))

        try:
            self.event_cli = BotEventClient("http://127.0.0.1:8000")
            write_log(ERROR_LOG_FILE, "[EVENT] BotEventClient 생성 OK (127.0.0.1:8000)")
        except Exception as e:
            self.event_cli = None
            write_log(ERROR_LOG_FILE, f"[EVENT] BotEventClient 생성 실패: {e}")

        self.telegram = TelegramManager(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
        await self.telegram.start()

        self.exchange = getattr(ccxt, Config.EXCHANGE_ID)(
            {
                "apiKey": Config.API_KEY,
                "secret": Config.SECRET,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                    "adjustForTimeDifference": True,
                    "recvWindow": 60000,
                },
            }
        )

        if Config.IS_TEST_MODE:
            self.exchange.set_sandbox_mode(True)
            write_log(ERROR_LOG_FILE, "Bybit Testnet(Sandbox) 모드 활성화됨")

        self.data_manager = DataManager(self.exchange, self.telegram, self)

        try:
            await self._api_call(self.exchange.load_time_difference, tag="load_time_difference")
            markets = await self._api_call(self.exchange.load_markets, tag="load_markets")

            all_symbols = []
            for s, m in (markets or {}).items():
                try:
                    if not m:
                        continue
                    if not m.get("active"):
                        continue
                    if not m.get("swap"):
                        continue
                    if not m.get("linear"):
                        continue
                    if m.get("quote") != Config.QUOTE_CURRENCY:
                        continue
                    all_symbols.append(s)
                except Exception:
                    continue

            if not all_symbols:
                raise RuntimeError("활성 선물 심볼을 찾지 못했습니다.")

            self.all_symbols = all_symbols[:]  # [PATCH] Universe refresh용 원본 심볼 저장


            tickers = None
            try:
                tickers = await self._api_call(self.exchange.fetch_tickers, tag="fetch_tickers_all")
            except Exception:
                tickers = None

            if not tickers:
                try:
                    tickers = await self._api_call(self.exchange.fetch_tickers, all_symbols, tag="fetch_tickers_symbols")
                except Exception:
                    tickers = None

            if not tickers:
                tickers = {}
                chunk = 150
                for i in range(0, len(all_symbols), chunk):
                    part = all_symbols[i : i + chunk]
                    try:
                        t = await self._api_call(self.exchange.fetch_tickers, part, tag=f"fetch_tickers_chunk:{i}")
                        if isinstance(t, dict):
                            tickers.update(t)
                    except Exception:
                        continue

            # ---------------------------------------------------------
            # [PATCH] Universe 선정 로직 변경(흐름 유지):
            # - TOP_COIN_LIMIT = 활성도 점수로 확정된 200
            # - CANDIDATE_LIMIT = 그 중 100
            # ---------------------------------------------------------
            if not isinstance(tickers, dict):
                tickers = {}

            universe, score_map = await self._build_universe_by_activity(all_symbols, tickers)

            if not universe:
                # fallback: 기존 quoteVolume 순
                def _qv(sym: str) -> float:
                    t = (tickers or {}).get(sym) or {}
                    return float(self._calc_quote_volume_from_ticker(t) or 0.0)
                universe = sorted(all_symbols, key=_qv, reverse=True)[: Config.TOP_COIN_LIMIT]
                score_map = {s: 0.0 for s in universe}

            self.universe_symbols = universe[:]
            self.activity_scores = dict(score_map or {})
            self.activity_scores_ts = time.time()

            # Candidates 초기값: Universe 내에서 점수 우선
            cand_n = min(getattr(Config, "CANDIDATE_LIMIT", 100), len(self.universe_symbols))
            self.candidate_symbols = sorted(
                self.universe_symbols,
                key=lambda s: float(self.activity_scores.get(s, 0.0)),
                reverse=True
            )[:cand_n]

            self.target_symbols = self.universe_symbols[:]

            self.queue_notify(
                f"[시스템] Universe={len(self.universe_symbols)} / Candidates={len(self.candidate_symbols)} "
                f"(전체 활성 심볼={len(all_symbols)})"
            )

        except Exception as e:
            write_log(CRASH_LOG_FILE, f"거래소 초기화 실패: {e}")
            raise

    def _calc_quote_volume(self, ticker: dict) -> float:
        if not ticker:
            return 0.0
        try:
            qv = ticker.get("quoteVolume", None)
            if qv is not None:
                qv = float(qv)
                if qv > 0:
                    return qv
        except Exception:
            pass

        try:
            bv = float(ticker.get("baseVolume") or 0.0)
            last = float(ticker.get("last") or 0.0)
            v = bv * last
            return v if v > 0 else 0.0
        except Exception:
            return 0.0

    async def refresh_candidates_once(self):
        """
        [PATCH]
        - Universe(활성도 점수로 확정된 200)는 유지
        - Candidate(100)는 Universe 내에서
          (활성도 캐시 점수 + quoteVolume) 혼합으로 재정렬
        """
        uni = self.universe_symbols[:] if self.universe_symbols else self.target_symbols[:]
        if not uni:
            return

        tickers = None
        try:
            tickers = await self._api_call(self.exchange.fetch_tickers, uni, tag="cand_fetch_tickers_universe", sem=self.api_sem)
        except Exception:
            tickers = None

        if not isinstance(tickers, dict) or not tickers:
            return

        # 정규화(0~1) 유틸
        def _minmax_norm(vals):
            try:
                arr = [float(v or 0.0) for v in vals]
                if not arr:
                    return []
                mn = min(arr)
                mx = max(arr)
                if mx <= mn:
                    return [0.0 for _ in arr]
                return [(v - mn) / (mx - mn + 1e-10) for v in arr]
            except Exception:
                return [0.0 for _ in vals]

        qv_list = []
        sc_list = []
        for s in uni:
            t = tickers.get(s) or {}
            qv_list.append(self._calc_quote_volume(t))
            sc_list.append(float(self.activity_scores.get(s, 0.0)))

        qv_n = _minmax_norm(qv_list)
        sc_n = _minmax_norm(sc_list)

        ranked = []
        for i, s in enumerate(uni):
            # 혼합: 활성도(캐시) 0.70 + qv(최근) 0.30
            mix = (float(sc_n[i]) * 0.70) + (float(qv_n[i]) * 0.30)
            ranked.append((s, mix))

        ranked.sort(key=lambda x: x[1], reverse=True)

        n = min(getattr(Config, "CANDIDATE_LIMIT", 100), len(ranked))
        new_candidates = [sym for sym, _ in ranked[:n]]

        async with self.candidate_lock:
            self.candidate_symbols = new_candidates
            self.last_candidate_refresh = time.time()

    async def candidate_refresher_loop(self):
        await asyncio.sleep(3)
        while self.is_running:
            try:
                await self.refresh_candidates_once()
            except Exception as e:
                write_log(ERROR_LOG_FILE, f"[CAND_LOOP_ERR] {e}", include_traceback=True)
            await asyncio.sleep(getattr(Config, "CANDIDATE_REFRESH_SEC", 60))




    async def refresh_universe_once(self):
        """
        [PATCH]
        - all_symbols(활성 심볼) 기준으로 유니버스를 재선정
        - 선정 결과로 universe_symbols / activity_scores / candidate_symbols 갱신
        """
        try:
            all_symbols = getattr(self, "all_symbols", None) or []
            if not all_symbols:
                return

            # tickers 확보(가능하면 전체)
            tickers = None
            try:
                tickers = await self._api_call(self.exchange.fetch_tickers, tag="uni_fetch_tickers_all")
            except Exception:
                tickers = None

            if not tickers:
                try:
                    tickers = await self._api_call(self.exchange.fetch_tickers, all_symbols, tag="uni_fetch_tickers_symbols")
                except Exception:
                    tickers = None

            if not tickers:
                tickers = {}
                chunk = 150
                for i in range(0, len(all_symbols), chunk):
                    part = all_symbols[i : i + chunk]
                    try:
                        t = await self._api_call(self.exchange.fetch_tickers, part, tag=f"uni_fetch_tickers_chunk:{i}")
                        if isinstance(t, dict):
                            tickers.update(t)
                    except Exception:
                        pass

            if not isinstance(tickers, dict):
                tickers = {}

            universe, score_map = await self._build_universe_by_activity(all_symbols, tickers)
            if not universe:
                return

            async with self.candidate_lock:
                self.universe_symbols = universe[:]
                self.activity_scores = dict(score_map or {})
                self.activity_scores_ts = time.time()
                self.target_symbols = self.universe_symbols[:]

                # Candidates도 즉시 재산정(유니버스 내부 점수 상위)
                cand_n = min(int(getattr(Config, "CANDIDATE_LIMIT", 100)), len(self.universe_symbols))
                self.candidate_symbols = sorted(
                    self.universe_symbols,
                    key=lambda s: float(self.activity_scores.get(s, 0.0)),
                    reverse=True
                )[:cand_n]

            self.queue_notify(
                f"[Universe Refresh] Universe={len(self.universe_symbols)} / Candidates={len(self.candidate_symbols)}"
            )

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[UNIVERSE_REFRESH_ERR] {e}", include_traceback=True)

    async def universe_refresher_loop(self):
        """
        [PATCH]
        - 유니버스를 주기적으로 재선정
        """
        await asyncio.sleep(3)

        refresh_sec = int(getattr(Config, "UNIVERSE_REFRESH_SEC", 300))
        refresh_sec = max(60, refresh_sec)

        while self.is_running:
            try:
                await self.refresh_universe_once()
            except Exception as e:
                write_log(ERROR_LOG_FILE, f"[UNIVERSE_LOOP_ERR] {e}", include_traceback=True)

            await asyncio.sleep(refresh_sec)













    def queue_notify(self, message=None, trade_data=None):
        if self.notification_queue:
            self.notification_queue.put_nowait({"type": "msg" if message else "trade", "data": message or trade_data})

    async def notification_worker(self):
        while self.is_running:
            try:
                task = await self.notification_queue.get()
                if task["type"] == "msg":
                    print(task["data"], flush=True)
                    if self.telegram:
                        await self.telegram.send_message_direct(task["data"])
                elif task["type"] == "trade":
                    # trade data도 로그/텔레그램 원하면 여기서 처리
                    td = task["data"]
                    print(td, flush=True)
                self.notification_queue.task_done()
            except Exception:
                await asyncio.sleep(1)


    async def periodic_reporter(self):
        while self.is_running:
            try:
                await asyncio.sleep(300)

                strong_items = list(self.found_strong_symbols.items())
                msg = f"[5M 리포트] 강세 감시 종목 요약 ({len(strong_items)}개)\n"

                if strong_items:
                    for sym, info in strong_items[:50]:
                        msg += f"{sym}: {info}\n"
                    if len(strong_items) > 50:
                        msg += f"... (생략: {len(strong_items)-50}개)\n"
                else:
                    msg += "현재 조건 만족 종목 없음\n"

                self.queue_notify(message=msg)
                self.found_strong_symbols.clear()

            except Exception:
                pass

    async def get_safe_position(self, symbol, pos_idx=None):
        """
        심볼별 포지션 조회를 안전하게 수행.
        - pos_idx 지정 시: 해당 positionIdx만 반환(hedge 대응)
        - 실패(None)와 '없음({})'을 구분: 예외 시 None 반환
        """
        try:
            positions = await self._api_call(
                self.exchange.fetch_positions,
                [symbol],
                params={"category": "linear"},
                tag=f"fetch_positions:{symbol}",
            )

            if not positions:
                return {}

            matched = [p for p in positions if p.get("symbol") == symbol]
            if not matched:
                return {}

            if pos_idx is not None:
                pi = int(pos_idx)
                filtered = []
                for p in matched:
                    try:
                        pidx = self._extract_position_idx(p, default=0)
                        if int(pidx) == pi:
                            filtered.append(p)
                    except Exception:
                        continue
                matched = filtered
                if not matched:
                    return {}

            for p in matched:
                c = abs(self._get_pos_num(p, "contracts", "size", default=0.0))
                if c > 0:
                    return p

            return matched[0]

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[POS_ERROR] {symbol} 포지션 조회 실패: {e}", include_traceback=True)
            return None

    async def apply_exchange_v5_trading_stop(self, symbol, sl_price=None, ts_dist=None, pos_idx=0, sem=None):
        """Bybit V5 서버 측 SL/TS 등록 (exit_sem 강제 가능)"""
        try:
            use_sem = sem if sem is not None else self.exit_sem

            raw_s = symbol.split(":")[0].replace("/", "")  # BTC/USDT:USDT -> BTCUSDT
            params = {"category": "linear", "symbol": raw_s, "positionIdx": int(pos_idx)}

            if sl_price is not None and float(sl_price) > 0:
                params["stopLoss"] = str(self.exchange.price_to_precision(symbol, float(sl_price)))
                params["slTriggerBy"] = "LastPrice"

            if ts_dist is not None and float(ts_dist) > 0:
                params["trailingStop"] = str(self.exchange.price_to_precision(symbol, float(ts_dist)))

            if hasattr(self.exchange, "privatePostV5PositionTradingStop"):
                await self._api_call(
                    self.exchange.privatePostV5PositionTradingStop,
                    params,
                    tag=f"v5_trading_stop:{symbol}",
                    sem=use_sem,
                )
            elif hasattr(self.exchange, "privatePostV5PositionSetTradingStop"):
                await self._api_call(
                    self.exchange.privatePostV5PositionSetTradingStop,
                    params,
                    tag=f"v5_set_trading_stop:{symbol}",
                    sem=use_sem,
                )

            return True

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[V5_에러] {symbol} 동기화 실패 (pos_idx={pos_idx}, SL:{sl_price}, TS:{ts_dist}): {e}", include_traceback=True)
            return False




    
    async def _compute_atr_pct(self, symbol: str, curr_p: float) -> float:
        """
        TS/activation용 ATR% 계산: ATR(15m) / current_price
        """
        try:
            tf = str(getattr(Config, "TS_ATR_TF", "15m") or "15m")
            df = await self.data_manager.fetch_timeframe_data(symbol, tf, limit=180)
            if df is None or len(df) < 30:
                return 0.0

            # 지표 생성(atr)
            try:
                pack = {tf: df.copy()}
                pack = TechnicalAnalyzer.add_indicators(pack)
                df2 = pack.get(tf)
            except Exception:
                df2 = df

            if df2 is None or "atr" not in df2.columns:
                return 0.0

            atr = float(df2["atr"].iloc[-1] or 0.0)
            p = float(curr_p or 0.0)
            if atr <= 0 or p <= 0:
                return 0.0
            return max(atr / p, 0.0)
        except Exception:
            return 0.0

    async def _compute_trailing_pct(self, symbol: str, curr_p: float) -> float:
        """
        trail_pct = clamp(ATR% * TS_ATR_MULT, TS_MIN_PCT, TS_MAX_PCT)
        """
        atr_pct = await self._compute_atr_pct(symbol, curr_p)
        mult = float(getattr(Config, "TS_ATR_MULT", 2.0) or 2.0)
        min_p = float(getattr(Config, "TS_MIN_PCT", 0.02) or 0.02)
        max_p = float(getattr(Config, "TS_MAX_PCT", 0.15) or 0.15)
        pct = float(atr_pct) * mult
        pct = max(min_p, min(max_p, pct))
        return pct

    async def _compute_activation_pct(self, symbol: str, curr_p: float) -> float:
        """
        activation_pct = clamp(ATR% * TS_ACTIVATION_ATR_MULT, TS_ACTIVATION_MIN_PCT, TS_ACTIVATION_MAX_PCT)
        """
        atr_pct = await self._compute_atr_pct(symbol, curr_p)
        mult = float(getattr(Config, "TS_ACTIVATION_ATR_MULT", 0.8) or 0.8)
        min_p = float(getattr(Config, "TS_ACTIVATION_MIN_PCT", 0.01) or 0.01)
        max_p = float(getattr(Config, "TS_ACTIVATION_MAX_PCT", 0.05) or 0.05)
        pct = float(atr_pct) * mult
        pct = max(min_p, min(max_p, pct))
        return pct


    async def manage_dynamic_stop_loss(
        self,
        symbol,
        pos,
        curr_p,
        ema_p=None,
        recovery_sl=None,
        ema_trail_ok=True,
    ):
        """
        [NEW] 보유 포지션 SL/TS/TimeStop 통합 관리

        - 포지션 방향을 자동 판별 (Buy=LONG, Sell=SHORT)
        - SL:
            - SL 없으면(0) 최초 SL(맵) 또는 recovery_sl로 복구
            - 5m SuperTrend(10,2) 기반으로 유리한 방향으로만 업데이트
              LONG: ST 상방(+1) & low >= ST 일 때, SL = ST - ATR*ATR_SL_5M_K
              SHORT: ST 하방(-1) & high <= ST 일 때, SL = ST + ATR*ATR_SL_5M_K
        - TS:
            - 15m ATR%로 trail_pct 산출 (TS_MIN_PCT~TS_MAX_PCT)
            - activation_pct(ATR% 기반) 만큼 유리하게 진행된 뒤에만 서버 TS 등록
            - 등록 후엔 '갑자기 타이트'해지는 걸 막기 위해 거리 감소(타이트닝)는 하지 않음(필요 시 옵션화 가능)
        - Time Stop:
            - 5m 기준 8봉(40분) 동안 유리방향 진행이 일정 수준 미만이면 시장가 청산
        """
        try:
            sem = self.exit_sem

            # -------- 방향 판별 --------
            side_raw = str(pos.get("side") or "").lower()
            is_long = ("buy" in side_raw) if side_raw else True
            if side_raw:
                is_long = ("buy" in side_raw)
            else:
                # fallback: contracts sign이 없는 경우가 많아서 기본 LONG
                is_long = True

            # hedge 포지션 idx
            pos_idx = self._extract_position_idx(pos, default=0)
            key = (symbol, int(pos_idx))

            entry_price = float(self._get_pos_num(pos, "entryPrice", "avgPrice", "average", default=0.0) or 0.0)
            curr_p = float(curr_p or 0.0)
            if curr_p <= 0:
                return

            # -------- meta 업데이트 (time stop / best move) --------
            now = time.time()
            meta = self.pos_meta_map.get(key) or {}
            if not meta:
                meta = {
                    "entry_ts": now,
                    "entry_price": entry_price if entry_price > 0 else curr_p,
                    "best_price": curr_p,
                    "best_profit_pct": 0.0,
                }
            ep = float(meta.get("entry_price") or (entry_price if entry_price > 0 else curr_p))
            best_price = float(meta.get("best_price") or curr_p)

            if is_long:
                best_price = max(best_price, curr_p)
                best_profit_pct = (best_price - ep) / ep if ep > 0 else 0.0
            else:
                best_price = min(best_price, curr_p)
                best_profit_pct = (ep - best_price) / ep if ep > 0 else 0.0

            meta["best_price"] = best_price
            meta["best_profit_pct"] = float(best_profit_pct)
            self.pos_meta_map[key] = meta

            # -------- Time Stop --------
            try:
                bars = int((now - float(meta.get("entry_ts") or now)) / (5 * 60))
                if bars >= int(getattr(Config, "TIME_STOP_BARS", 8) or 8):
                    # 진행 기준 = (trail_pct * 0.5) vs 최소 1% 중 큰 값
                    trail_pct = await self._compute_trailing_pct(symbol, curr_p)
                    prog = max(float(trail_pct) * float(getattr(Config, "TIME_STOP_PROGRESS_ATR_MULT", 0.5) or 0.5), 0.01)
                    if float(best_profit_pct) < float(prog):
                        qty = abs(self._get_pos_num(pos, "contracts", "size", default=0.0))
                        if qty > 0:
                            close_side = self._close_side_for_pos(pos)
                            self.queue_notify(
                                f"[TIME_STOP] {symbol} {'LONG' if is_long else 'SHORT'} bars={bars} "
                                f"best={best_profit_pct*100:.2f}% < {prog*100:.2f}% -> CLOSE"
                            )
                            await self._api_call(
                                self.exchange.create_order,
                                symbol,
                                "market",
                                close_side,
                                float(qty),
                                None,
                                params={"reduceOnly": True, "category": "linear", "positionIdx": int(pos_idx)},
                                tag=f"time_stop_close:{symbol}",
                                sem=sem,
                            )
                            return
            except Exception:
                pass

            # -------- 현재 SL/TS 확인 --------
            curr_sl = float(self._get_pos_num(pos, "stopLoss", "stop_loss", "sl", default=0.0) or 0.0)
            curr_ts = float(self._get_pos_num(pos, "trailingStop", "trailing_stop", "ts", default=0.0) or 0.0)

            # -------- SL 복구(누락) --------
            if curr_sl <= 0:
                fallback = self.initial_sl_map.get(key)
                if fallback and float(fallback) > 0:
                    curr_sl = float(fallback)
                elif recovery_sl and float(recovery_sl) > 0:
                    curr_sl = float(recovery_sl)

                if curr_sl > 0:
                    await self.apply_exchange_v5_trading_stop(
                        symbol,
                        sl_price=float(curr_sl),
                        ts_dist=None,
                        pos_idx=int(pos_idx),
                        sem=sem,
                    )

            # -------- 5m SuperTrend 기반 SL 추적 --------
            desired_sl = None
            try:
                dfs = await self.data_manager.fetch_timeframe_data(symbol, "5m", limit=180)
                if dfs is not None and len(dfs) >= 30:
                    # 필요 칼럼 정규화
                    df = dfs.copy()
                    # add_indicators가 이미 될 수도 있으나, 최소한 atr/st를 확보
                    try:
                        df_pack = {"5m": df}
                        df_pack = TechnicalAnalyzer.add_indicators(df_pack)
                        df = df_pack["5m"]
                    except Exception:
                        pass

                    if "atr" in df.columns and "st" in df.columns and "st_dir" in df.columns:
                        atr = float(df["atr"].iloc[-1] or 0.0)
                        st = float(df["st"].iloc[-1] or 0.0)
                        st_dir = float(df["st_dir"].iloc[-1] or 0.0)
                        high_now = float(df["high"].iloc[-1] if "high" in df.columns else df["h"].iloc[-1])
                        low_now = float(df["low"].iloc[-1] if "low" in df.columns else df["l"].iloc[-1])

                        k = float(getattr(Config, "ATR_SL_5M_K", 0.20) or 0.20)

                        if is_long:
                            if st > 0 and st_dir > 0 and low_now >= st:
                                desired_sl = st - (atr * k) if (atr > 0 and k > 0) else st
                                # 유리하게만: SL 올리기
                                if desired_sl and desired_sl > 0:
                                    if curr_sl <= 0 or (desired_sl > curr_sl and desired_sl < curr_p):
                                        curr_sl = float(desired_sl)
                                        await self.apply_exchange_v5_trading_stop(
                                            symbol,
                                            sl_price=float(curr_sl),
                                            ts_dist=None,
                                            pos_idx=int(pos_idx),
                                            sem=sem,
                                        )
                        else:
                            if st > 0 and st_dir < 0 and high_now <= st:
                                desired_sl = st + (atr * k) if (atr > 0 and k > 0) else st
                                # 유리하게만: SL 내리기(숏은 SL이 내려갈수록 유리)
                                if desired_sl and desired_sl > 0:
                                    if curr_sl <= 0 or (desired_sl < curr_sl and desired_sl > curr_p):
                                        curr_sl = float(desired_sl)
                                        await self.apply_exchange_v5_trading_stop(
                                            symbol,
                                            sl_price=float(curr_sl),
                                            ts_dist=None,
                                            pos_idx=int(pos_idx),
                                            sem=sem,
                                        )
            except Exception:
                pass

            # -------- Trailing Stop (ATR% 기반, activation 지연) --------
            try:
                # 이미 TS가 서버에 있으면 curr_ts > 0
                trail_pct = await self._compute_trailing_pct(symbol, curr_p)
                act_pct = await self._compute_activation_pct(symbol, curr_p)

                if ep <= 0:
                    ep = entry_price if entry_price > 0 else curr_p

                if is_long:
                    profit_pct = (curr_p - ep) / ep if ep > 0 else 0.0
                else:
                    profit_pct = (ep - curr_p) / ep if ep > 0 else 0.0

                armed = bool(self.ts_armed_map.get(key) or False)

                if (not armed) and (profit_pct >= act_pct):
                    armed = True
                    self.ts_armed_map[key] = True
                    self.queue_notify(
                        f"[TS_ARM] {symbol} {'LONG' if is_long else 'SHORT'} profit={profit_pct*100:.2f}% >= {act_pct*100:.2f}%"
                    )

                if armed:
                    ts_dist = float(curr_p) * float(trail_pct)
                    last_applied = float(self.ts_dist_map.get(key) or 0.0)

                    # 타이트닝(거리 감소)은 하지 않음: new_dist >= last_dist*1.02 일 때만 갱신
                    if (curr_ts <= 0) or (ts_dist > max(last_applied, curr_ts) * 1.02):
                        await self.apply_exchange_v5_trading_stop(
                            symbol,
                            sl_price=None,
                            ts_dist=float(ts_dist),
                            pos_idx=int(pos_idx),
                            sem=sem,
                        )
                        self.ts_dist_map[key] = float(ts_dist)

            except Exception:
                pass

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[SL_MGR_ERR] {symbol} {e}", include_traceback=True)



    async def execute_aggressive_order(self, symbol, side, qty, note, total_bal, sl_p=None, depth=None):
        """
        [교체 버전]
        - ENTRY가 찍혔는데 주문 단계가 안 보이는 문제를 강제로 드러내고(에러 텔레그램),
        실제 주문이 나가게 안정화.
        - IOC 지정가 + 잔량 시장가 fallback (0 fill이어도 fallback 허용)
        - SL은 '주문 파라미터'로 넣다가 거부되는 케이스가 있어서,
        기본은 진입 후 tradingStop 엔드포인트로 세팅(가장 안정적).
        - hedge 모드에서 positionIdx 필요 오류가 나면 1회 자동 재시도.
        """
        try:
            # -----------------------
            # 사전 체크
            # -----------------------
            pos_before = await self.get_safe_position(symbol)
            if pos_before is None:
                self.queue_notify(f"[SYSTEM_SKIP] {symbol} pos_before 조회 실패(None) -> 주문 스킵")
                return

            c_before = self._get_pos_num(pos_before, "contracts", "size", default=0.0)
            c_before_abs = abs(float(c_before))

            # 오더북 기반 목표가
            step = depth if depth is not None else Config.ORDER_BOOK_DEPTH_STEP
            target_p = await self.data_manager.get_target_price_by_orderbook(symbol, side, depth_step=step)

            if not target_p:
                t = await self._api_call(self.exchange.fetch_ticker, symbol, tag=f"fetch_ticker:{symbol}")
                target_p = t.get("last")

            if not target_p:
                self.queue_notify(f"[ERROR] {symbol} 가격 산출 실패 (orderbook/ticker 모두 실패)")
                return

            target_p = float(self.exchange.price_to_precision(symbol, float(target_p)))

            # qty 정밀도/최소수량 처리
            qty_raw = float(qty)
            qty_prec = float(self.exchange.amount_to_precision(symbol, qty_raw))
            min_q = float(self.get_symbol_min_amount(symbol) or 0.0)

            if qty_prec <= 0:
                self.queue_notify(f"[ENTRY_BLOCK] {symbol} qty가 0으로 반올림됨 (raw={qty_raw}, min={min_q})")
                return

            if min_q > 0 and qty_prec < min_q:
                self.queue_notify(f"[WARN] {symbol} qty<{min_q} → minQty로 보정 (qty:{qty_prec}→{min_q})")
                qty_prec = float(self.exchange.amount_to_precision(symbol, min_q))
                if qty_prec <= 0:
                    self.queue_notify(f"[ENTRY_BLOCK] {symbol} minQty 보정 후에도 qty=0")
                    return

            # SL sanity (롱 기준만)
            sl_sanitized = None
            try:
                if sl_p and float(sl_p) > 0:
                    s = str(side).lower()
                    if s == "buy":
                        sl_sanitized = self._sanitize_long_sl(target_p, float(sl_p), min_gap_pct=0.002)
                    elif s == "sell":
                        sl_sanitized = self._sanitize_short_sl(target_p, float(sl_p), min_gap_pct=0.002)
            except Exception:
                sl_sanitized = None

            # ---------------------------------------------------
            # [PATCH] target_p/SL 기준으로 레버리지·수량 재산출
            # - float 레버리지 유지
            # - 심볼별 max_lev, step 적용
            # - 최소 레버리지 1.5 적용
            # ---------------------------------------------------
            try:
                if sl_sanitized and float(sl_sanitized) > 0 and float(target_p) > 0:
                    max_lev2 = await self.get_symbol_max_leverage(symbol)
                    lev_step2 = await self.get_symbol_leverage_step(symbol)

                    curr_m2 = self._estimate_position_margin(pos_before, curr_price=float(target_p))

                    lev2, qty2, _ = RiskManager.calculate_entry_params(
                        float(total_bal),
                        float(target_p),
                        float(sl_sanitized),
                        float(max_lev2),
                        current_m=float(curr_m2 or 0.0),
                        leverage_step=float(lev_step2),
                        min_leverage=1.5,
                    )

                    # 최종 레버리지로 덮어쓰기(진입 직전 정합)
                    if lev2 and float(lev2) > 0:
                        try:
                            await self._api_call(
                                self.exchange.set_leverage,
                                float(lev2),
                                symbol,
                                params={"category": "linear"},
                                tag=f"set_leverage_final:{symbol}",
                            )
                        except Exception:
                            pass

                    # 리스크 초과 방지: 더 작아지면 축소
                    if qty2 and float(qty2) > 0:
                        qty2_prec = float(self.exchange.amount_to_precision(symbol, float(qty2)))
                        if qty2_prec > 0 and qty2_prec < qty_prec:
                            self.queue_notify(
                                f"[RISK_CAP] {symbol} qty 조정: {qty_prec}->{qty2_prec} "
                                f"(lev:{lev2}, P:{target_p}, SL:{sl_sanitized})"
                            )
                            qty_prec = qty2_prec
            except Exception:
                pass

            self.queue_notify(
                f"[ORDER_ATTEMPT] {side.upper()} {symbol}\n전략: {note}\n가격: {target_p}\n수량: {qty_prec}\nSL:{sl_sanitized if sl_sanitized else 'post-set'}"
            )

            # -----------------------
            # 주문 파라미터
            # -----------------------
            base_params = {"category": "linear"}
            if Config.ENTRY_TIME_IN_FORCE:
                base_params["timeInForce"] = str(Config.ENTRY_TIME_IN_FORCE)

            # entry는 reduceOnly 금지
            base_params.pop("reduceOnly", None)

            # -----------------------
            # 1) IOC 지정가 주문 (positionIdx 필요하면 자동 재시도)
            # -----------------------
            async def _try_limit(params: dict):
                return await self._api_call(
                    self.exchange.create_order,
                    symbol,
                    "limit",
                    side,
                    qty_prec,
                    target_p,
                    params=params,
                    tag=f"create_order_limit:{symbol}",
                )

            order = None
            try:
                order = await _try_limit(dict(base_params))
            except Exception as e:
                msg = str(e)

                # hedge mode / positionIdx required 케이스 자동 재시도
                if ("positionIdx" in msg) or ("position idx" in msg) or ("position index" in msg):
                    retry_params = dict(base_params)
                    retry_params["positionIdx"] = 1 if str(side).lower() == "buy" else 2
                    self.queue_notify(f"[RETRY] {symbol} positionIdx 필요로 판단 → positionIdx={retry_params['positionIdx']} 재시도")
                    order = await _try_limit(retry_params)
                else:
                    raise

            await asyncio.sleep(float(Config.ENTRY_STATUS_WAIT_SEC))

            # -----------------------
            # 2) 체결 추정(포지션 델타 기반) + 잔량 시장가 fallback
            #    (중요) 0 fill이어도 fallback 허용!
            # -----------------------
            pos_after = await self.get_safe_position(symbol)
            if pos_after is None:
                self.queue_notify(f"[SYSTEM_WARN] {symbol} pos_after 조회 실패(None) -> fallback 판단은 order.filled 기준만")
                pos_after = {}

            c_after = self._get_pos_num(pos_after, "contracts", "size", default=0.0)
            c_after_abs = abs(float(c_after))

            try:
                order_filled = float((order or {}).get("filled") or 0.0)
            except Exception:
                order_filled = 0.0

            pos_delta = max(0.0, c_after_abs - c_before_abs)
            filled_est = max(order_filled, pos_delta)
            filled_est = float(self.exchange.amount_to_precision(symbol, filled_est))

            rem = max(0.0, float(qty_prec) - float(filled_est))
            rem = float(self.exchange.amount_to_precision(symbol, rem))

            # 시장가 fallback (0 fill도 포함)
            if Config.ENTRY_MARKET_FALLBACK and rem > 0:
                market_params = {"category": "linear"}
                # hedge 모드일 수 있으면 positionIdx도 같이
                try:
                    pidx = self._extract_position_idx(pos_after, default=0)
                    if int(pidx) in (1, 2):
                        market_params["positionIdx"] = int(pidx)
                except Exception:
                    pass

                self.queue_notify(f"[FALLBACK] {symbol} 잔량 시장가 처리 rem={rem}")
                await self._api_call(
                    self.exchange.create_market_order,
                    symbol,
                    side,
                    rem,
                    params=market_params,
                    tag=f"create_order_market_fallback:{symbol}",
                )

            self.queue_notify(f"[ORDER_DONE] {side.upper()} {symbol} IOC+fallback 완료")

            # -----------------------
            # 3) 진입 후: SL/TS 서버 등록(가장 안정적인 방식)
            # -----------------------
            try:
                await asyncio.sleep(0.2)
                pos_now = await self.get_safe_position(symbol)
                if pos_now is not None:
                    pidx_now = self._extract_position_idx(pos_now, default=0)

                    # 최초 SL 저장(전략 note / pos idx)
                    self.entry_strategy_map[(symbol, pidx_now)] = str(note)
                    self.bb_armed_map[(symbol, pidx_now)] = False
                    if sl_sanitized and float(sl_sanitized) > 0:
                        self.initial_sl_map.setdefault((symbol, pidx_now), float(sl_sanitized))

                    # TS 거리(5%) - 현재 TS OFF 상태면 계산만 하고 사용 안 함
                    try:
                        t2 = await self._api_call(self.exchange.fetch_ticker, symbol, tag=f"postset_ticker:{symbol}")
                        px_now = float(t2.get("last") or 0.0) or float(target_p)
                    except Exception:
                        px_now = float(target_p)

                    ts_gap_dist = float(px_now) * float(Config.EXCHANGE_TRAILING_STOP_GAP)

                    # SL/TS 등록 (TS OFF)
                    await self.apply_exchange_v5_trading_stop(
                        symbol,
                        sl_price=float(sl_sanitized) if (sl_sanitized and float(sl_sanitized) > 0) else None,
                        ts_dist=None,  # TS OFF
                        pos_idx=int(pidx_now),
                        sem=self.exit_sem,
                    )

                    self.queue_notify(f"[POST_SET] {symbol} SL/TS 서버 등록 완료 (pos_idx={pidx_now})")

            except Exception as e:
                self.queue_notify(f"[POST_SET_WARN] {symbol} SL/TS 등록 실패: {e}")

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[ORDER_FATAL] {symbol} {e}", include_traceback=True)
            self.queue_notify(f"[ORDER_FATAL] {symbol} 주문 집행 실패: {e}")



    async def manage_profit_taking(self, symbol, pos, curr_p, total_bal):
        """
        부분익절: total_bal 기준 PnL(USDT) 임계치 도달 시 청산
        """
        try:
            if total_bal is None or float(total_bal) <= 0:
                return

            current_contracts = abs(self._get_pos_num(pos, "contracts", "size", default=0.0))
            if current_contracts <= 0:
                return

            pnl_usdt = self._get_pos_num(pos, "unrealisedPnl", "unrealised_pnl", default=0.0)
            if pnl_usdt <= 0:
                return

            tp1_usdt = float(total_bal) * 0.03
            tp2_usdt = float(total_bal) * 0.05

            pos_idx = self._extract_position_idx(pos, default=0)
            key = (symbol, pos_idx)
            status_idx = self.tp_status.get(key, 0)

            close_ratio = 0.0
            tp_stage = 0

            if pnl_usdt >= tp2_usdt and status_idx < 2:
                close_ratio = 0.50
                tp_stage = 2
            elif pnl_usdt >= tp1_usdt and status_idx < 1:
                close_ratio = 0.20
                tp_stage = 1
            else:
                return

            close_side = self._close_side_for_pos(pos)

            close_q = float(self.exchange.amount_to_precision(symbol, current_contracts * close_ratio))
            try:
                min_order = float(self.exchange.market(symbol)["limits"]["amount"]["min"])
            except Exception:
                min_order = 0.0

            if close_q <= 0 or close_q < min_order:
                write_log(ERROR_LOG_FILE, f"[TP_SKIP_MIN] {symbol} close_q={close_q} < min={min_order}")
                return

            self.tp_status[key] = tp_stage

            await self.execute_exit_order(symbol, close_side, close_q, f"TP{tp_stage}", total_bal, pos_idx=pos_idx)

            self.queue_notify(
                f"[TP{tp_stage}] {symbol} pnl={pnl_usdt:.2f}USDT "
                f"(>= {tp1_usdt:.2f}/{tp2_usdt:.2f}) -> {int(close_ratio*100)}% 청산"
            )

        except Exception as e:
            write_log(ERROR_LOG_FILE, f"[TP_ERROR] {symbol}: {e}", include_traceback=True)



    async def manage_hard_stop(self, symbol, pos, total_bal):
        """
        손실 한도 도달 시 전량 청산
        """
        try:
            if total_bal is None or float(total_bal) <= 0:
                return False

            pnl_val = self._get_pos_num(pos, "unrealisedPnl", "unrealised_pnl", default=0.0)

            if (pnl_val / float(total_bal)) * 100 <= Config.HARD_STOP_LOSS_PERCENT:
                qty = abs(self._get_pos_num(pos, "contracts", "size", default=0.0))
                if qty > 0:
                    close_side = self._close_side_for_pos(pos)
                    pos_idx = self._extract_position_idx(pos, default=0)
                    lock_key = (symbol, pos_idx)

                    self.queue_notify(f"[HARD_STOP] {symbol} 손실 한도 초과. 전량 청산(시장가) 시도.")
                    await self.execute_exit_order(symbol, close_side, qty, "HardStop", total_bal, pos_idx=pos_idx)

                    try:
                        await self._api_call(self.exchange.cancel_all_orders, symbol, tag=f"exit_cancel_all:{symbol}", sem=self.exit_sem)
                    except Exception:
                        pass

                    self._purge_position_state(lock_key)
                    return True
        except Exception:
            pass

        return False



    
    async def exit_loop(self):
        """
        [NEW] 포지션 관리 루프 (LONG/SHORT 공용)
        - 긴 로직(다중 조건/예외) 대신 "안정성 우선" 버전으로 단순화
        - 핵심:
            1) 5m EMA20 반대 돌파 + 반대봉 확인 -> 즉시 시장가 청산(강제)
            2) 그 외에는 manage_dynamic_stop_loss()에서 SL/TS/TimeStop 관리
            3) (옵션) ST flip top-up은 LONG에서만 유지
        """
        await asyncio.sleep(3)
        while self.is_running:
            try:
                positions = await self.get_all_open_positions()
                if not positions:
                    await asyncio.sleep(Config.EXIT_LOOP_SEC)
                    continue

                for pos in positions:
                    try:
                        symbol = pos.get("symbol")
                        if not symbol:
                            continue

                        pos_idx = self._extract_position_idx(pos, default=0)
                        side_raw = str(pos.get("side") or "").lower()
                        is_long = True if not side_raw else ("buy" in side_raw)

                        qty = abs(self._get_pos_num(pos, "contracts", "size", default=0.0))
                        if qty <= 0:
                            continue

                        # 현재가
                        try:
                            t = await self._api_call(self.exchange.fetch_ticker, symbol, tag=f"ticker:{symbol}", sem=self.api_sem)
                            curr_p = float(t.get("last") or t.get("close") or 0.0)
                        except Exception:
                            curr_p = 0.0
                        if curr_p <= 0:
                            continue

                        # 5m 데이터로 EMA20 확인(긴급청산)
                        df5 = await self.data_manager.fetch_timeframe_data(symbol, "5m", limit=120)
                        if df5 is None or len(df5) < 30:
                            # 데이터 부족이면 SL/TS만 관리
                            await self.manage_dynamic_stop_loss(symbol, pos, curr_p)
                            continue

                        try:
                            pack = {"5m": df5.copy()}
                            pack = TechnicalAnalyzer.add_indicators(pack)
                            df5i = pack["5m"]
                        except Exception:
                            df5i = df5

                        # 캔들 + EMA20
                        try:
                            o_now = float(df5i["open"].iloc[-1])
                            c_now = float(df5i["close"].iloc[-1])
                            e_now = float(df5i["ema20"].iloc[-1]) if "ema20" in df5i.columns else None
                        except Exception:
                            o_now, c_now, e_now = 0.0, 0.0, None

                        full_exit = False
                        if e_now and float(e_now) > 0:
                            bull_now = (c_now > o_now)
                            bear_now = (c_now < o_now)

                            if is_long:
                                # 롱: EMA20 하향 이탈 + 음봉
                                if (c_now < float(e_now)) and bear_now:
                                    full_exit = True
                            else:
                                # 숏: EMA20 상향 돌파 + 양봉
                                if (c_now > float(e_now)) and bull_now:
                                    full_exit = True

                        if full_exit:
                            close_side = self._close_side_for_pos(pos)
                            self.queue_notify(
                                f"[EXIT_EMA20] {symbol} {'LONG' if is_long else 'SHORT'} | qty={qty} P={curr_p}"
                            )
                            await self._api_call(
                                self.exchange.create_order,
                                symbol,
                                "market",
                                close_side,
                                float(qty),
                                None,
                                params={"reduceOnly": True, "category": "linear", "positionIdx": int(pos_idx)},
                                tag=f"exit_ema20:{symbol}",
                                sem=self.exit_sem,
                            )
                            continue

                        # (옵션) ST flip top-up: LONG에서만
                        try:
                            if is_long:
                                await self.maybe_stflip_topup(symbol, pos, curr_p)
                        except Exception:
                            pass

                        # SL/TS/TimeStop 관리
                        await self.manage_dynamic_stop_loss(symbol, pos, curr_p)

                    except Exception as e:
                        write_log(ERROR_LOG_FILE, f"[EXIT_POS_ERR] {e}", include_traceback=True)
                        continue

                await asyncio.sleep(Config.EXIT_LOOP_SEC)

            except Exception as e:
                write_log(ERROR_LOG_FILE, f"[EXIT_LOOP_ERR] {e}", include_traceback=True)
                await asyncio.sleep(Config.EXIT_LOOP_SEC)


    async def process_symbol(self, symbol, total_bal, side: str = "long", common_msg: str = ""):
        """
        [NEW] 엔트리 처리 (롱/숏 공용)
        - 공통조건 통과(상위에서 필터링) 후, 엔트리 TF(1m/5m/15m) 시그널을 검사
        - side: "long" or "short"
        """
        side = str(side or "long").lower().strip()
        if side not in ("long", "short"):
            side = "long"

        async with self._get_entry_lock(symbol):
            try:
                # 이미 포지션 있으면 스킵(방향 무관)
                pos0 = await self.get_safe_position(symbol)
                if pos0 is None:
                    return
                c0 = abs(self._get_pos_num(pos0, "contracts", "size", default=0.0))
                if c0 > 0:
                    return

                # 데이터 fetch + 지표
                dfs = await self.data_manager.fetch_entry_data(symbol, limit=260)
                if not dfs:
                    return
                try:
                    dfs = TechnicalAnalyzer.add_indicators(dfs)
                except Exception:
                    return

                sig, _, entry_sl, st_name, sl_src, _, details = await TechnicalAnalyzer.check_signals(dfs, side=side)
                if not sig:
                    return

                df5 = dfs.get("5m")
                if df5 is None or len(df5) < 5:
                    return

                entry_price = float(df5["close"].iloc[-1])

                # SL sanitize
                sl_final = float(entry_sl or 0.0)
                if sl_final <= 0:
                    return

                if side == "long":
                    sl_final = float(self._sanitize_long_sl(entry_price, sl_final, min_gap_pct=0.002))
                    order_side = "buy"
                else:
                    sl_final = float(self._sanitize_short_sl(entry_price, sl_final, min_gap_pct=0.002))
                    order_side = "sell"

                # 레버리지/수량 산정
                try:
                    max_lev = await self.get_symbol_max_leverage(symbol)
                    lev_step = await self.get_symbol_leverage_step(symbol)
                except Exception:
                    max_lev = float(getattr(Config, "MAX_LEVERAGE", 3.0) or 3.0)
                    lev_step = 0.1

                curr_m = self._estimate_position_margin(pos0, curr_price=float(entry_price))

                lev, qty, risk_pct = RiskManager.calculate_entry_params(
                    float(total_bal),
                    float(entry_price),
                    float(sl_final),
                    float(max_lev),
                    current_m=float(curr_m or 0.0),
                    leverage_step=float(lev_step),
                    min_leverage=1.5,
                )
                if not qty or float(qty) <= 0:
                    return

                # 레버리지 세팅
                try:
                    await self._api_call(
                        self.exchange.set_leverage,
                        float(lev),
                        symbol,
                        params={"category": "linear"},
                        tag=f"set_leverage:{symbol}",
                    )
                except Exception:
                    pass

                note = f"{'LONG' if side=='long' else 'SHORT'}|{st_name}"
                if common_msg:
                    self.queue_notify(f"[COMMON_OK] {symbol}\n{common_msg}")

                self.queue_notify(
                    f"[ENTRY] {symbol} {note}\nP:{entry_price:.6f} SL:{sl_final:.6f} lev:{lev} qty:{qty}"
                )

                # 주문 집행(진입 후 SL 서버등록)
                await self.execute_aggressive_order(
                    symbol,
                    order_side,
                    qty,
                    note,
                    total_bal,
                    sl_p=float(sl_final),
                    depth=None,
                )

            except Exception as e:
                write_log(ERROR_LOG_FILE, f"[PROCESS_ERR] {symbol} {e}", include_traceback=True)
                self.queue_notify(f"[PROCESS_ERR] {symbol} 예외: {e}")



    async def run_loop(self):
        """
        [NEW] 메인 루프
        1) 후보심볼(최대 100) 선정
        2) 후보 100개 대상으로 LONG/SHORT 공통조건을 동시에 평가 -> 레짐(롱/숏) 결정
        3) 심볼별 독립 방향 결정: long_ok → 롱, short_ok → 숏 (레짐 없음)
        4) 엔트리 시그널 체크 후 진입
        """
        await asyncio.sleep(2)
        while self.is_running:
            try:
                # -------------------------
                # 0) 잔고
                # -------------------------
                try:
                    bal_info = await self._api_call(self.exchange.fetch_balance, tag="main_bal", sem=self.api_sem)
                    total_bal = float((bal_info.get("total") or {}).get("USDT") or 0.0)
                except Exception:
                    total_bal = 0.0

                # -------------------------
                # 1) 후보 심볼
                # -------------------------
                try:
                    async with self.candidate_lock:
                        sym_list = list(self.candidate_symbols or [])
                except Exception:
                    sym_list = []

                if not sym_list:
                    await asyncio.sleep(Config.MAIN_LOOP_SEC)
                    continue

                random.shuffle(sym_list)

                # -------------------------
                # 2) 공통조건 평가 (LONG/SHORT 동시, 심볼별 독립)
                # -------------------------
                common_map = {}  # symbol -> (long_ok, short_ok, msg_long, msg_short)

                async def _eval_common_one(sym):
                    try:
                        dfs_common = await self.data_manager.fetch_common_data(sym, limit=260)
                        if not dfs_common:
                            return sym, (False, False, "", "")
                        dfs_common = TechnicalAnalyzer.add_indicators(dfs_common)
                        l_ok, s_ok, l_msg, s_msg = await TechnicalAnalyzer.check_common_conditions_sides(dfs_common)
                        return sym, (l_ok, s_ok, l_msg, s_msg)
                    except Exception:
                        return sym, (False, False, "", "")

                batch = []
                for s in sym_list:
                    batch.append(s)
                    if len(batch) >= int(getattr(Config, "MAIN_BATCH_SIZE", 10) or 10):
                        tasks = [_eval_common_one(x) for x in batch]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for r in results:
                            if isinstance(r, Exception):
                                continue
                            sym, tup = r
                            common_map[sym] = tup
                        batch = []

                if batch:
                    tasks = [_eval_common_one(x) for x in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for r in results:
                        if isinstance(r, Exception):
                            continue
                        sym, tup = r
                        common_map[sym] = tup

                # -------------------------
                # 3) 심볼별 독립 방향 결정
                # - long_ok만 → 롱 진입 후보
                # - short_ok만 → 숏 진입 후보 (ENABLE_SHORT=True일 때)
                # - 둘 다 or 둘 다 아님 → 스킵 (신호 불명확)
                # -------------------------
                enable_short = bool(getattr(Config, "ENABLE_SHORT", True))
                # (sym, msg, side) 형태로 구성
                ent_candidates = []
                long_cnt = 0
                short_cnt = 0
                for sym, (l_ok, s_ok, l_msg, s_msg) in common_map.items():
                    if l_ok and not s_ok:
                        ent_candidates.append((sym, l_msg, "long"))
                        long_cnt += 1
                    elif s_ok and not l_ok and enable_short:
                        ent_candidates.append((sym, s_msg, "short"))
                        short_cnt += 1
                    # 둘 다 True거나 둘 다 False면 스킵

                if not ent_candidates:
                    await asyncio.sleep(Config.MAIN_LOOP_SEC)
                    continue

                # 텔레그램에 현재 루프 요약 알림 (너무 자주 보내지 않도록 10루프마다)
                if not hasattr(self, "_loop_cnt"):
                    self._loop_cnt = 0
                self._loop_cnt += 1
                if self._loop_cnt % 10 == 1:
                    self.queue_notify(
                        f"[SCAN] long_cand={long_cnt} short_cand={short_cnt} "
                        f"total={len(ent_candidates)}/{len(sym_list)}"
                    )

                # -------------------------
                # 4) 엔트리 시그널 체크 (배치)
                # -------------------------
                ent_batch = []
                for sym, msg, side in ent_candidates:
                    ent_batch.append((sym, msg, side))
                    if len(ent_batch) >= int(getattr(Config, "MAIN_BATCH_SIZE", 10) or 10):
                        tasks = [self.process_symbol(x, total_bal, side=sd, common_msg=m) for x, m, sd in ent_batch]
                        await asyncio.gather(*tasks, return_exceptions=True)
                        ent_batch = []

                if ent_batch:
                    tasks = [self.process_symbol(x, total_bal, side=sd, common_msg=m) for x, m, sd in ent_batch]
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(Config.MAIN_LOOP_SEC)

            except Exception as e:
                write_log(ERROR_LOG_FILE, f"[MAIN_LOOP_ERR] {e}", include_traceback=True)
                self.queue_notify(f"[MAIN_LOOP_ERR] {e}")
                await asyncio.sleep(Config.MAIN_LOOP_SEC)
def _configure_event_loop_policy():
    try:
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


async def _main_async():
    """
    엔트리/청산/알림/유니버스 갱신 루프를 함께 구동하는 메인 엔트리포인트.
    - setup()에서 거래소/텔레그램/초기 유니버스 준비
    - notification_worker: 알림 큐 배출
    - candidate/universe refresher: 후보/유니버스 주기 갱신
    - exit_loop: 포지션 관리(스탑/TS/타임스탑 등)
    - run_loop: 신규 진입 감시
    """
    bot = AsyncTradingBot()
    await bot.setup()

    # 알림 워커 먼저(BOOT/START 메시지 누락 방지)
    notif_task = asyncio.create_task(bot.notification_worker(), name="notification_worker")

    # 백그라운드 루프들
    cand_task = asyncio.create_task(bot.candidate_refresher_loop(), name="candidate_refresher")
    uni_task = asyncio.create_task(bot.universe_refresher_loop(), name="universe_refresher")
    rep_task = asyncio.create_task(bot.periodic_reporter(), name="periodic_reporter")

    # 포지션 관리 루프(필수)
    exit_task = asyncio.create_task(bot.exit_loop(), name="exit_loop")

    # 시작 알림(큐로 1회만)
    try:
        start_msg = (
            f"[BOT START] mode={'TESTNET' if Config.IS_TEST_MODE else 'REAL'} | "
            f"short={'ON' if getattr(Config, 'ENABLE_SHORT', True) else 'OFF'} | "
            f"universe={len(getattr(bot, 'universe_symbols', []) or [])} "
            f"candidates={len(getattr(bot, 'candidate_symbols', []) or [])}"
        )
        bot.queue_notify(start_msg)
    except Exception:
        pass

    main_task = asyncio.create_task(bot.run_loop(), name="main_loop")

    try:
        await asyncio.gather(main_task, exit_task)
    finally:
        bot.is_running = False
        tasks = [main_task, exit_task, cand_task, uni_task, rep_task, notif_task]
        for task in tasks:
            try:
                if not task.done():
                    task.cancel()
            except Exception:
                pass
        await asyncio.gather(*tasks, return_exceptions=True)

        # 리소스 정리
        try:
            if bot.telegram:
                await bot.telegram.close()
        except Exception:
            pass
        try:
            if bot.exchange:
                await bot.exchange.close()
        except Exception:
            pass


def main():
    multiprocessing.freeze_support()
    _configure_event_loop_policy()
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        print("[시스템] 수동 종료되었습니다.")


if __name__ == "__main__":
    main()
