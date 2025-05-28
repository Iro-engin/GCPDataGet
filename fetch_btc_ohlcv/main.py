import functions_framework as fw
import ccxt
import pandas as pd
import ta
from google.cloud import bigquery
import os
from datetime import datetime, timezone
import numpy

GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
BQ_DATASET_ID = os.environ.get('BQ_DATASET_ID')
BQ_TABLE_ID = os.environ.get('BQ_TABLE_ID') # テーブル名のみ (例: 'ohlcv_indicators')

EXCHANGE_ID = os.environ.get('EXCHANGE_ID', 'binance')
SYMBOL = os.environ.get('SYMBOL', 'BTC/USDT')
TIMEFRAME = os.environ.get('TIMEFRAME', '15m')
# テクニカル指標の計算に必要な期間よりも十分に大きい値を設定してください
LIMIT_CANDLES = int(os.environ.get('LIMIT_CANDLES', '100')) # 例: 100本のローソク足を取得

# テクニカル指標のパラメータ (環境変数で設定可能にする)
SMA_PERIOD = int(os.environ.get('SMA_PERIOD', '100'))
EMA_PERIOD = int(os.environ.get('EMA_PERIOD', '200'))
MACD_FAST_PERIOD = int(os.environ.get('MACD_FAST_PERIOD', '6'))
MACD_SLOW_PERIOD = int(os.environ.get('MACD_SLOW_PERIOD', '13'))
MACD_SIGN_PERIOD = int(os.environ.get('MACD_SIGN_PERIOD', '9'))

# BigQueryクライアントの初期化
bigquery_client = bigquery.Client(project=GCP_PROJECT_ID)

@functions_frammework.http
def fetch_btc_ohlcv(request):
    try:
        print(f"Function triggered. Fetching {SYMBOL} {TIMEFRAME} data from {EXCHANGE_ID}.")
        print(f"Calculating SMA({SMA_PERIOD}), EMA({EMA_PERIOD}), MACD({MACD_FAST_PERIOD},{MACD_SLOW_PERIOD},{MACD_SIGN_PERIOD}).")
        print(f"Target BigQuery table: {GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}")

        # 1. CCXTでOHLCVデータを取得
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        exchange = exchange_class({'enableRateLimit': True})
        
        ohlcv_raw = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT_CANDLES)
        if not ohlcv_raw:
            message = f"No OHLCV data returned for {SYMBOL} from {EXCHANGE_ID}."
            print(message)
            return message, 204

        # 2. Pandas DataFrameに変換
        # カラム名: timestamp_ms, open, high, low, close, volume
        df = pd.DataFrame(ohlcv_raw, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            message = "Fetched OHLCV data is empty after DataFrame conversion."
            print(message)
            return message, 204
            
        # データ型を適切に設定
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])

        # 3. テクニカル指標の計算
        # SMA
        sma_indicator = ta.trend.SMAIndicator(close=df['close'], window=SMA_PERIOD, fillna=True)
        df['sma'] = sma_indicator.sma_indicator()

        # EMA
        ema_indicator = ta.trend.EMAIndicator(close=df['close'], window=EMA_PERIOD, fillna=True)
        df['ema'] = ema_indicator.ema_indicator()

        # MACD
        macd_indicator = ta.trend.MACD(
            close=df['close'],
            window_slow=MACD_SLOW_PERIOD,
            window_fast=MACD_FAST_PERIOD,
            window_sign=MACD_SIGN_PERIOD,
            fillna=True
        )
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff() # ヒストグラム (差分)

        # 4. BigQuery用のデータ準備
        # BigQueryのTIMESTAMP型に合うように変換 (ミリ秒から秒へ、そしてdatetimeオブジェクトへ)
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
        
        # 共通情報を追加
        df['exchange'] = EXCHANGE_ID
        df['symbol'] = SYMBOL
        df['timeframe'] = TIMEFRAME

        # BigQueryに投入するカラムを選択
        # 注意: BigQueryのテーブルスキーマとカラム名・順序を一致させるか、
        # insert_rows_jsonではカラム名が一致していれば順序は問われません。
        columns_for_bq = [
            'timestamp', 'exchange', 'symbol', 'timeframe',
            'open', 'high', 'low', 'close', 'volume',
            'sma', 'ema', 'macd', 'macd_signal', 'macd_hist'
        ]
        df_to_load = df[columns_for_bq].copy()

        # NaN値をNoneに置換 (BigQueryはNoneをNULLとして扱える)
        # pandas 1.x.x では df.replace({np.nan: None}) は文字列の'NaN'も置換してしまう可能性があるので注意
        # df_to_load = df_to_load.astype(object).where(pd.notnull(df_to_load), None) # より安全な方法
        for col in ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'macd', 'macd_signal', 'macd_hist']:
            # np.isfiniteを使って数値型カラムの +/- inf も NaN 扱いにする
            df_to_load[col] = df_to_load[col].apply(lambda x: x if np.isfinite(x) else None)


        rows_to_insert = df_to_load.to_dict('records')

        if not rows_to_insert:
            message = "No data to insert into BigQuery after processing."
            print(message)
            return message, 204

        # 5. BigQueryにデータを追記
        table_ref_str = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"
        
        errors = bigquery_client.insert_rows_json(table_ref_str, rows_to_insert)
        if errors == []:
            success_message = f"Successfully loaded {len(rows_to_insert)} rows into {table_ref_str}"
            print(success_message)
            return success_message, 200
        else:
            error_message = f"Errors encountered while inserting rows into {table_ref_str}: {errors}"
            print(error_message)
            # エラー内容によっては部分的に成功している可能性もある
            return error_message, 500

    except ccxt.NetworkError as e:
        error_message = f"CCXT NetworkError: {e}"
        print(error_message)
        return error_message, 503 # Service Unavailable
    except ccxt.ExchangeError as e:
        error_message = f"CCXT ExchangeError: {e} (Detail: {str(e.args)})"
        print(error_message)
        return error_message, 502 # Bad Gateway
    except AttributeError as e:
        error_message = f"AttributeError (e.g., invalid exchange ID '{EXCHANGE_ID}' for ccxt or issue with 'ta' library): {e}"
        print(error_message)
        return error_message, 400 # Bad Request
    except Exception as e:
        import traceback
        error_message = f"An unexpected error occurred: {e}\nTraceback: {traceback.format_exc()}"
        print(error_message)
        return error_message, 500