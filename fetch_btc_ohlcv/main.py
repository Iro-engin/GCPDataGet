# crypto-indicator-service-docker/main.py
from flask import Flask, request as flask_request
import ccxt
import pandas as pd
import ta
from google.cloud import bigquery
import os
from datetime import datetime, timezone
import numpy as np
import traceback # エラー詳細出力用

app = Flask(__name__)

# --- 環境変数 (Cloud Runサービスに設定) ---
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
BQ_DATASET_ID = os.environ.get('BQ_DATASET_ID')
BQ_TABLE_ID = os.environ.get('BQ_TABLE_ID')

EXCHANGE_ID = os.environ.get('EXCHANGE_ID', 'binance')
SYMBOL = os.environ.get('SYMBOL', 'BTC/USDT')
TIMEFRAME = os.environ.get('TIMEFRAME', '15m')
LIMIT_CANDLES = int(os.environ.get('LIMIT_CANDLES', '100'))

SMA_PERIOD = int(os.environ.get('SMA_PERIOD', '20'))
EMA_PERIOD = int(os.environ.get('EMA_PERIOD', '50'))
MACD_FAST_PERIOD = int(os.environ.get('MACD_FAST_PERIOD', '12'))
MACD_SLOW_PERIOD = int(os.environ.get('MACD_SLOW_PERIOD', '26'))
MACD_SIGN_PERIOD = int(os.environ.get('MACD_SIGN_PERIOD', '9'))

bigquery_client = None # グローバル変数として宣言

def get_bq_client():
    """ BigQueryクライアントを初期化 (必要な場合のみ) """
    global bigquery_client
    if bigquery_client is None:
        if not GCP_PROJECT_ID:
            print("Error: GCP_PROJECT_ID environment variable is not set.")
            # 実際のアプリケーションではここでエラーを発生させるか、デフォルト値を設定
            return None
        bigquery_client = bigquery.Client(project=GCP_PROJECT_ID)
    return bigquery_client

def core_logic():
    """ 主要なデータ取得・計算・BigQuery保存ロジック """
    client = get_bq_client()
    if not client:
        return "BigQuery client could not be initialized due to missing GCP_PROJECT_ID.", 500
    if not BQ_DATASET_ID or not BQ_TABLE_ID:
        return "BQ_DATASET_ID or BQ_TABLE_ID environment variables are not set.", 500

    print(f"Logic triggered. Fetching {SYMBOL} {TIMEFRAME} data from {EXCHANGE_ID}.")
    print(f"Calculating SMA({SMA_PERIOD}), EMA({EMA_PERIOD}), MACD({MACD_FAST_PERIOD},{MACD_SLOW_PERIOD},{MACD_SIGN_PERIOD}).")
    print(f"Target BigQuery table: {GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}")

    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({'enableRateLimit': True})
    
    ohlcv_raw = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT_CANDLES)
    if not ohlcv_raw:
        return f"No OHLCV data returned for {SYMBOL} from {EXCHANGE_ID}.", 204

    df = pd.DataFrame(ohlcv_raw, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
    if df.empty:
        return "Fetched OHLCV data is empty after DataFrame conversion.", 204
        
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])

    sma_indicator = ta.trend.SMAIndicator(close=df['close'], window=SMA_PERIOD, fillna=True)
    df['sma'] = sma_indicator.sma_indicator()
    ema_indicator = ta.trend.EMAIndicator(close=df['close'], window=EMA_PERIOD, fillna=True)
    df['ema'] = ema_indicator.ema_indicator()
    macd_indicator = ta.trend.MACD(
        close=df['close'], window_slow=MACD_SLOW_PERIOD, window_fast=MACD_FAST_PERIOD,
        window_sign=MACD_SIGN_PERIOD, fillna=True
    )
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    df['exchange'] = EXCHANGE_ID
    df['symbol'] = SYMBOL
    df['timeframe'] = TIMEFRAME

    columns_for_bq = [
        'timestamp', 'exchange', 'symbol', 'timeframe',
        'open', 'high', 'low', 'close', 'volume',
        'sma', 'ema', 'macd', 'macd_signal', 'macd_hist'
    ]
    df_to_load = df[columns_for_bq].copy()

    for col in ['open', 'high', 'low', 'close', 'volume', 'sma', 'ema', 'macd', 'macd_signal', 'macd_hist']:
        df_to_load[col] = df_to_load[col].apply(lambda x: x if np.isfinite(x) else None)

    rows_to_insert = df_to_load.to_dict('records')

    if not rows_to_insert:
        return "No data to insert into BigQuery after processing.", 204

    table_ref_str = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"
    errors = client.insert_rows_json(table_ref_str, rows_to_insert)
    
    if not errors:
        return f"Successfully loaded {len(rows_to_insert)} rows into {table_ref_str}", 200
    else:
        return f"Errors encountered while inserting rows into {table_ref_str}: {errors}", 500

@app.route('/', methods=['POST', 'GET']) # Cloud Schedulerからの呼び出しを想定 (POST推奨)
def handler():
    """ HTTPリクエストを受け付けてメインロジックを実行するハンドラ """
    try:
        message, status_code = core_logic()
        print(message) # ログ出力
        return message, status_code
    except ccxt.NetworkError as e:
        error_message = f"CCXT NetworkError: {e}"
        print(error_message)
        return error_message, 503
    except ccxt.ExchangeError as e:
        error_message = f"CCXT ExchangeError: {e} (Detail: {str(e.args)})"
        print(error_message)
        return error_message, 502
    except AttributeError as e:
        error_message = f"AttributeError (e.g., invalid exchange ID '{EXCHANGE_ID}' for ccxt or 'ta' lib): {e}"
        print(error_message)
        return error_message, 400
    except Exception as e:
        error_details = traceback.format_exc()
        error_message = f"An unexpected error occurred: {e}\nTraceback: {error_details}"
        print(error_message)
        return error_message, 500

if __name__ == "__main__":
    # ローカル開発用: Cloud Runではgunicornがこのファイル内の`app`オブジェクトを起動する
    # PORT環境変数はCloud Runによって提供される
    server_port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=server_port)