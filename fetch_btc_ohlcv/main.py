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

# --- Helper for Environment Variables ---
def get_int_env_var(var_name: str, default_value: int) -> int:
    """ Safely get an integer environment variable. """
    value_str = os.environ.get(var_name, str(default_value))
    try:
        return int(value_str)
    except ValueError:
        print(f"Warning: Invalid value for {var_name}: '{value_str}'. Using default: {default_value}.")
        return default_value

# --- 環境変数 (Cloud Runサービスに設定) ---


GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
BQ_DATASET_ID = os.environ.get('BQ_DATASET_ID')
BQ_TABLE_ID = os.environ.get('BQ_TABLE_ID')
BQ_STATE_TABLE_ID = os.environ.get('BQ_STATE_TABLE_ID') # For storing last processed timestamps

EXCHANGE_ID = os.environ.get('EXCHANGE_ID', 'binance')
SYMBOL = os.environ.get('SYMBOL', 'BTC/USDT')
TIMEFRAME = os.environ.get('TIMEFRAME', '15m')
LIMIT_CANDLES = get_int_env_var('LIMIT_CANDLES', 100)

SMA_PERIOD = get_int_env_var('SMA_PERIOD', 20)
EMA_PERIOD = get_int_env_var('EMA_PERIOD', 50)
MACD_FAST_PERIOD = get_int_env_var('MACD_FAST_PERIOD', 12)
MACD_SLOW_PERIOD = get_int_env_var('MACD_SLOW_PERIOD', 26)
MACD_SIGN_PERIOD = get_int_env_var('MACD_SIGN_PERIOD', 9)

bigquery_client = None # グローバル変数として宣言

def get_bq_client() -> bigquery.Client | None:
    """ BigQueryクライアントを初期化 (必要な場合のみ) """
    global bigquery_client
    if bigquery_client is None:
        if not GCP_PROJECT_ID:
            print("Error: GCP_PROJECT_ID environment variable is not set.")
            return None
        try:
            bigquery_client = bigquery.Client(project=GCP_PROJECT_ID)
        except Exception as e:
            print(f"Error: Failed to initialize BigQuery client: {e}")
            traceback.print_exc()
            return None
    return bigquery_client

def get_last_processed_timestamp(client: bigquery.Client, state_table_ref_str: str, config_key: str) -> int | None:
    """ Fetches the last processed timestamp from the state table. """
    query = f"""
        SELECT last_timestamp_ms
        FROM `{state_table_ref_str}`
        WHERE config_key = @config_key
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("config_key", "STRING", config_key)]
    )
    try:
        query_job = client.query(query, job_config=job_config)
        results = list(query_job) # Waits for the job to complete
        if results:
            return results[0].last_timestamp_ms
    except Exception as e:
        print(f"Error: Error fetching last processed timestamp from {state_table_ref_str} for {config_key}: {e}")
        traceback.print_exc()
    return None

def update_last_processed_timestamp(client: bigquery.Client, state_table_ref_str: str, config_key: str, new_timestamp_ms: int):
    """ Updates the last processed timestamp in the state table using MERGE. """
    merge_query = f"""
        MERGE `{state_table_ref_str}` T
        USING (SELECT @config_key AS config_key, @new_timestamp_ms AS last_timestamp_ms) S
        ON T.config_key = S.config_key
        WHEN MATCHED THEN
            UPDATE SET T.last_timestamp_ms = S.last_timestamp_ms
        WHEN NOT MATCHED THEN
            INSERT (config_key, last_timestamp_ms) VALUES (S.config_key, S.last_timestamp_ms)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("config_key", "STRING", config_key),
            bigquery.ScalarQueryParameter("new_timestamp_ms", "INT64", new_timestamp_ms),
        ]
    )
    try:
        query_job = client.query(merge_query, job_config=job_config)
        query_job.result()  # Wait for the job to complete
        print(f"Info: Successfully updated last_timestamp_ms to {new_timestamp_ms} for {config_key} in {state_table_ref_str}")
    except Exception as e:
        print(f"Error: Error updating last processed timestamp in {state_table_ref_str} for {config_key}: {e}")
        traceback.print_exc()


def core_logic() -> tuple[str, int]:
    """ 主要なデータ取得・計算・BigQuery保存ロジック """
    client = get_bq_client()
    if not client:
        return "BigQuery client could not be initialized due to missing GCP_PROJECT_ID.", 500
    if not BQ_DATASET_ID or not BQ_TABLE_ID or not BQ_STATE_TABLE_ID:
        return "BQ_DATASET_ID, BQ_TABLE_ID, or BQ_STATE_TABLE_ID environment variables are not set.", 500

    config_key = f"{EXCHANGE_ID}_{SYMBOL}_{TIMEFRAME}"
    state_table_ref_str = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_STATE_TABLE_ID}"
    main_table_ref_str = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"

    print(f"Info: Logic triggered for {config_key}.")
    print(f"Info: Calculating SMA({SMA_PERIOD}), EMA({EMA_PERIOD}), MACD({MACD_FAST_PERIOD},{MACD_SLOW_PERIOD},{MACD_SIGN_PERIOD}).")
    print(f"Info: Target BigQuery table: {main_table_ref_str}")
    print(f"Info: State table for timestamps: {state_table_ref_str}")

    last_ts_ms = get_last_processed_timestamp(client, state_table_ref_str, config_key)
    fetch_params = {'limit': LIMIT_CANDLES}
    if last_ts_ms is not None:
        fetch_params['since'] = last_ts_ms + 1 # Fetch candles *after* the last one
        print(f"Info: Fetching data since timestamp: {last_ts_ms + 1}")
    else:
        print("Info: No last processed timestamp found. Fetching latest candles.")

    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({'enableRateLimit': True})
    
    ohlcv_raw = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, params=fetch_params)
    if not ohlcv_raw:
        print(f"Info: No new OHLCV data returned for {config_key} (since={fetch_params.get('since')}).")
        return f"No new OHLCV data returned for {config_key}.", 200 # 204 might be misinterpreted as no content to process by scheduler

    df = pd.DataFrame(ohlcv_raw, columns=['timestamp_ms', 'open', 'high', 'low', 'close', 'volume'])
    if df.empty:
        print(f"Info: Fetched OHLCV data is empty for {config_key} after DataFrame conversion.")
        return f"Fetched OHLCV data is empty for {config_key}.", 200
        
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

    errors = client.insert_rows_json(main_table_ref_str, rows_to_insert)
    
    if not errors:
        latest_inserted_timestamp_ms = int(df['timestamp_ms'].max()) # Ensure it's Python int
        # Update last processed timestamp only if new data was actually later than previously stored
        if last_ts_ms is None or latest_inserted_timestamp_ms > last_ts_ms:
            update_last_processed_timestamp(client, state_table_ref_str, config_key, latest_inserted_timestamp_ms)
        else:
            print(f"Info: No newer data processed. Last known timestamp {last_ts_ms}, latest in batch {latest_inserted_timestamp_ms}.")
        return f"Successfully loaded {len(rows_to_insert)} rows into {main_table_ref_str}", 200
    else:
        print(f"Error: Errors encountered while inserting rows into {main_table_ref_str}: {errors}")
        return f"Errors encountered while inserting rows into {main_table_ref_str}: {errors}", 500

@app.route('/', methods=['POST', 'GET']) # Cloud Schedulerからの呼び出しを想定 (POST推奨)
def handler():
    """ HTTPリクエストを受け付けてメインロジックを実行するハンドラ """
    try:
        print(f"Info: Handler invoked by {flask_request.method} request from {flask_request.remote_addr}")
        message, status_code = core_logic()
        print(f"Info: Core logic finished. Status: {status_code}, Message: {message}")
        return message, status_code
    except ccxt.NetworkError as e:
        error_message = f"CCXT NetworkError: {e}"        
        print(f"Error: {error_message}")
        traceback.print_exc()
        return error_message, 503
    except ccxt.ExchangeError as e:
        error_message = f"CCXT ExchangeError: {e} (Detail: {str(e.args)})"
        print(f"Error: {error_message}")
        traceback.print_exc()
        return error_message, 502
    except AttributeError as e:
        error_message = f"AttributeError (e.g., invalid exchange ID '{EXCHANGE_ID}' for ccxt or 'ta' lib): {e}"
        print(f"Error: {error_message}")
        traceback.print_exc()
        return error_message, 400
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"Error: {error_message}")
        traceback.print_exc()
        return error_message, 500

if __name__ == "__main__":
    # ローカル開発用: Cloud Runではgunicornがこのファイル内の`app`オブジェクトを起動する
    # PORT環境変数はCloud Runによって提供される
    server_port = get_int_env_var("PORT", 8080)
    app.run(debug=True, host="0.0.0.0", port=server_port)