# crypto-indicator-service-docker/Dockerfile

# 1. ベースイメージを選択 (PythonのバージョンをCloud Functionsと合わせるなど)
FROM python:3.11-slim

# 2. 環境変数を設定
ENV PYTHONUNBUFFERED True   # Pythonのprintなどがすぐに出力されるようにする
ENV APP_HOME /app           # アプリケーションを配置するディレクトリ
WORKDIR $APP_HOME

# (オプション) 必要なOSレベルのパッケージがあればここでインストール
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# 3. 依存関係ファイルをコピーし、ライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションのソースコードをコピー
COPY . .

# 5. Cloud Runがリッスンするポートを指定 (Cloud Runから環境変数PORTで渡される)
# デフォルトは8080だが、Cloud Runが設定する$PORTを使うのが一般的
ENV PORT 8080
EXPOSE $PORT

# 6. コンテナ起動時にgunicorn Webサーバーを実行
# main:app は、main.pyファイル内のFlaskインスタンスである`app`オブジェクトを指定
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app