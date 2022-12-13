# research-covid-19-mbti
## 感情推定モデルの作成および推定方法
### 前準備
1. このリポジトリを pull もしくはダウンロードする
2. 別途、共有した `database.zip` と `model.zip` を解凍する
3. `research-covid-19-mbti` フォルダ直下に解凍した `database` フォルダを置く
4. `research-covid-19-mbti/model` フォルダ直下に解凍した `model/trained` フォルダを置く
5. `python3 -m venv venv` で仮想環境を作成
6. `. venv/bin/activate` コマンドで仮想環境をアクティベート
7. `pip3 install -r requirements.txt` で必要な Python パーケージをインストール

### 感情推定モデルの学習・評価
- `create_emotion_model.ipynb` を実行
- `train` メソッドの `model_type` と `model_name` を設定し実行すると、学習が始まり、`./model` 直下にモデルが作成される
- `evaluate_emotion_model` を実行

### 未知データの推定
- `estimate_emotion.ipynb` を実行
- `estimate_tweets_and_output` メソッドを実行すると、`./database/emotion/emotion_result` 直下に結果の TSV が作成される

### 推定された感情のカウント
- `anlysis_emotion_tweets.ipynb` を実行
- `emotion_count_before_after_by_mbti` メソッドを実行すると MBTI 別に感情をカウントする
- `emotion_count_before_after` メソッドを実行すると MBTI に関係なく感情をカウントする