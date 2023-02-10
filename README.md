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

## 各ファイルの簡易説明
### mbti_util.py
MBTI 関連のユーティリティクラスであり、MBTIの全てのタイプやタイプを日本語表記に変換するといった機能を備えている。

### denoicer.py
`neologdn`、`demoji` といったパッケージを使用し、ツイートを正規化する。

### mecab_wakati.py
文を入力すると分かち書きされた単語リストを返す。空白文字はリストの中に入らないように除去している。

### multi_label_classification_model_wrapper.py
`SimpleTransformer` の `MultiLabelClassificationModel` で HuggingFace のモデルを指定した時にトークナイザー関連でエラーがでることがあった。`roberta_ja` や `twhinbert` で、トークナイザーを指定できるようにした。

### SentenceBertJapanese.py
テキストに `Sentence BERT` のベクトルを付与するためのクラス。[【日本語モデル付き】2020年に自然言語処理をする人にお勧めしたい文ベクトルモデル](https://qiita.com/sonoisa/items/1df94d0a98cd4f209051) を参照。

### anlysis_emotion_tweets.ipynb
作成した感情推定モデルから収集したツイートに感情ラベルを付与し、コロナ禍前後で感情ごとの内訳を見ている
。

### anlysis_emotion_tweets_joy_anticipation.ipynb
コロナ禍前の Joy, Anticipation のツイートとコロナ禍後の Joy, Anticipation のツイートを抜き出し、そこから特徴語を抽出する。`WordCloud` あたりの実装が微妙な気がするのでこのファイルを参考にしないほうがいいと思う。

### count.py
ツイートに `Sentence BERT` のベクトルを付与し、コロナ禍前後のユークリッド距離を測定・図示している。

### create_emotion_model.ipynb
感情推定モデルの学習・評価、この `README` 最初に使用方法を記述している。

### estimate_emotion.ipynb
未知データの推定、この `README` 最初に使用方法を記述している。

### extract_feature_words_by_covid_tweets.ipynb
コロナ|武漢|Covid|COVID|ワクチン|パンデミック|マスク|自粛|クラスター|蔓延防止|マンボウ|まん延防止|給付金の文字列を含んだ行をデータフレームから抜き出し、特徴語を分析している。最終的に利用しなかった。

### tf-idf.ipynb
性格タイプのコロナ禍前後の特徴語を抽出している。