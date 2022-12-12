import re
import neologdn
import demoji


class Denoicer:
    def __init__(self) -> None:
        self.stop_words = []
        with open('./database/StopWordsJapanese.txt', 'r', encoding='utf8') as f:
            for l in f.readlines():
                stop = l.strip()
                if stop != '':
                    self.stop_words.append(stop)

    def normalize_text(self, text) -> str:
        # url除去
        doc = re.sub(r"http\S+", "", text)
        # メンション除去
        doc = re.sub(r"@(\w+) ", "", doc)
        # 絵文字を消す
        doc = demoji.replace(string=doc, repl="")
        # リツイートを消す
        doc = re.sub(r"(^RT.*)", "", doc, flags=re.MULTILINE | re.DOTALL)
        doc = neologdn.normalize(doc)
        # 数字を０に
        doc = re.sub(r'\d+', '0', doc)
        # 大文字・小文字変換
        doc = doc.lower()
        return doc
