import MeCab

class MecabWakati:
    def __init__(self) -> None:
        self.mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        self.mecab.parse('')  # バグ対処

    def wakati_sentence(self, sentence: str) -> list:
        words = []
        node = self.mecab.parseToNode(sentence.strip())
        while node:
            if node.feature.split(",")[6] == '*':
                if node.surface not in [" ","　","\t"]:
                    words.append(str(node.surface))
            else:
                words.append(str(node.feature.split(",")[6]))
            node = node.next
        return [w for w in words if w != '']