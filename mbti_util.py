class MbtiUtil():
    def __init__(self) -> None:
        
        self.m_types = ["ESFP", "ISTP", "ENTP", "ESFJ", "INFP", "ENFP", "ISFP", "ISFJ", "INFJ", "ENFJ", "INTP", "ESTP", "INTJ", "ENTJ", "ESTJ", "ISTJ"]
        self.type_dic: dict = {
            "INTJ": "建築家", "INTP": "論理学者", "ENTJ": "指揮官", "ENTP": "討論者",
            "INFJ": "提唱者", "INFP": "仲介者", "ENFJ": "主人公", "ENFP": "広報運動家",
            "ISTJ": "管理者", "ISFJ": "擁護者", "ESTJ": "幹部", "ESFJ": "領事官",
            "ISTP": "巨匠", "ISFP": "冒険家", "ESTP": "起業家", "ESFP": "エンターテイナー",
        }

        self.type_num_dic: dict = {
            "INTJ": 0, "INTP": 1, "ENTJ": 2, "ENTP": 3,
            "INFJ": 4, "INFP": 5, "ENFJ": 6, "ENFP": 7,
            "ISTJ": 8, "ISFJ": 9, "ESTJ": 10, "ESFJ": 11,
            "ISTP": 12, "ISFP": 13, "ESTP": 14, "ESFP": 15,
        }

    def ToEngType(self, type_ja: str) -> str:
        for key, value in self.type_dic.items():
            if value == type_ja:
                return key
        return ""

    def ToJaType(self, type_en: str) -> str:
        if type_en in self.type_dic.keys():
            return self.type_dic[type_en]
        else:
            ""
