import fasttext
import hgtk
from konlpy.tag import Okt, Hannanum, Mecab
hannanum = Hannanum()
def word_to_jamo(token):
    def to_special_token(jamo):
        if not jamo:
            return '-'
        else:
            return jamo
    decomposed_token = ''
    for char in token:
        try:
            cho, jung, jong = hgtk.letter.decompose(char)
            cho = to_special_token(cho)
            jung = to_special_token(jung)
            jong = to_special_token(jong)
            decomposed_token = decomposed_token + cho + jung + jong
        except Exception as exception:
            if type(exception).__name__ == 'NotHangulException':
                decomposed_token += char
    return decomposed_token

def tokenize_by_jamo(s):
    return [word_to_jamo(token) for token in hannanum.morphs(s)]

def jamo_to_word(jamo_sequence):
    tokenized_jamo = []
    index = 0

    while index < len(jamo_sequence):
        if not hgtk.checker.is_hangul(jamo_sequence[index]):
            tokenized_jamo.append(jamo_sequence[index])
            index = index + 1

        else:
            tokenized_jamo.append(jamo_sequence[index:index + 3])
            index = index + 3
    word = ''
    try:
        for jamo in tokenized_jamo:
            if len(jamo) == 3:
                if jamo[2] == "-":
                    word = word + hgtk.letter.compose(jamo[0], jamo[1])
                else:
                    word = word + hgtk.letter.compose(jamo[0], jamo[1], jamo[2])
            else:
                word = word + jamo

    except Exception as exception:
        if type(exception).__name__ == 'NotHangulException':
            return jamo_sequence

    return word

def transform(word_sequence):
    return [(jamo_to_word(word), similarity) for (similarity, word) in word_sequence]

def augment_keyword(model, keywords, aug_num):
    final_keyword = []
    for keyword in keywords:
        added = transform(model.get_nearest_neighbors(word_to_jamo(keyword), k=aug_num))
        for i in added:
            final_keyword.append(i[0])
    return keywords+final_keyword