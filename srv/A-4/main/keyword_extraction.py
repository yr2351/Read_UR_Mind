#pip install sentence_transformers
#pip install konlpy

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from collections import Counter 
def diary_keyword(diary):
    def input_doc(doc, range_1, range_2, top_n):
        try:
            okt = Okt()
            tokenized_doc = okt.pos(doc)
            tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

            n_gram_range = (range_1, range_2)
            count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
            candidates = count.get_feature_names_out()

            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('jhgan/ko-sroberta-nli') # 좋다
            doc_embedding = model.encode([doc])
            candidate_embeddings = model.encode(candidates)

            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

            return keywords
        except:
            return []
    diary_key = []
    diary_final = []
    if len(diary) >= 500:
        keyword = input_doc(diary, 2, 3, 3) # 수정!
        for i in keyword:
            tmp = i.split()
            for j in tmp:
                diary_key.append(j)
    else:
        keyword = input_doc(diary, 2, 3, 5) # 수정!
        for i in keyword:
            tmp = i.split()
            for j in tmp:
                diary_key.append(j)
    diary_counter = Counter(diary_key).most_common() # 수정!
    for key, value in enumerate(diary_counter):
        diary_final.append(value[0])
    return diary_final
