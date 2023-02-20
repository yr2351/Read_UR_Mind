# 일기를 입력 받으면
from .keyword_extraction import diary_keyword
from .keyword_augment import augment_keyword
import fasttext, pickle, random
from collections import Counter
from .models import book_final
filter_list = ['그녀', '덕분', '전국', '때문', '그날', '우리','오늘', '당신',\
               '정말', '여러분', '신간', '보편', '독자', '다섯',\
               '따위','수록', '누구', '무엇', '당시', '기존', \
               '한가운데', '국내외','유태', '간주', '무언가', '곳곳','최소한', '그것',\
               '일해', '도래','주요','해당', '종류', '계속', '비롯', '진정', '선사', '관련',\
               '맞이', '절실', '부문', '주제', '까닭', '언제','개별','너머', '선정', '난생처음',\
               '가운데', '이번', '자네', '이후', '특유', '궤멸적', '만큼', '에이','처음',\
               '몇번', '설트란린정', '추가','자꾸', '월요일','화요일','수요일','목요일',\
                '금요일','토요일','일요일','제적']
# 일기에서 추출한 감정
def transform_emotion(emotion):
    if emotion == '행복':
        return ['감동', '공감', '자신감', '위로', '치유', '행복', '힐링']
    elif emotion == '분노':
        return ['고발', '고정관념', '고통', '불평등', '불행', '자존감', '차별', '문제', '혐오', '인간관계']
    elif emotion == '불안':
        return ['걱정', '고민', '괴로움', '불안', '스트레스', '우울']
    elif emotion == '슬픔':
        return ['가슴', '관계', '극복', '깨달음', '상처', '슬픔', '외로움', '절망']

def get_book(keywords, emotions):
    # model = fasttext.load_model("/srv/A-4/word/fasttext-jamo.bin") # 경로 변경!

    # # 키워드가 적은 경우 fasttext를 통해 증강
    # # 수정!
    # if len(keywords) > 0 and len(keywords) <= 2:
    #     final_keywords = augment_keyword(model, keywords, 2)
    # elif len(keywords) > 2 and len(keywords) <= 4:
    #     final_keywords = augment_keyword(model, keywords[:2], 1)
    # else:
    #     final_keywords = keywords
    final_keywords = keywords
    books = {}
    if keywords != []: # 키워드가 존재하는 경우
        # 감정없는 dict
        with open('/srv/A-4/word/total_book.pkl','rb') as f: # 경로 변경!
            keyword_dict = pickle.load(f)
        book_no_emotion_list = []
        book_no_emotion = Counter()
        for key in final_keywords:  
            if keyword_dict[key] != []:
                for book in set(keyword_dict[key]):
                    book_no_emotion.update([book])
        book_no_emotion_rank = book_no_emotion.most_common()
        for key, value in enumerate(book_no_emotion_rank):
            book_no_emotion_list.append(value[0])

        # 감정있는 dict
        with open('/srv/A-4/word/keyword_with_emotion.pkl','rb') as f: # 경로 변경!
            keyword_dict = pickle.load(f)  
        emotion_check = []
        for emotion in emotions: # 각 감정에 대해 처리
            emotion_keyword = transform_emotion(emotion)
            book_with_emotion = []
            for key in emotion_keyword:  
                for book in keyword_dict[key]:
                        book_with_emotion.append(book)
            book_with_emotion = list(set(book_with_emotion)) # 감정에 대한 책 리스트

            final_emotion = []
            for emotion_book in book_with_emotion:           # 동일 감정과 비슷한 키워드까지 있는 경우
                if emotion_book in set(book_no_emotion_list):
                    final_emotion.append(emotion_book)
                    # book_no_emotion_list.remove(emotion_book)
            
            for ij in final_emotion: # 감정별로 책 겹치지 않게
                if ij in emotion_check:
                    final_emotion.remove(ij)

            if len(final_emotion) < 3: # 감정 & 키워드 책이 3개 미만이면
                while len(final_emotion) != 3:
                    num = random.randint(1, len(book_with_emotion)-1)
                    if book_with_emotion not in final_emotion:
                        final_emotion.append(book_with_emotion[num])
            elif len(final_emotion) == 3: # 감정 & 키워드 책이 3개면 통과
                pass
            else:
                random_list = [i for i in range(len(final_emotion))]
                temp = []
                for i in range(3):
                    temp.append(final_emotion[random_list[i]])
                final_emotion = temp
            emotion_check += final_emotion
            books[emotion] = final_emotion
        books['키워드'] = book_no_emotion_list[:9]
        # 쓸모없는 키워드 최종 키워드에서 제외
        temp = []
        for final_key in final_keywords:
            if final_key not in filter_list:
                temp.append(final_key)
        final_keywords = temp 

        if book_no_emotion_list != []: # 키워드에 해당하는 책이 있는 경우       
            return books, final_keywords[:3], 0
        else: # 그럼에도 키워드에 해당하는 책이 전혀 없는 경우
            with open('/srv/A-4/word/random_book.pkl','rb') as f: # 경로 변경!
                random_book_list = pickle.load(f)
            random_list = [i for i in range(len(random_book_list))]
            temp = []
            for i in range(9):
                temp.append(random_book_list[random_list[i]])
            books['키워드'] = temp
            return books, final_keywords[:3], 1

    else:
        with open('/srv/A-4/word/random_book.pkl','rb') as f: # 경로 변경!
            random_book_list = pickle.load(f)
        with open('/srv/A-4/word/keyword_with_emotion.pkl','rb') as f: # 경로 변경!
            keyword_dict = pickle.load(f)  
        for emotion in emotions: # 각 감정에 대해 처리
            emotion_keyword = transform_emotion(emotion)
            book_with_emotion = []
            for key in emotion_keyword:  
                for book in keyword_dict[key]:
                    book_with_emotion.append(book)
            book_with_emotion = list(set(book_with_emotion)) # 감정에 대한 책 리스트
            random_list = [i for i in range(len(book_with_emotion))]
            temp = []
            for i in range(3):
                temp.append(book_with_emotion[random_list[i]])
                try:
                    random_book_list.remove(book_with_emotion[random_list[i]])
                except:
                    pass
            books[emotion] = temp
        random_list = [i for i in range(len(random_book_list))]
        temp = []
        for i in range(9):
            temp.append(random_book_list[random_list[i]])
        books['키워드'] = temp
        return books, 0, 2
    
    

class BookInfo():
    def __init__(self, title, author, url, img, keyword):
        self.title = title
        self.author = author.split(',')[0]
        self.url = url
        bookid = url.split('/')[-1]
        self.img = 'http://image.yes24.com/goods/' + bookid
        self.keyword = keyword.split(',')

def find_book(bookname):
    temp = book_final.objects.get(title=bookname)
    newBook = BookInfo(temp.title, temp.author, temp.link, temp.img_link, temp.keyword)
    return newBook


# a = get_book(diary, ['행복', '불안', '슬픔'])
# print(a)

# a = find_book('아주 작은 시작의 힘(더 이상 미루지 않고 지금 당장 실행하는 기술)')
# print(a)

def emotion_find(book):
    new_book = {}
    for emotion, booklist in book.items():
        bookitems = []
        for bookname in booklist:
            temp = find_book(bookname)
            bookitems.append(temp)
        new_book[emotion] = bookitems
    return new_book

def keywords_find(book):
    new_book = {}
    bookitems = []
    print(book)
    for i in range(len(book)):
        temp = find_book(book[i])
        bookitems.append(temp)
        if (i + 1) % 3 == 0:
            new_book[(i+1) / 3] = bookitems
            bookitems = []
    return new_book        