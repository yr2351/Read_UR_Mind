from django.shortcuts import render, redirect
from .models import Diary_content, book_final
from django.http import HttpResponse

# from torchvision import models
# from torchvision import transforms
# model = torch.load('kcelectra_total_model.pt')
# model = models.densenet121(pretrained=True)

from main.templates.main.trch import model, infer_re
from main.main2 import get_book, find_book, emotion_find, keywords_find
from main.keyword_extraction import diary_keyword

model.eval()

# Create your views here.
def index(request):
    return render(request, "main/index.html")

def main(request):
    return render(request, "main/desktop-app.html")


def introduction(request):
    return render(request, 'main/introduction.html')

def submit(request):
    if request.method == 'POST' and request.POST['diary_input'] != '': # 빈 값 x
        content = Diary_content()
        content.para = request.POST['diary_input']
        emotions = infer_re(content.para) 
        keywords = diary_keyword(content.para)
        book, keywords, check = get_book(keywords, emotions)             
        keywordsbooklist = book.pop('키워드')
        emotionbooklist = book
        booklist_K = keywords_find(keywordsbooklist)
        booklist_E = emotion_find(emotionbooklist)
        context = {'para' : content.para, 'emotions':emotions, 'first_emo':emotions[0], 'keywords':keywords, 'check':check, 'keywordsbooklist':booklist_K, 'emotionbooklist':booklist_E,}
        content.save()
        return render(request, 'main/result.html', context)
    else:
        return HttpResponse("<script>alert('빈 값입니다, 다시 입력 해주세요!');location.href='/main';</script>")
        
def team(request):
    return render(request, 'main/team.html')

def result(request):
    return render(request, "main/result.html")

def pr(request):
    # a = book_final.objects.get(id=3)
    a = find_book('아내를 모자로 착각한 남자')
    pep = a
    return render(request, "main/pr.html", {'pep':pep})