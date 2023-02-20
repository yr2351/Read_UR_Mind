from django.db import models

# Create your models here.

class Diary_content(models.Model):
    para = models.CharField(max_length = 1000, blank=False)

class book_final(models.Model):
    title = models.CharField(max_length=100, null=False)
    author = models.CharField(max_length=100, null=True)
    link = models.CharField(max_length=100, null=False)
    img_link = models.CharField(max_length=100, null=False, default='')
    keyword = models.CharField(max_length=100, null=False)
