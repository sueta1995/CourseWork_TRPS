from django.shortcuts import render
from django.http import HttpResponse
from main.models import Find

# Create your views here.

def index(request):
    return HttpResponse("Пиздец")

def detail(request, find_id):
    find = Find.objects.get(id=find_id)
    response = f'Ты смотришь находку с видом {find.dragonfly.common_name}'

    return HttpResponse(response)
