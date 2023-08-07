from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from .models import Find

# Create your views here.

def index(request):
    return HttpResponse("Пиздец")

def detail(request, find_id):
    find = get_object_or_404(Find, id=find_id)
    response = f'Ты смотришь находку с видом {find.dragonfly.common_name}'

    return HttpResponse(response)
