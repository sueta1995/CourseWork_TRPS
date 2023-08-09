from django.shortcuts import render, get_object_or_404
from django.contrib.auth.forms import UserCreationForm
from django.template import loader
from django.http import HttpResponse, HttpResponseNotFound
from django.views.generic import CreateView
from django.urls import reverse_lazy


from .models import *

# Create your views here.

def index(request):
    finds = Find.objects.order_by('-time_create')[:5]
    template = loader.get_template('finds/index.html')
    context = {
        'finds': finds,
        'title': 'Стрекозки'
    }

    return HttpResponse(template.render(context, request))


def detail_find(request, find_id):
    find = get_object_or_404(Find, id=find_id)
    template = loader.get_template('finds/detail.html')
    context = {
        'find': find,
        'title': find.common_name
    }

    return HttpResponse(template.render(context, request))


def detail_dragonfly(request, dragonfly_id):
    return HttpResponse(dragonfly_id, request)


def contacts(request):
    template = loader.get_template('finds/contacts.html')

    return HttpResponse(template.render({ 'title': 'Контакты' }, request))


def pageNotFound(request, exception):
    return HttpResponseNotFound('Пиздец, не найдено')
