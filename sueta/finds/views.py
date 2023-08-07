from django.shortcuts import render, get_object_or_404
from django.template import loader
from django.http import HttpResponse
from .models import Find

# Create your views here.

def index(request):
    finds = Find.objects.order_by('-publication_date')[:5]
    template = loader.get_template('finds/index.html')
    context = {
        'finds': finds,
    }

    return HttpResponse(template.render(context, request))

def detail(request, find_id):
    find = get_object_or_404(Find, id=find_id)
    template = loader.get_template('finds/detail.html')
    context = {
        'find': find,
    }

    return HttpResponse(template.render(context, request))

def contacts(request):
    template = loader.get_template('finds/contacts.html')

    return HttpResponse(template.render(None, request))
