from typing import Any, Dict
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.forms import UserCreationForm
from django.template import loader
from django.http import HttpResponse, HttpResponseNotFound
from django.views.generic import CreateView, ListView
from django.urls import reverse_lazy
from django.contrib.auth import logout

from .models import *


class DragonfliesList(ListView):
    model = Dragonfly
    template_name = 'finds/dragonflies_list.html'
    
    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context['title'] = 'Стрекозки'

        return context
    

class FindsList(ListView):
    model = Find
    template_name = 'finds/finds_list.html'
    context_object_name = 'finds'

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)

        if 'dragonfly_id' in self.kwargs:
            d = get_object_or_404(Dragonfly, pk=self.kwargs['dragonfly_id'])

            context['title'] = d.common_name
            context['specie'] = d.specific_name
        else:
            context['title'] = 'Последние находки'
            context['specie'] = None

        return context

    def get_queryset(self):
        return Find.objects.filter(dragonfly__id=self.kwargs['dragonfly_id']) if 'dragonfly_id' in self.kwargs else Find.objects.all
    


def detail(request, find_id):
    find = get_object_or_404(Find, pk=find_id)
    template = loader.get_template('finds/finds_list.html')
    context = {
        'find': find,
        'title': find.common_name
    }

    return HttpResponse(template.render(context, request))


def contacts(request):
    template = loader.get_template('finds/contacts.html')

    return HttpResponse(template.render({ 'title': 'Контакты' }, request))


def logout(request):
    logout(request)

    return redirect('index')


def pageNotFound(request, exception):
    return HttpResponseNotFound('Пиздец, не найдено')
