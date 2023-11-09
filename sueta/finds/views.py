from typing import Any, Dict
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.forms import UserCreationForm
from django.template import loader
from django.http import HttpResponse, HttpResponseNotFound
from django.views.generic import ListView
from django.views.generic.detail import DetailView
from django.urls import reverse_lazy
from django.contrib.auth import logout
from django.conf import settings

import os

from .models import *
from .forms import *


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
            form = AddFindForm()

            context['title'] = 'Последние находки'
            context['specie'] = None
            context['form'] = form

        return context

    def get_queryset(self):
        return Find.objects.filter(dragonfly__id=self.kwargs['dragonfly_id']) if 'dragonfly_id' in self.kwargs else Find.objects.all


class FindDetailView(DetailView):
    model = Find

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        find = get_object_or_404(Find, pk=self.kwargs['pk'])
        context['title'] = find.dragonfly.common_name
        context['find'] = find

        return context


def contacts(request):
    template = loader.get_template('finds/contacts.html')

    return HttpResponse(template.render({ 'title': 'Контакты' }, request))


def logout(request):
    logout(request)

    return redirect('index')


def finds_upload(request):
    if request.method == 'POST':
        form = AddFindForm(request.POST, request.FILES)

        if form.is_valid():
            Find.objects.filter(confirmed=False, user_id=request.user.id).delete()

            find = Find.objects.create(**form.cleaned_data)

            find.dragonfly = Dragonfly.objects.filter(specific_name=find.predict_specie())[0]
            find.user = request.user
            find.save()                

            template = loader.get_template('finds/finds_upload.html')
            context = {
                'n': range(1, 5),
                'find': find,
                'title': 'Подтверждение находки'
            }

            return HttpResponse(template.render(context, request))
        else:
            return redirect("finds_list_all")
    else:
        return redirect('index')


def finds_update(request):
    if request.method == 'POST':
        find = Find.objects.filter(pk=request.POST['find_id'])[0]
        find.confirmed = True
        find.save()

    return redirect('finds_list_all')


def finds_delete(request):
    if request.method == 'POST':
        find = Find.objects.filter(pk=request.POST['find_id'])
        find.delete()

    return redirect('finds_list_all')


def pageNotFound(request, exception):
    return HttpResponseNotFound('Страница не найдена')
