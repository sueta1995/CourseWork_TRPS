from django import forms
from .models import *


class AddFindForm(forms.Form):
    comment = forms.CharField(label='Комментарий', widget=forms.Textarea)
    photo = forms.ImageField(label='Фотография')
    latitude = forms.FloatField(label='', widget=forms.HiddenInput)
    longitude = forms.FloatField(label='', widget=forms.HiddenInput)

    latitude.widget.attrs.update({'id': 'latitude-input'})
    longitude.widget.attrs.update({'id': 'longitude-input'})
    comment.widget.attrs.update({'class': 'form-control'})
    photo.widget.attrs.update({'class': 'form-control'})
