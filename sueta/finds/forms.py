from django import forms
from .models import *


class AddFindForm(forms.Form):
    comment = forms.CharField(label='Комментарий', widget=forms.Textarea)
    photo = forms.ImageField(label='Фотография')

    comment.widget.attrs.update({'class': 'form-control'})
    photo.widget.attrs.update({'class': 'form-control'})
