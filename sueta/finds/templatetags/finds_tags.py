from django import template
from finds.models import *

import os


register = template.Library()

@register.simple_tag()
def get_finds():
    return Find.objects.all()

@register.simple_tag()
def get_dragonflies():
    return Dragonfly.objects.all()

@register.simple_tag()
def get_env(key):
    return os.environ.get(key)
