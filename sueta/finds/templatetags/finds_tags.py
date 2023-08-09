from django import template
from finds.models import *


register = template.Library()

@register.simple_tag()
def get_finds():
    return Find.objects.all()

@register.simple_tag()
def get_dragonflies():
    return Dragonfly.objects.all()
