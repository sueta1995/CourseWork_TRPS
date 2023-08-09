from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('contacts', views.contacts, name='contacts'),
    path('find/<int:find_id>/', views.detail_find, name='detail_find'),
    path('dragonfly/<int:dragonfly_id>/', views.detail_dragonfly, name='detail_dragonfly')
]
