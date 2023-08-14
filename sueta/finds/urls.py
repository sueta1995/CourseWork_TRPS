from django.urls import path

from . import views

urlpatterns = [
    path('', views.DragonfliesList.as_view(), name='index'),
    path('contacts', views.contacts, name='contacts'),
    path('find/<int:find_id>/', views.detail, name='detail'),
    path('dragonfly/<int:dragonfly_id>/', views.FindsList.as_view(), name='finds_list'),
    path('dragonfly/', views.FindsList.as_view(), name='finds_list_all'),
]
