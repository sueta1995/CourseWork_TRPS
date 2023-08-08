from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('contacts', views.contacts, name='contacts'),
    path('<int:find_id>/', views.detail, name='detail'),
]
