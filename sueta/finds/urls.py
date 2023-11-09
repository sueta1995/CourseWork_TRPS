from django.urls import path

from . import views

urlpatterns = [
    path('', views.DragonfliesList.as_view(), name='index'),
    path('contacts', views.contacts, name='contacts'),
    path('find/<int:pk>/', views.FindDetailView.as_view(), name='detail'),
    path('dragonfly/<int:dragonfly_id>/', views.FindsList.as_view(), name='finds_list'),
    path('dragonfly/', views.FindsList.as_view(), name='finds_list_all'),
    path('finds_upload/', views.finds_upload, name='finds_upload'),
    path('finds_update/', views.finds_update, name='finds_update'),
    path('finds_delete/', views.finds_delete, name='finds_delete'),
]
