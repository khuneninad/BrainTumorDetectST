from django.urls import path
from .import views

urlpatterns = [
    path('', views.index, name='dashboard-index'),
    path('predictions/', views.predictions, name='dashboard-predictions'),
    path('result/<int:pk>/', views.result, name='dashboard-result'),
]