from django.urls import path
from drowiness_app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('start_tracking/', views.start_tracking, name='start_tracking'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
