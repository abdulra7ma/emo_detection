# Django imports
from django.urls import re_path

# app imports
from apps.emotion_detection import views

app_name = "emotion_detection"


urlpatterns = [
    re_path(r"^$", views.MainView.as_view(), name="main"),
    re_path(r"^detecator/$", views.DetectorView.as_view(), name="detecator"),
]
