from django.urls import path
from . import views

urlpatterns = [
    path("", views.landing, name="landing"),
    path("about/", views.aboutus, name="aboutus"),
    path("contact/", views.contactus, name="contactus"),
]

