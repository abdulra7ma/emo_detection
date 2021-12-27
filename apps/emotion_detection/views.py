# Python imports
from base64 import b64encode
from io import BytesIO
from os.path import join
from posixpath import abspath
import re

# Django imports
from django import template
from django.http import HttpResponse
from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import FormView, TemplateView, View
from django.views.generic.edit import FormMixin

# external imports
import PIL.Image as Image

# app imports
from .forms import ImageForm
from .singlemotiondetector import SingleMotionDetector


class MainView(TemplateView):
    template_name = "index.html"

    def get(self, request, *args, **kwargs):
        print(dir(self.request))
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["url_class"] = "main"
        return context


class DetectorView(FormView):
    form_class = ImageForm
    template_name = "game2.html"

    def form_valid(self, form) -> HttpResponse:
        cleaned_data = form.cleaned_data
        # frontend checkboxs boolean values
        age, emotion, race = (
            cleaned_data["age"],
            cleaned_data["emotion"],
            cleaned_data["race"],
        )
        image_name, img_file = cleaned_data["image"].name, cleaned_data["image"].file
        # read the binary format of the file and save it to 'media/recieved_imgs/' directory
        file_binary = BytesIO.read(img_file)
        image = Image.open(BytesIO(file_binary))
        image.save("media/recieved_imgs/" + image_name)
        # relative path for the image that got to be process
        img_to_process = "media/recieved_imgs/" + image_name

        if emotion:
            driver = SingleMotionDetector(img_to_process, True, False, False, False)
            driver_output = driver()

            context = self.get_context_data()
            context["image"] = "/".join(driver_output.split("/")[1:])

            return render(self.request, self.template_name, context)
        else:
            return HttpResponseRedirect(self.get_success_url())

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["url_class"] = "detecator"
        return context

    def get_success_url(self) -> str:
        return reverse("emotion_detection:main")
