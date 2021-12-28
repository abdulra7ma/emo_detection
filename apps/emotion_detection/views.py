# Python imports
from base64 import b64encode
from io import BytesIO
from os.path import join
from posixpath import abspath

# Django imports
from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import FormView, TemplateView

# external imports
import PIL.Image as Image

# app imports
from .forms import ImageForm
from .singlemotiondetector import SingleMotionDetector


class MainView(TemplateView):
    """ Main view for displaying the main page"""
    template_name = "index.html"

    def get(self, request, *args, **kwargs):
        print(dir(self.request))
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["url_class"] = "main"
        return context


class DetectorView(FormView):
    """ Form view to retrieve the images from
    the form and save to the local directory,
    and runs the SingleEmotionDetector class
    and return output to the frontend.
    """
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

        # relative path for the image that got to be process
        image_path = "media/recieved_imgs/" + image_name
        image_media_path = "static/media/recieved_imgs/" + image_name

        image = Image.open(BytesIO(file_binary))

        # save to the main media file
        image.save(image_path)

        # save to the static media file
        image.save(image_media_path)

        # django context variables for fronted usage
        context = self.get_context_data()

        if emotion:
            driver = SingleMotionDetector(image_path, "emotion")
        elif age:
            driver = SingleMotionDetector(image_path, "age")
        elif race:
            driver = SingleMotionDetector(image_path, "race")
        else:
            context["image"] = image_path
            return render(self.request, self.template_name, context)

        driver_output = driver()

        context["image"] = "/".join(driver_output.split("/")[1:])

        return render(self.request, self.template_name, context)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["url_class"] = "detecator"
        return context

    def get_success_url(self) -> str:
        return reverse("emotion_detection:main")
