# Django imports
from django import forms

# external imports
from tensorflow.keras import models


class ImageForm(forms.Form):
    image = forms.ImageField()
    emotion = forms.BooleanField(required=False)
    age = forms.BooleanField(required=False)
    race = forms.BooleanField(required=False)

    CHOICES = (("E", "Emotion"), ("A", "Age"), ("G", "Gender"))

    image_choices = forms.RadioSelect(choices=CHOICES)

    image.widget.attrs.update({"class": "form-control", "id": "formFile"})
