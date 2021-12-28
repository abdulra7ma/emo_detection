# Django imports
from django import forms


class ImageForm(forms.Form):
    """Image form for initializing the main frontend variables"""
    image = forms.ImageField()
    emotion = forms.BooleanField(required=False)
    age = forms.BooleanField(required=False)
    race = forms.BooleanField(required=False)

    # add attributes to the image tag <img>
    image.widget.attrs.update({"class": "form-control", "id": "formFile"})
