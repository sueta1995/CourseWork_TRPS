from django.db import models
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.conf import settings

import numpy as np
import cv2


class Dragonfly(models.Model):
    specific_name = models.CharField(max_length=32)
    common_name = models.CharField(max_length=32)
    wikipedia_url = models.URLField(max_length=255)

    def __str__(self):
        return self.specific_name
    
    def get_absolute_url(self):
        return reverse("finds_list", kwargs={"dragonfly_id": self.pk})


class Find(models.Model):
    longitude = models.FloatField(blank=True, null=True)
    latitude = models.FloatField(blank=True, null=True)
    comment = models.TextField(blank=True)
    time_create = models.DateTimeField(auto_now_add=True)
    time_update = models.DateTimeField(auto_now=True)
    photo = models.ImageField(upload_to="photos/")
    confirmed = models.BooleanField(default=False)
    dragonfly = models.ForeignKey(Dragonfly, on_delete=models.CASCADE, blank=True, null=True)
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, blank=True, null=True)

    def __str__(self):
        return self.dragonfly.specific_name
    
    def get_absolute_url(self):
        return reverse("detail", kwargs={"pk": self.pk})
    
    def predict_specie(self):
        img = cv2.imread(self.photo.path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = np.array(img)
        x = np.expand_dims(x, axis=0)

        features = settings.MODEL.predict(x)
        classes = np.argmax(features, axis=1)
        category = settings.CATEGORIES[classes]

        return category[0]
    
