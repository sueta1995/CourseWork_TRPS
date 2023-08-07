from django.db import models

from datetime import datetime


class Dragonfly(models.Model):
    specific_name = models.CharField(max_length=32)
    common_name = models.CharField(max_length=32)
    wikipedia_url = models.URLField(max_length=256)
    photos_url = models.URLField(max_length=256)

    def __str__(self):
        return self.specific_name


class Find(models.Model):
    longitude = models.FloatField()
    latitude = models.FloatField()
    comment = models.CharField(max_length=256)
    publication_date = models.DateTimeField('date published', default=datetime.now)
    photo_url = models.URLField(max_length=256)
    dragonfly = models.ForeignKey(Dragonfly, on_delete=models.CASCADE)

    def __str__(self):
        return self.dragonfly.specific_name
