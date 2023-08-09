from django.db import models
from django.urls import reverse


class Dragonfly(models.Model):
    specific_name = models.CharField(max_length=32)
    common_name = models.CharField(max_length=32)
    wikipedia_url = models.URLField(max_length=255)

    def __str__(self):
        return self.specific_name
    
    def get_absolute_url(self):
        return reverse("detail_dragonfly", kwargs={"dragonfly_id": self.pk})


class Find(models.Model):
    longitude = models.FloatField()
    latitude = models.FloatField()
    comment = models.TextField(blank=True)
    time_create = models.DateTimeField(auto_now_add=True)
    time_update = models.DateTimeField(auto_now=True)
    photo = models.ImageField(upload_to="photos/")
    dragonfly = models.ForeignKey(Dragonfly, on_delete=models.CASCADE)

    def __str__(self):
        return self.dragonfly.specific_name
    
    def get_absolute_url(self):
        return reverse("detail_find", kwargs={"pk": self.pk})
    
