# Generated by Django 4.2.3 on 2023-08-17 16:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('finds', '0004_alter_find_latitude_alter_find_longitude'),
    ]

    operations = [
        migrations.AlterField(
            model_name='find',
            name='photo',
            field=models.ImageField(blank=True, null=True, upload_to='photos/'),
        ),
    ]
