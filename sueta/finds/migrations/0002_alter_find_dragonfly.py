# Generated by Django 4.2.3 on 2023-08-06 13:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('finds', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='find',
            name='dragonfly',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='finds.dragonfly'),
        ),
    ]
