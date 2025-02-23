# Generated by Django 5.1.2 on 2024-12-07 08:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("plantapp", "0002_remove_disease_medicine"),
    ]

    operations = [
        migrations.AddField(
            model_name="disease",
            name="amount",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="disease",
            name="area",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="disease",
            name="protect",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="disease",
            name="severinty",
            field=models.TextField(blank=True, null=True),
        ),
    ]
