# Generated by Django 5.1.5 on 2025-01-30 03:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Dashboard', '0002_alter_data_tumor_img'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='data',
            options={'ordering': ['-date']},
        ),
    ]
