from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=255)
    file_path = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    vector_ids = models.JSONField(default=list)
