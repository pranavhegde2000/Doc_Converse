from django.urls import path
from .views import DocumentUploadView, QueryView

urlpatterns = [
    path('upload/', DocumentUploadView.as_view(), name='upload'),
    path('query/', QueryView.as_view(), name='query'),
]