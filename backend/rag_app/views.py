from django.shortcuts import render
from langchain.evaluation.qa.eval_prompt import context_template
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Document
from .utils import RAGProcessor
import os

class DocumentUploadView(APIView):
    def post(self, request):
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'},
                             status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']
        file_path = f'uploads/{file.name}'
        # file object will contain attributes such as name, size, read()

        # Save file
        os.makedirs('uploads', exist_ok=True)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Process document
        rag_processor = RAGProcessor()
        vector_ids = rag_processor.process_document(file_path)

        # Create document instance, save document info
        document = Document.objects.create(
            title=file.name,
            file_path=file_path,
            vector_ids=vector_ids
        )

        return Response({'message': 'Document processed successfully'},
                        status=status.HTTP_201_CREATED)

class QueryView(APIView):
    def post(self, request):
        query = request.data.get('query')
        if not query:
            return Response({'error': 'No query provided'},
                            status=status.HTTP_400_BAD_REQUEST)
        rag_processor = RAGProcessor()
        context = rag_processor.query_documents(query)
        answer = rag_processor.generate_answer(query, context)

        return Response({
            'context': context,
            'answer': answer
        })