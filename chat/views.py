import json
import traceback
import ollama
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class OllamaChatView(APIView):
    def post(self, request):
        messages = request.data.get("messages")
        schema = request.data.get("schema")
        options = request.data.get("options", {"temperature": 0})
        if not messages:
            return Response({"error": "messages is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            kwargs = {
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "options": options,
            }
            if schema:
                kwargs["format"] = schema
            response = ollama.chat(**kwargs)
            return Response({"content": response.message.content})
        except Exception as exc:
            traceback.print_exc()
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)