import json
import time
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

        print(f"🚀 Received prompt: {messages}")

        if not messages:
            return Response(
                {"error": "messages is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            print("🧠 Estimating tokens ...")

            input_chars = sum(len(m.get("content", "")) for m in messages)

            estimation_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Estimate how many TOKENS the assistant will generate.\n"
                        "Return JSON: {\"tokens\": number}\n"
                        "Be conservative (+20%)."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "messages": messages,
                        "schema": schema,
                        "input_chars": input_chars
                    }),
                },
            ]

            # estimate_resp = ollama.chat(
            #     model=settings.OLLAMA_MODEL,
            #     messages=estimation_prompt,
            #     format={
            #         "type": "object",
            #         "properties": {
            #             "tokens": {"type": "number"},
            #         },
            #         "required": ["tokens"],
            #     },
            #     options={"temperature": 0},
            # )

            # try:
                # estimate_json = json.loads(estimate_resp["message"]["content"])
                # estimated_tokens = int(estimate_json.get("tokens", 1000))
            # except Exception:
            #     print("⚠️ Estimator failed, using fallback")
            estimated_tokens = 1000

            estimated_tokens = max(300, min(estimated_tokens, 20000))

            print(f"📊 Estimated tokens: {estimated_tokens}")

            kwargs = {
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "options": options,
                "stream": True,
            }

            if schema:
                kwargs["format"] = schema

            stream = ollama.chat(**kwargs)

            full_content = ""
            generated_tokens = 0

            print("🔄 Streaming response:")

            start_time = time.time()

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if not content:
                    continue

                full_content += content

                generated_tokens += max(1, int(len(content) / 3.5))

                progress = int((generated_tokens / estimated_tokens) * 100)
                progress = min(progress, 95)

                elapsed = time.time() - start_time

                preview = full_content[-80:].replace("\n", " ")

                print(
                    f"\r⏳ {progress}% | "
                    f"{generated_tokens}/{estimated_tokens} tokens | "
                    f"{elapsed:.1f}s | {preview}",
                    end="",
                    flush=True,
                )

            total_time = time.time() - start_time
            print(
                f"\r✅ 100% | {generated_tokens}/{estimated_tokens} tokens | {total_time:.1f}s"
            )
            print("✅ Streaming complete")

            return Response({"content": full_content})

        except Exception as exc:
            traceback.print_exc()
            return Response(
                {"error": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )