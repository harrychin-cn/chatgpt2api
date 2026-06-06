from __future__ import annotations

import unittest
from unittest import mock

from services.protocol import openai_v1_response
from services.protocol.conversation import ImageOutput


class ResponseImageToolModelTests(unittest.TestCase):
    def test_image_generation_tool_model_overrides_response_model(self) -> None:
        seen_models: list[str] = []

        def fake_stream(request):
            seen_models.append(request.model)
            yield ImageOutput(
                kind="result",
                model=request.model,
                index=1,
                total=1,
                data=[{"b64_json": "aW1hZ2U=", "revised_prompt": request.prompt}],
            )

        with mock.patch.object(openai_v1_response, "stream_image_outputs_with_pool", fake_stream):
            response = openai_v1_response.handle({
                "model": "gpt-5",
                "input": "draw a small red envelope icon",
                "tools": [{"type": "image_generation", "model": "gpt-image-2"}],
            })

        self.assertEqual(seen_models, ["gpt-image-2"])
        self.assertEqual(response["model"], "gpt-image-2")
        self.assertEqual(response["output"][0]["type"], "image_generation_call")


if __name__ == "__main__":
    unittest.main()
