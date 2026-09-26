import pytest

from legal_assistant.docgen.ocr.base import OcrProvider, PageText, get_provider


class FakeProvider:
    async def extract(self, images):
        return [PageText(i.page, "نص", 1.0) for i in images]

    async def extract_fields(self, images, schema):
        return {}


def test_a_structurally_compatible_object_satisfies_the_protocol():
    assert isinstance(FakeProvider(), OcrProvider)


def test_an_incomplete_object_does_not():
    class Partial:
        async def extract(self, images):
            return []

    assert not isinstance(Partial(), OcrProvider)


def test_get_provider_returns_the_gemini_provider_by_default():
    from legal_assistant.docgen.ocr.gemini import GeminiOcrProvider

    assert isinstance(get_provider(), GeminiOcrProvider)


def test_get_provider_rejects_an_unknown_provider_name(monkeypatch):
    monkeypatch.setenv("OCR_PROVIDER", "tesseract")
    with pytest.raises(ValueError) as excinfo:
        get_provider()
    assert "tesseract" in str(excinfo.value)
