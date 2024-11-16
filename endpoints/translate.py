from fastapi import APIRouter, Depends
from models.inference import TRANSLATOR
from schemas.request import TranslationRequest
from schemas.response import TranslationResponse

router = APIRouter()

@router.post(
    "/translate",
    response_model=TranslationResponse,
    summary="Translate English to Amharic",
    description="Accepts an English text string and returns the translated Amharic text.",
    tags=["Translation"],
)
async def translate(payload: TranslationRequest, translator = Depends(lambda: TRANSLATOR)):
    """
    Translate English text to Amharic using a transformer model.
    """
    translated_text = translator.translate(payload.text, max_len=50)
    return TranslationResponse(original_text=payload.text, translated_text=translated_text.strip())

