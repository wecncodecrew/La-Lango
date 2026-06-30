# lalango/api/routes.py
#
# This file defines the API endpoints.
#
# Each function below is a "route handler" — FastAPI calls it
# when a matching HTTP request comes in.

from fastapi import APIRouter, HTTPException
from lalango.api.schemas import (
    TranslationRequest,
    TranslationResponse,
    LanguageInfo,
    HealthResponse,
)

# APIRouter lets us group routes together.
# The router is registered in main.py.
router = APIRouter()

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# This dictionary maps language pairs to the model that handles them.
# When you add a new language, register it here.
#
# Format: (source_lang, target_lang) → model_name
#
# TODO: As models get trained, replace "coming_soon" with the model identifier.
#       The actual model loading logic will go in main.py.
SUPPORTED_PAIRS = {
    # ("konkani", "english"): "seq2seq_lstm",
    # ("yoruba", "english"):  "seq2seq_lstm",
    #
    # Add your language pair here once the model is trained!
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """
    Check if the API is up and running.

    This is useful for deployment monitoring — a load balancer can ping
    this endpoint to verify the server is alive.
    """
    return HealthResponse(status="ok", message="La Lango AI is running.")


@router.get("/languages", response_model=list[LanguageInfo], tags=["Languages"])
def list_languages():
    """
    List all supported language pairs.

    Returns a list of language pairs that currently have a trained model.
    Check the ROADMAP.md for pairs that are in progress.
    """
    return [
        LanguageInfo(source_lang=src, target_lang=tgt, model=model)
        for (src, tgt), model in SUPPORTED_PAIRS.items()
    ]


@router.post("/translate", response_model=TranslationResponse, tags=["Translation"])
def translate(request: TranslationRequest):
    """
    Translate a sentence from one language to another.

    The request must include:
    - `text`: The sentence to translate (max 500 characters)
    - `source_lang`: The input language (e.g. "konkani")
    - `target_lang`: The output language (e.g. "english")

    Returns the translated text and metadata about which model was used.

    ---

    **Note for contributors:** This endpoint currently returns a placeholder
    response until a trained model is loaded. See main.py for the model
    loading logic that needs to be connected here.
    """
    pair = (request.source_lang.lower(), request.target_lang.lower())

    # Check that we support this language pair
    if pair not in SUPPORTED_PAIRS:
        supported = [f"{s}→{t}" for s, t in SUPPORTED_PAIRS.keys()]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Language pair '{request.source_lang}→{request.target_lang}' "
                f"is not supported yet. "
                f"Supported pairs: {supported if supported else 'None yet — contribute one!'}. "
                f"See languages/ to add a new language."
            )
        )

    model_name = SUPPORTED_PAIRS[pair]

    # ---------------------------------------------------------------------------
    # TODO (Phase 1 — final step):
    #   Load the trained model from main.py's model registry and call .translate()
    #
    #   The translate() method should be available on all model objects.
    #   Here is the rough shape of what this should look like:
    #
    #   from lalango.api.main import loaded_models, tokenizers
    #   model = loaded_models[pair]
    #   src_tokenizer, tgt_tokenizer = tokenizers[pair]
    #
    #   encoded = src_tokenizer.encode(request.text)
    #   source_tensor = torch.tensor([encoded])
    #   output_indices = model.translate(source_tensor)
    #   translation = tgt_tokenizer.decode(output_indices)
    #
    #   return TranslationResponse(
    #       translation=translation,
    #       source_lang=request.source_lang,
    #       target_lang=request.target_lang,
    #       model=model_name,
    #   )
    # ---------------------------------------------------------------------------

    # Placeholder until a trained model is connected
    return TranslationResponse(
        translation=f"[Model '{model_name}' is registered but not yet loaded. "
                    f"Complete Phase 1 to connect a trained model here.]",
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        model=model_name,
    )
