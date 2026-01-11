from typing import Optional
from src.config.config import config
from keycycle import MultiProviderWrapper

def get_model(provider: Optional[str] = None, model_id: Optional[str] = None):
    final_provider = provider or config.FAST_MODEL_PROVIDER
    final_model_id = model_id or config.FAST_MODEL_NAME
    
    return MultiProviderWrapper.from_env(
        provider=final_provider,
        default_model_id=final_model_id,
    ).get_model()