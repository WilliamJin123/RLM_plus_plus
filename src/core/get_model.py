from src.config import config
from keycycle import MultiProviderWrapper
def get_model():
    config.FAST_MODEL_PROVIDER.lower()
        
    return MultiProviderWrapper.from_env(
        provider=config.FAST_MODEL_PROVIDER,
        default_model_id=config.FAST_MODEL_NAME,
    ).get_model()