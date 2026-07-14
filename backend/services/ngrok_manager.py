import logging
from pyngrok import ngrok, conf
from backend.config import settings

logger = logging.getLogger(__name__)

_public_url: str | None = None


def start_ngrok() -> str | None:
    global _public_url
    try:
        if settings.NGROK_AUTH_TOKEN:
            conf.get_default().auth_token = settings.NGROK_AUTH_TOKEN
        tunnel = ngrok.connect(settings.PORT, bind_tls=True)
        _public_url = tunnel.public_url
        logger.info("Ngrok tunnel established at %s", _public_url)
        return _public_url
    except Exception as e:
        logger.warning("Ngrok failed to start: %s", e)
        return None


def stop_ngrok():
    global _public_url
    try:
        ngrok.disconnect(_public_url)
        ngrok.kill()
    except Exception:
        pass
    _public_url = None
    logger.info("Ngrok tunnel closed")


def get_public_url() -> str | None:
    return _public_url
