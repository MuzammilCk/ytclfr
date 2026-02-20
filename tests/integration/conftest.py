"""
tests/integration/conftest.py

Patches out heavy system-level imports before any app module is loaded.
This lets integration tests run on bare dev machines or CI without:
  - Kerberos for Windows (amqp/gssapi Windows dependency)
  - A running Redis/Celery broker
  - A running PostgreSQL/MongoDB instance
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock

# ── 0. Patch bcrypt.hashpw to silently truncate >72-byte passwords ────────────
# passlib 1.7.x runs an internal anti-regression test during bcrypt backend
# initialisation that hashes a 112-byte password.  bcrypt 4.x added a strict
# check that raises ValueError for passwords longer than 72 bytes, which breaks
# that initialization entirely.  We restore the old silent-truncation behaviour
# so passlib's self-test succeeds without modifying production code.
import bcrypt as _bcrypt_module
_original_hashpw = _bcrypt_module.hashpw

def _hashpw_truncate(password: bytes, salt: bytes) -> bytes:
    if isinstance(password, bytes) and len(password) > 72:
        password = password[:72]
    return _original_hashpw(password, salt)

_bcrypt_module.hashpw = _hashpw_truncate

# ── 1. Stub out the entire celery stack before any import touches it ──────────
# amqp pulls gssapi which requires Kerberos for Windows to be installed.
# We prevent that by replacing celery with a lightweight mock.
for _mod in [
    "celery",
    "celery.app",
    "celery.app.builtins",
    "celery.utils",
    "celery.utils.nodenames",
    "kombu",
    "kombu.utils",
    "amqp",
    "gssapi",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Ensure celery.Celery is a usable class mock
import celery as _celery_mock  # noqa: E402  (the mock we just inserted)
_celery_app_instance = MagicMock()
_celery_app_instance.conf = MagicMock()
_celery_app_instance.conf.update = MagicMock()
_celery_mock.Celery = MagicMock(return_value=_celery_app_instance)

# ── 2. Stub out heavy ML imports used transitively by service modules ─────────
_HEAVY_MODS = [
    # torch and all dotted submodules the services import at module level
    "torch", "torch.nn", "torch.nn.functional", "torch.optim",
    "torch.utils", "torch.utils.data", "torch.cuda", "torch.backends",
    "torch.backends.cudnn",
    # torchvision
    "torchvision", "torchvision.models", "torchvision.transforms",
    "torchvision.transforms.functional",
    # torchaudio
    "torchaudio",
    # transformers
    "transformers",
    # ML vision / audio
    "ultralytics", "ultralytics.models", "ultralytics.engine",
    "ultralytics.engine.results",
    # whisper (openai-whisper)
    "whisper",
    # opencv
    "cv2",
    # moviepy
    "moviepy", "moviepy.editor",
    # spleeter
    "spleeter", "spleeter.separator",
    # OCR
    "pytesseract",
    # misc
    "openai",
]
for _heavy in _HEAVY_MODS:
    if _heavy not in sys.modules:
        sys.modules[_heavy] = MagicMock()
