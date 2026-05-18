import os
from pathlib import Path

from dotenv import load_dotenv

APP_NAME = "StackWire"
ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_ENV_FILE = ROOT_DIR / "stackwire.local.env"


def load_local_env() -> None:
    if LOCAL_ENV_FILE.exists():
        load_dotenv(LOCAL_ENV_FILE, override=False)
