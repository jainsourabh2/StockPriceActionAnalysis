"""
Loads credentials from .env file.

All SmartAPI and Anthropic credentials are read from environment variables
so nothing sensitive ends up in source control.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

ANGELONE_API_KEY = os.getenv("ANGELONE_API_KEY")
ANGELONE_CLIENT_ID = os.getenv("ANGELONE_CLIENT_ID")
ANGELONE_PASSWORD = os.getenv("ANGELONE_PASSWORD")
ANGELONE_TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")


def validate() -> bool:
    """Check that all required credentials are present. Prints what's missing."""
    missing = []
    if not ANGELONE_API_KEY:
        missing.append("ANGELONE_API_KEY")
    if not ANGELONE_CLIENT_ID:
        missing.append("ANGELONE_CLIENT_ID")
    if not ANGELONE_PASSWORD:
        missing.append("ANGELONE_PASSWORD")
    if not ANGELONE_TOTP_SECRET:
        missing.append("ANGELONE_TOTP_SECRET")
    if not GOOGLE_CLOUD_API_KEY:
        missing.append("GOOGLE_CLOUD_API_KEY")

    if missing:
        print("Missing credentials in .env file:")
        for var in missing:
            print(f"  - {var}")
        print("\nCopy .env.example to .env and fill in your values:")
        print("  cp .env.example .env")
        return False
    return True


if __name__ == "__main__":
    if validate():
        print("All credentials loaded successfully.")
    else:
        sys.exit(1)
