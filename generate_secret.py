"""
generate_secret.py
------------------
Run this script ONCE from the project root to generate the Streamlit Cloud
secret for your Google service account credentials.

Usage:
    python generate_secret.py

Then copy the printed output and paste it into:
    Streamlit Cloud → Your App → Settings → Secrets
"""

import base64
import pathlib
import sys

CREDS_FILE = pathlib.Path("credentials.json")

if not CREDS_FILE.exists():
    print(f"ERROR: '{CREDS_FILE}' not found in current directory.")
    print("Run this script from the project root folder where credentials.json lives.")
    sys.exit(1)

raw_bytes = CREDS_FILE.read_bytes()
b64_string = base64.b64encode(raw_bytes).decode("utf-8")

print("\n" + "=" * 60)
print("COPY EVERYTHING BELOW THIS LINE")
print("=" * 60)
print()
print("[gsc_credentials]")
print(f'json_b64 = "{b64_string}"')
print()
print("=" * 60)
print("COPY EVERYTHING ABOVE THIS LINE")
print("=" * 60)
print()
print("Paste into: Streamlit Cloud → App → Settings → Secrets")
print("Then click Save.")
