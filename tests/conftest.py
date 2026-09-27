"""Give every test a valid-but-fake Settings environment.

The docgen unit tests must run with no .env, no network, and no database.
`Settings` has required fields, so they are filled with obvious dummies here
BEFORE any test imports a module that calls get_settings(). Anything that
needs real credentials belongs in scripts/, not in tests/.
"""

import os

os.environ.setdefault("QDRANT_CLOUD_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_CLOUD_API_KEY", "test")
os.environ.setdefault("QDRANT_COLLECTION_NAME", "test")
os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://localhost:9999")
os.environ.setdefault("EMBEDDING_SERVICE_TOKEN", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("LLM_MODEL", "gemini-3-pro")
