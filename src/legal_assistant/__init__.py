"""Legal Assistant: Arabic-language legal research assistant for Egyptian lawyers."""

# --- Windows workaround, must run before any dependency touches SSL --------
# Some Windows machines have a malformed certificate in the Windows cert
# store that makes ssl.create_default_context()/SSLContext.load_default_certs()
# raise "SSLError: [ASN1: NOT_ENOUGH_DATA]". Several dependencies build an SSL
# context at *import time* (aiohttp, used by google-genai and by
# FlagEmbedding's `datasets` dependency), so this must be patched before any
# submodule of this package is imported. Route cert loading through
# certifi's CA bundle instead of the Windows store. Same root cause as
# arabic_ingest/ingest.py, embeddings.py, and embedding_client.py.
#
# This is a Windows-only cert-store bug -- Linux (Railway's containers) has
# no such malformed store, so the patch must be a no-op there.
import sys

if sys.platform == "win32":
    import ssl

    import certifi

    _orig_create_default_context = ssl.create_default_context

    def _create_default_context_via_certifi(*args, **kwargs):
        kwargs.setdefault("cafile", certifi.where())
        return _orig_create_default_context(*args, **kwargs)

    ssl.create_default_context = _create_default_context_via_certifi

    _orig_load_default_certs = ssl.SSLContext.load_default_certs

    def _load_default_certs_via_certifi(self, *args, **kwargs):
        try:
            return _orig_load_default_certs(self, *args, **kwargs)
        except ssl.SSLError:
            self.load_verify_locations(cafile=certifi.where())

    ssl.SSLContext.load_default_certs = _load_default_certs_via_certifi
