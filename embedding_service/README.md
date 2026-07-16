# Legal Assistant Embedding Service

Standalone BGE-M3 embed + ColBERT rerank service. Self-contained: does not
import the `legal_assistant` package. Deployed separately, on its own VPS.

Reproduces the ingestion project's embedding pipeline exactly (same model,
same `normalize_for_embedding`, same encode configuration) so that vectors
produced here are numerically consistent with the vectors already stored in
Qdrant Cloud. See `scripts/check_embedding_consistency.py` in the main
project for the empirical proof.

## Endpoints

- `GET /health` — liveness, no auth required.
- `POST /embed` — `{"texts": ["..."]}` → dense + sparse vectors per text.
  Applies `normalize_for_embedding` internally, so pass raw article text or a
  raw query.
- `POST /rerank` — `{"query": "...", "passages": ["..."]}` → ColBERT scores,
  passages returned sorted by relevance (`index` refers to the original
  input order). Capped at 50 passages per request; encoded in small internal
  batches to bound peak memory.

Both `/embed` and `/rerank` require `Authorization: Bearer <token>`.

## Deploying on the Hetzner VPS

Prerequisites: Ubuntu VPS, root SSH access.

1. **Install Docker:**

   ```
   curl -fsSL https://get.docker.com | sh
   ```

2. **Add a swap file** (safety cushion against a hard OOM kill during the
   ColBERT rerank spike):

   ```
   fallocate -l 4G /swapfile
   chmod 600 /swapfile
   mkswap /swapfile
   swapon /swapfile
   echo '/swapfile none swap sw 0 0' >> /etc/fstab
   ```

3. **Copy this `embedding_service/` folder to the VPS** (e.g. `scp -r` or
   `rsync`), then on the VPS:

   ```
   cd embedding_service
   cp .env.example .env
   # edit .env: set EMBEDDING_SERVICE_TOKEN to a strong random value
   docker compose up -d --build
   ```

4. **Verify:**

   ```
   curl -sk https://<vps-ip>/health
   ```

   (`-k` is needed only if using Caddy's self-signed internal CA — see
   `Caddyfile` comments. If you have a real domain pointed at the VPS,
   replace the address in `Caddyfile` with that domain and Caddy will
   obtain a Let's Encrypt certificate automatically; drop `-k` in that case.)

5. **Record the public URL** (e.g. `https://<vps-ip>` or your domain) and the
   token — put both into the main project's `.env` as
   `EMBEDDING_SERVICE_URL` and `EMBEDDING_SERVICE_TOKEN`.

## Memory notes (4 GB trial)

`docker-compose.yml` caps the embedding container at `mem_limit: 3800m` with
`memswap_limit: 7800m` so it can spill into the swap file under a spike
rather than being hard OOM-killed. If `docker stats` shows swap usage during
rerank load, or the container gets OOM-killed, that is the signal to move up
to an 8 GB box (see `scripts/check_retrieval.py` in the main project for the
measurement).
