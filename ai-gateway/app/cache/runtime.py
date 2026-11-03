"""The cache service used by the running gateway, built from config/cache.yaml.

Creating it does not load the embedding model: that happens in warm_up() at startup.
"""

from app.cache.embedder import Embedder
from app.cache.service import CacheService, CacheSettings
from app.settings import load_cache_config

cache_service = CacheService(CacheSettings(**load_cache_config()), Embedder())