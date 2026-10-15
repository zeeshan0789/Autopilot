from app.providers.base import BaseProvider
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai_compat import OpenAICompatProvider
from app.settings import load_fallbacks_config, load_providers_config


def build_provider(cfg: dict) -> BaseProvider:
    provider_type = cfg["type"]

    if provider_type == "mock":
        return MockProvider(
            name=cfg["name"],
            mode=cfg.get("mode", "ok"),
            delay=cfg.get("delay", 0.0),
            fail_first=cfg.get("fail_first", 0),
        )

    if provider_type == "ollama":
        return OllamaProvider(
            name=cfg["name"],
            base_url=cfg["base_url"],
            timeout=cfg.get("timeout", 120),
        )

    if provider_type == "openai_compat":
        return OpenAICompatProvider(
            name=cfg["name"],
            base_url=cfg["base_url"],
            api_key_env=cfg["api_key_env"],
            timeout=cfg.get("timeout", 30),
        )

    raise ValueError(f"Unknown provider type: {provider_type}")


class ProviderRegistry:
    def __init__(self, configs: list[dict], fallbacks: dict[str, list[str]]):
        # Har entry: (provider object, us ke models ka set)
        self._entries = [
            (build_provider(cfg), set(cfg["models"])) for cfg in configs
        ]
        self._fallbacks = fallbacks

    def for_model(self, model: str) -> list[BaseProvider]:
        """Is model ko handle karne wale providers, file ke order mein."""
        return [p for p, models in self._entries if model in models]

    def chain_for(self, model: str) -> list[tuple[BaseProvider, str]]:
        """(provider, model) ki poori chain: pehle asli model, phir fallbacks."""
        chain = []
        for m in [model] + self._fallbacks.get(model, []):
            for provider in self.for_model(m):
                chain.append((provider, m))
        return chain


registry = ProviderRegistry(load_providers_config(), load_fallbacks_config())