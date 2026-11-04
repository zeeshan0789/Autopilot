"""Rule based routing policy.

A request with model "auto" (the configured alias) is routed to a concrete model by
declarative rules from config/policy.yaml. Any other model name passes through untouched.

The rules are heuristics over the prompt (length, code, keywords, history, output size).
They say nothing about answer quality. Whether routing helps must be measured.
"""

import logging
import re
from dataclasses import dataclass, field

from app.schemas.openai import ChatCompletionRequest
from app.settings import load_policy_config

logger = logging.getLogger("autopilot.policy")

INT_CONDITIONS = {
    "min_chars", "max_chars", "min_messages", "min_questions", "min_max_tokens",
}
KNOWN_CONDITIONS = INT_CONDITIONS | {"any_keywords", "contains_code"}


@dataclass(frozen=True)
class Rule:
    name: str
    tier: str
    when: dict


@dataclass(frozen=True)
class PolicySettings:
    tiers: dict
    default_tier: str
    rules: tuple = ()
    enabled: bool = True
    alias: str = "auto"

    def __post_init__(self):
        if not self.alias or not self.alias.strip():
            raise ValueError("alias must not be empty")
        if not self.tiers:
            raise ValueError("at least one tier is required")
        for tier, model in self.tiers.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"tier '{tier}' needs a model name")
        if self.default_tier not in self.tiers:
            raise ValueError(f"default_tier '{self.default_tier}' is not a defined tier")

        seen = set()
        for rule in self.rules:
            if not rule.name or rule.name in seen:
                raise ValueError(f"rule names must be unique and not empty: '{rule.name}'")
            seen.add(rule.name)
            if rule.tier not in self.tiers:
                raise ValueError(f"rule '{rule.name}' uses unknown tier '{rule.tier}'")
            _validate_conditions(rule)

    @classmethod
    def from_config(cls, config: dict) -> "PolicySettings":
        rules = tuple(
            Rule(name=r.get("name", ""), tier=r.get("tier", ""), when=r.get("when") or {})
            for r in config.get("rules") or []
        )
        return cls(
            enabled=config.get("enabled", True),
            alias=config.get("alias", "auto"),
            tiers=dict(config.get("tiers") or {}),
            default_tier=config.get("default_tier", ""),
            rules=rules,
        )


def _validate_conditions(rule: Rule) -> None:
    if not rule.when:
        raise ValueError(f"rule '{rule.name}' has no conditions (use default_tier instead)")
    for key, value in rule.when.items():
        if key not in KNOWN_CONDITIONS:
            raise ValueError(f"rule '{rule.name}': unknown condition '{key}'")
        if key in INT_CONDITIONS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"rule '{rule.name}': {key} must be a non negative integer")
        elif key == "any_keywords":
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(k, str) and k.strip() for k in value)
            ):
                raise ValueError(f"rule '{rule.name}': any_keywords must be a list of words")
        elif key == "contains_code" and not isinstance(value, bool):
            raise ValueError(f"rule '{rule.name}': contains_code must be true or false")


# ---------- Reading the prompt ----------

# Deliberately strict: "class action" or "import duties" in plain English must not count
_CODE_RE = re.compile(
    r"```"
    r"|\bdef\s+\w+\s*\("
    r"|\bclass\s+\w+\s*[:({]"
    r"|^\s*import\s+[\w.]+\s*$"
    r"|^\s*from\s+[\w.]+\s+import\s+"
    r"|\bfunction\s+\w*\s*\("
    r"|=>"
    r"|#include\s*<",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class PromptFeatures:
    text: str
    chars: int
    messages: int
    questions: int
    has_code: bool
    max_tokens: int | None


def extract_features(request: ChatCompletionRequest) -> PromptFeatures:
    turns = [m.content for m in request.messages if m.role != "system"]
    text = "\n".join(turns).strip()
    return PromptFeatures(
        text=text,
        chars=len(text),
        messages=len(turns),
        questions=text.count("?"),
        has_code=_CODE_RE.search(text) is not None,
        max_tokens=request.max_tokens,
    )


@dataclass(frozen=True)
class RouteDecision:
    model: str  # the concrete model to use
    rule: str  # rule name, "default", "explicit" (no routing) or "policy_error"
    tier: str | None
    routed: bool  # True when the policy chose the model


class PolicyRouter:
    def __init__(self, settings: PolicySettings):
        self.settings = settings
        # Compile keyword patterns once: whole words or phrases, any case
        self._keyword_patterns = {
            rule.name: [
                re.compile(rf"\b{re.escape(k.strip())}\b", re.IGNORECASE)
                for k in rule.when["any_keywords"]
            ]
            for rule in settings.rules
            if "any_keywords" in rule.when
        }

    def decide(self, request: ChatCompletionRequest) -> RouteDecision:
        s = self.settings
        if not s.enabled or request.model != s.alias:
            return RouteDecision(request.model, "explicit", None, False)

        try:
            features = extract_features(request)
            for rule in s.rules:
                if self._matches(rule, features):
                    return RouteDecision(s.tiers[rule.tier], rule.name, rule.tier, True)
            return RouteDecision(s.tiers[s.default_tier], "default", s.default_tier, True)
        except Exception as e:
            # A policy bug must not fail the request: use the default tier
            logger.warning("Policy failed (%s), using the default tier", e)
            return RouteDecision(s.tiers[s.default_tier], "policy_error", s.default_tier, True)

    def _matches(self, rule: Rule, f: PromptFeatures) -> bool:
        for key, value in rule.when.items():
            if key == "min_chars" and not f.chars >= value:
                return False
            if key == "max_chars" and not f.chars <= value:
                return False
            if key == "min_messages" and not f.messages >= value:
                return False
            if key == "min_questions" and not f.questions >= value:
                return False
            if key == "min_max_tokens" and not (f.max_tokens is not None and f.max_tokens >= value):
                return False
            if key == "contains_code" and f.has_code != value:
                return False
            if key == "any_keywords" and not any(
                p.search(f.text) for p in self._keyword_patterns[rule.name]
            ):
                return False
        return True


policy_router = PolicyRouter(PolicySettings.from_config(load_policy_config()))