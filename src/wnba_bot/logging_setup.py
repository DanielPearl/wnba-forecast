"""Re-export shim. Canonical implementation lives in ``kalshi_sdk.logging``."""
from kalshi_sdk.logging import setup_logging

__all__ = ["setup_logging"]
