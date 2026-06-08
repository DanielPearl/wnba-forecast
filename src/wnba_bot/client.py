"""Re-export shim for the Kalshi REST client (canonical impl in ``kalshi_sdk``)."""
from kalshi_sdk import KalshiClient, KalshiError, Orderbook
from kalshi_sdk.orderbook import parse_orderbook  # noqa: F401

__all__ = ["KalshiClient", "KalshiError", "Orderbook"]
