"""Decision manual auction-guidance V5 (keeps the v3 import path for compatibility)."""

from .config import AuctionV3Config
from .engine import AuctionV3Engine, RunResult

__all__ = ["AuctionV3Config", "AuctionV3Engine", "RunResult"]
