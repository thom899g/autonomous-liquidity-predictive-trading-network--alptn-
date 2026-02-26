"""
Real-time market data collector with error handling and rate limiting
Architectural Choice: Use ccxt for exchange connectivity as it's standard,
well-documented, and supports multiple exchanges consistently.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from config import EXCHANGES, logger
from state_manager import StateManager

@dataclass
class MarketData:
    """Structured market data container"""
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    mid_price: float
    spread_bps: float
    order_book_depth: Dict[str, float]
    trade_volume: Optional[float] = None
    funding_rate: Optional[float] = None
    
class DataCollector:
    """Robust market data collection with fault tolerance"""
    
    def __init__(self, exchange_name: str = "binance"):
        self.exchange_name = exchange_name
        self.exchange = self._initialize_exchange()
        self.state_manager = StateManager()
        self.log