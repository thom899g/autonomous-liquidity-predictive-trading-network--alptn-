"""
ALPTN Configuration Management
Centralized configuration with environment variables and defaults
"""
import os
from dataclasses import dataclass
from typing import Optional
import logging

# Firebase configuration (MUST use Firebase per ecosystem rules)
FIREBASE_CONFIG = {
    "project_id": os.getenv("FIREBASE_PROJECT_ID", "alptn-production"),
    "storage_bucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
    "database_url": os.getenv("FIREBASE_DATABASE_URL", "")
}

# Exchange configuration
EXCHANGES = {
    "binance": {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "api_secret": os.getenv("BINANCE_API_SECRET", ""),
        "rate_limit": 1200  # requests per minute
    },
    "coinbase": {
        "api_key": os.getenv("COINBASE_API_KEY", ""),
        "api_secret": os.getenv("COINBASE_API_SECRET", ""),
        "rate_limit": 300
    }
}

# Trading parameters
TRADING_CONFIG = {
    "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "0.1")),  # 10% of portfolio
    "max_slippage_tolerance": float(os.getenv("MAX_SLIPPAGE", "0.001")),  # 0.1%
    "circuit_breaker_threshold": float(os.getenv("CIRCUIT_BREAKER", "0.05")),  # 5% drawdown
    "minimum_liquidity_score": float(os.getenv("MIN_LIQUIDITY", "0.3"))  # 0-1 scale
}

# ML Model configuration
MODEL_CONFIG = {
    "lstm_units": 50,
    "dense_units": 25,
    "dropout_rate": 0.2,
    "sequence_length": 100,
    "prediction_horizon": 10,  # 10 periods ahead
    "training_interval": "1h",  # Retrain every hour
    "validation_split": 0.15
}

@dataclass
class LogConfig:
    """Logging configuration"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/alptn_system.log"
    
    def setup(self) -> logging.Logger:
        """Configure and return logger"""
        logging.basicConfig(
            level=getattr(logging, self.level),
            format=self.format,
            handlers=[
                logging.FileHandler(self.file_path),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

# Initialize logger
logger = LogConfig().setup()

def validate_config() -> bool:
    """Validate critical configuration parameters"""
    required_vars = [
        "FIREBASE_PROJECT_ID",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False
    
    # Validate Firebase config
    if not FIREBASE_CONFIG["project_id"]:
        logger.error("Firebase project ID must be configured")
        return False
        
    return True

# Edge case: Handle missing configuration gracefully
try:
    config_valid = validate_config()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    config_valid = False