"""
Embedding configuration management.

Stores which embedding model was used to create the vector database,
ensuring retrieval uses the same model for dimension consistency.
"""

import os
import json
import logging
from pathlib import Path
from .logging_config import get_logger

logger = get_logger(__name__)

CONFIG_FILENAME = ".embedding_config.json"

class EmbeddingConfig:
    """Manages embedding model configuration for consistency between ingestion and retrieval."""
    
    def __init__(self, db_path: str):
        """Initialize with the database path."""
        self.db_path = db_path
        self.config_file = os.path.join(db_path, CONFIG_FILENAME)
        self.model = None
        self.load_config()
    
    def load_config(self):
        """Load embedding model from config file if it exists."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.model = config.get("embedding_model", "text-embedding-3-small")
                    logger.info("Loaded embedding model from config: %s", self.model)
            except Exception as e:
                logger.warning("Failed to load embedding config: %s, using default", str(e)[:100])
                self.model = "text-embedding-3-small"
        else:
            self.model = "text-embedding-3-small"
            logger.info("No embedding config found, using default: %s", self.model)
    
    def save_config(self, model: str):
        """Save embedding model to config file."""
        try:
            # Create db_path if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            config = {"embedding_model": model}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.model = model
            logger.info("Saved embedding model config: %s", model)
        except Exception as e:
            logger.error("Failed to save embedding config: %s", str(e)[:100])
    
    def get_model(self) -> str:
        """Get the configured embedding model."""
        return self.model or "text-embedding-3-small"
    
    def set_model(self, model: str):
        """Set and persist the embedding model."""
        self.save_config(model)
