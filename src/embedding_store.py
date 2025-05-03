"""SQLite-based storage for issue embeddings.

This module provides a persistent storage solution for issue embeddings using SQLite.
It handles the storage, retrieval, and management of embeddings and associated metadata
for GitHub issues.

Key Features:
- Efficient storage of embeddings using SQLite
- Automatic database initialization and schema creation
- Serialization/deserialization of numpy arrays
- Comprehensive metadata management
- Update detection for stale embeddings

The storage schema includes:
- issue_number: Unique identifier for the issue
- title: Issue title
- body: Issue description/content
- embedding: Binary blob of the embedding vector
- last_updated: Timestamp of last modification

Dependencies:
    - sqlite3: For database operations
    - numpy: For embedding array operations
    - pathlib: For path handling
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

class EmbeddingStore:
    """Stores and retrieves issue embeddings using SQLite.
    
    This class provides a comprehensive interface for managing issue embeddings
    in a SQLite database. It handles all aspects of embedding storage, including
    serialization, deserialization, and metadata management.
    
    Attributes:
        db_path (Path): Path to the SQLite database file
        
    The database schema includes:
        - issue_number (INTEGER): Primary key, the GitHub issue number
        - title (TEXT): The issue title
        - body (TEXT): The issue description
        - embedding (BLOB): The serialized embedding vector
        - last_updated (TEXT): Timestamp of last modification
    """
    
    def __init__(self, db_path: str = ".github/cache/embeddings.db"):
        """Initialize the embedding store.
        
        Args:
            db_path (str): Path to the SQLite database file. Defaults to
                          ".github/cache/embeddings.db"
                          
        Creates the database and necessary tables if they don't exist.
        Ensures the parent directory exists.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    issue_number INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT,
                    embedding BLOB NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage.
        
        Args:
            embedding (np.ndarray): The embedding vector to serialize
            
        Returns:
            bytes: The serialized embedding as bytes
            
        Converts the embedding to float32 for consistent storage format.
        """
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Convert bytes back to numpy array.
        
        Args:
            data (bytes): The serialized embedding data
            
        Returns:
            np.ndarray: The deserialized embedding vector
            
        Reconstructs the original float32 numpy array from bytes.
        """
        return np.frombuffer(data, dtype=np.float32)
    
    def store_embedding(
        self,
        issue_number: int,
        title: str,
        body: str,
        embedding: np.ndarray,
        last_updated: str
    ):
        """Store an embedding for an issue.
        
        Args:
            issue_number (int): The GitHub issue number
            title (str): The issue title
            body (str): The issue description
            embedding (np.ndarray): The embedding vector
            last_updated (str): Timestamp of last modification
            
        Updates existing embedding if issue_number exists, otherwise creates new entry.
        Handles serialization of the embedding vector automatically.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (issue_number, title, body, embedding, last_updated)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    issue_number,
                    title,
                    body,
                    self._serialize_embedding(embedding),
                    last_updated
                )
            )
    
    def get_embedding(self, issue_number: int) -> Optional[Tuple[np.ndarray, str]]:
        """Get the embedding and last_updated time for an issue.
        
        Args:
            issue_number (int): The GitHub issue number
            
        Returns:
            Optional[Tuple[np.ndarray, str]]: Tuple of (embedding, last_updated)
                if found, None if not found
                
        The embedding is automatically deserialized from its stored format.
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT embedding, last_updated FROM embeddings WHERE issue_number = ?",
                (issue_number,)
            ).fetchone()
            
            if result:
                return self._deserialize_embedding(result[0]), result[1]
            return None
    
    def get_issue_metadata(self, issue_number: int) -> Optional[Tuple[str, str, np.ndarray, str]]:
        """Get the full metadata for an issue.
        
        Args:
            issue_number (int): The GitHub issue number
            
        Returns:
            Optional[Tuple[str, str, np.ndarray, str]]: Tuple of
                (title, body, embedding, last_updated) if found, None if not found
                
        Retrieves all stored information about an issue, including the
        deserialized embedding vector.
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT title, body, embedding, last_updated FROM embeddings WHERE issue_number = ?",
                (issue_number,)
            ).fetchone()
            
            if result:
                return result[0], result[1], self._deserialize_embedding(result[2]), result[3]
            return None
    
    def get_all_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """Get all stored embeddings.
        
        Returns:
            List[Tuple[int, np.ndarray]]: List of tuples containing
                (issue_number, embedding) for all stored issues
                
        Useful for batch operations or similarity comparisons across all issues.
        Embeddings are automatically deserialized.
        """
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute(
                "SELECT issue_number, embedding FROM embeddings"
            ).fetchall()
            
            return [
                (num, self._deserialize_embedding(emb))
                for num, emb in results
            ]
    
    def get_all_issue_metadata(self) -> Dict[int, Tuple[str, str, np.ndarray, str]]:
        """Get metadata for all issues.
        
        Returns:
            Dict[int, Tuple[str, str, np.ndarray, str]]: Dictionary mapping
                issue numbers to tuples of (title, body, embedding, last_updated)
                
        Provides complete metadata for all stored issues in a single query.
        Useful for bulk operations or full database analysis.
        """
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute(
                "SELECT issue_number, title, body, embedding, last_updated FROM embeddings"
            ).fetchall()
            
            return {
                row[0]: (
                    row[1],  # title
                    row[2],  # body
                    self._deserialize_embedding(row[3]),  # embedding
                    row[4]   # last_updated
                )
                for row in results
            }
    
    def needs_update(self, issue_number: int, last_updated: str) -> bool:
        """Check if an issue's embedding needs to be updated.
        
        Args:
            issue_number (int): The GitHub issue number
            last_updated (str): Current timestamp to compare against
            
        Returns:
            bool: True if the issue needs updating (either doesn't exist
                 or has a different last_updated time), False otherwise
                 
        Used to determine if an issue's embedding should be recalculated
        based on modification time.
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT last_updated FROM embeddings WHERE issue_number = ?",
                (issue_number,)
            ).fetchone()
            
            if not result:
                return True
            
            return result[0] != last_updated
    
    def remove_embedding(self, issue_number: int):
        """Remove an issue's embedding.
        
        Args:
            issue_number (int): The GitHub issue number to remove
            
        Permanently deletes all stored data for the specified issue.
        Commits the deletion immediately.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE issue_number = ?",
                (issue_number,)
            )
            conn.commit() 