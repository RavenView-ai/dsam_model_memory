"""
Integration between the queue handler and embedding system.

This module provides processors for different embedding tasks and manages
the asynchronous processing of memory embeddings.
"""

import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .queue_handler import (
    MemoryQueueHandler,
    TaskPriority,
    get_queue_handler
)
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .embedding import get_llama_embedder
from .config import cfg

logger = logging.getLogger(__name__)


class EmbeddingQueueProcessor:
    """
    Processor for embedding-related tasks in the queue.

    This class handles various embedding operations including:
    - Main memory embeddings
    - Component embeddings (WHO, WHERE, WHAT)
    - Batch processing for backfills
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        index: Optional[FaissIndex] = None,
        embedder: Optional[Any] = None
    ):
        """
        Initialize the embedding processor.

        Args:
            store: Memory store instance
            index: FAISS index instance
            embedder: Embedder instance
        """
        self.store = store or MemoryStore(cfg.db_path)
        self.index = index or FaissIndex(cfg.embed_dim, cfg.index_path)
        self.embedder = embedder or get_llama_embedder()

        # Initialize queue handler
        self.queue_handler = get_queue_handler()

        # Register processors
        self._register_processors()

        logger.info("EmbeddingQueueProcessor initialized")

    def _register_processors(self):
        """Register all embedding processors with the queue handler."""
        self.queue_handler.register_processor(
            'embed_memory',
            self._process_memory_embedding
        )
        self.queue_handler.register_processor(
            'embed_components',
            self._process_component_embeddings
        )
        self.queue_handler.register_processor(
            'update_embedding',
            self._process_embedding_update
        )
        self.queue_handler.register_processor(
            'batch_embed',
            self._process_batch_embeddings
        )

    def _process_memory_embedding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single memory embedding task.

        Args:
            payload: Task payload containing memory_id and content

        Returns:
            Result dictionary
        """
        memory_id = payload['memory_id']
        content = payload['content']

        try:
            # Generate embedding
            embedding = self.embedder.encode([content], normalize_embeddings=True)[0]

            # Store in FAISS
            self.index.add(memory_id, embedding)

            # Update memory with embedding
            with self.store.connect() as con:
                con.execute(
                    """
                    UPDATE memories
                    SET embedding = ?, embedding_dim = ?
                    WHERE memory_id = ?
                    """,
                    (
                        embedding.astype('float32').tobytes(),
                        embedding.shape[0],
                        memory_id
                    )
                )
                con.commit()

            logger.debug(f"Embedded memory {memory_id}")
            return {'status': 'success', 'memory_id': memory_id}

        except Exception as e:
            logger.error(f"Failed to embed memory {memory_id}: {e}")
            raise

    def _process_component_embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process component embeddings for a memory.

        Args:
            payload: Task payload containing memory data

        Returns:
            Result dictionary
        """
        memory_id = payload['memory_id']
        components_processed = []

        try:
            # Process WHO embeddings
            if payload.get('who_id'):
                who_vec = self.embedder.encode(
                    [payload['who_id']],
                    normalize_embeddings=True
                )[0]
                self.store.store_component_embedding(
                    memory_id, 'who', payload['who_id'],
                    who_vec.astype('float32').tobytes(), who_vec.shape[0]
                )
                self.index.add(f"who:{memory_id}", who_vec)
                components_processed.append('who_id')

            # Process WHO_LIST embeddings
            if payload.get('who_list'):
                try:
                    who_entities = json.loads(payload['who_list'])
                    for entity in who_entities:
                        if entity and entity != payload.get('who_id'):
                            entity_vec = self.embedder.encode(
                                [entity],
                                normalize_embeddings=True
                            )[0]
                            self.store.store_component_embedding(
                                memory_id, 'who', entity,
                                entity_vec.astype('float32').tobytes(),
                                entity_vec.shape[0]
                            )
                    components_processed.append('who_list')
                except (json.JSONDecodeError, TypeError):
                    pass

            # Process WHERE embeddings
            if payload.get('where_value'):
                where_vec = self.embedder.encode(
                    [payload['where_value']],
                    normalize_embeddings=True
                )[0]
                self.store.store_component_embedding(
                    memory_id, 'where', payload['where_value'],
                    where_vec.astype('float32').tobytes(), where_vec.shape[0]
                )
                self.index.add(f"where:{memory_id}", where_vec)
                components_processed.append('where_value')

            # Process WHERE_LIST embeddings
            if payload.get('where_list'):
                try:
                    where_entities = json.loads(payload['where_list'])
                    for entity in where_entities:
                        if entity and entity != payload.get('where_value'):
                            entity_vec = self.embedder.encode(
                                [entity],
                                normalize_embeddings=True
                            )[0]
                            self.store.store_component_embedding(
                                memory_id, 'where', entity,
                                entity_vec.astype('float32').tobytes(),
                                entity_vec.shape[0]
                            )
                    components_processed.append('where_list')
                except (json.JSONDecodeError, TypeError):
                    pass

            # Process WHAT embeddings (limit to first 3 entities)
            if payload.get('what'):
                try:
                    what_content = payload['what']
                    if isinstance(what_content, str):
                        if what_content.startswith('['):
                            what_entities = json.loads(what_content)
                        else:
                            what_entities = [what_content]
                    else:
                        what_entities = what_content

                    for entity in what_entities[:3]:
                        if entity:
                            entity_vec = self.embedder.encode(
                                [entity],
                                normalize_embeddings=True
                            )[0]
                            self.store.store_component_embedding(
                                memory_id, 'what', entity,
                                entity_vec.astype('float32').tobytes(),
                                entity_vec.shape[0]
                            )
                    components_processed.append('what')
                except (json.JSONDecodeError, TypeError):
                    pass

            logger.debug(f"Processed components for {memory_id}: {components_processed}")
            return {
                'status': 'success',
                'memory_id': memory_id,
                'components_processed': components_processed
            }

        except Exception as e:
            logger.error(f"Failed to process components for {memory_id}: {e}")
            raise

    def _process_embedding_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing memory embedding.

        Args:
            payload: Task payload containing memory_id and new content

        Returns:
            Result dictionary
        """
        memory_id = payload['memory_id']
        new_content = payload['content']

        try:
            # Generate new embedding
            new_embedding = self.embedder.encode(
                [new_content],
                normalize_embeddings=True
            )[0]

            # Update in FAISS
            self.index.update(memory_id, new_embedding)

            # Update in database
            with self.store.connect() as con:
                con.execute(
                    """
                    UPDATE memories
                    SET embedding = ?, embedding_dim = ?, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        new_embedding.astype('float32').tobytes(),
                        new_embedding.shape[0],
                        datetime.now().isoformat(),
                        memory_id
                    )
                )
                con.commit()

            logger.debug(f"Updated embedding for memory {memory_id}")
            return {'status': 'success', 'memory_id': memory_id}

        except Exception as e:
            logger.error(f"Failed to update embedding for {memory_id}: {e}")
            raise

    def _process_batch_embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of embeddings.

        Args:
            payload: Task payload containing list of memories

        Returns:
            Result dictionary
        """
        memories = payload['memories']
        successful = 0
        failed = 0

        for memory_data in memories:
            try:
                # Process each memory's embeddings
                self._process_memory_embedding({
                    'memory_id': memory_data['memory_id'],
                    'content': memory_data.get('content', '')
                })

                # Process components if available
                if any(k in memory_data for k in ['who_id', 'where_value', 'what']):
                    self._process_component_embeddings(memory_data)

                successful += 1

            except Exception as e:
                logger.error(f"Failed to process memory in batch: {e}")
                failed += 1

        # Save index after batch processing
        self.index.save()

        return {
            'status': 'success',
            'successful': successful,
            'failed': failed
        }

    def queue_memory_embedding(
        self,
        memory_id: str,
        content: str,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Queue a memory for embedding generation.

        Args:
            memory_id: Memory ID
            content: Memory content to embed
            priority: Task priority

        Returns:
            Task ID
        """
        return self.queue_handler.enqueue(
            task_type='embed_memory',
            payload={'memory_id': memory_id, 'content': content},
            priority=priority
        )

    def queue_component_embeddings(
        self,
        memory_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Queue component embeddings for a memory.

        Args:
            memory_data: Memory data including WHO, WHERE, WHAT fields
            priority: Task priority

        Returns:
            Task ID
        """
        return self.queue_handler.enqueue(
            task_type='embed_components',
            payload=memory_data,
            priority=priority
        )

    def queue_batch_embeddings(
        self,
        memories: List[Dict[str, Any]],
        priority: TaskPriority = TaskPriority.LOW
    ) -> str:
        """
        Queue a batch of memories for embedding.

        Args:
            memories: List of memory data
            priority: Task priority

        Returns:
            Task ID
        """
        return self.queue_handler.enqueue(
            task_type='batch_embed',
            payload={'memories': memories},
            priority=priority
        )

    def start_processing(self):
        """Start the queue processing workers."""
        self.queue_handler.start()
        logger.info("Started embedding queue processing")

    def stop_processing(self, timeout: float = 10.0):
        """
        Stop the queue processing workers.

        Args:
            timeout: Maximum time to wait for workers to finish
        """
        self.queue_handler.stop(timeout=timeout)
        logger.info("Stopped embedding queue processing")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the embedding queue.

        Returns:
            Status dictionary
        """
        return self.queue_handler.get_status()


# Singleton instance
_embedding_processor: Optional[EmbeddingQueueProcessor] = None


def get_embedding_processor() -> EmbeddingQueueProcessor:
    """Get or create the singleton embedding processor instance."""
    global _embedding_processor
    if _embedding_processor is None:
        _embedding_processor = EmbeddingQueueProcessor()
    return _embedding_processor