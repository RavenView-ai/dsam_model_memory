"""
Memory Queue Handler for asynchronous processing of memories and embeddings.

This module provides a queue-based system for processing memories without blocking
the main application. It supports priority queues, retry logic, and concurrent processing.
"""

import threading
import queue
import time
import json
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for queue tasks."""
    CRITICAL = 1  # System-critical tasks
    HIGH = 2      # User-initiated tasks
    NORMAL = 3    # Regular background tasks
    LOW = 4       # Batch processing, backfills


class TaskStatus(Enum):
    """Status of a queued task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class QueueTask:
    """Represents a task in the processing queue."""
    task_id: str
    task_type: str  # 'embed', 'extract', 'component_embed', etc.
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None

    def __lt__(self, other):
        """Enable priority queue comparison."""
        return self.priority.value < other.priority.value


class MemoryQueueHandler:
    """
    Handles queuing and async processing of memory-related tasks.

    This handler manages a priority queue of tasks and processes them
    using worker threads, ensuring the main application remains responsive.
    """

    def __init__(
        self,
        max_workers: int = 2,
        max_queue_size: int = 1000,
        retry_delay: float = 1.0,
        task_timeout: float = 30.0
    ):
        """
        Initialize the queue handler.

        Args:
            max_workers: Maximum number of concurrent worker threads
            max_queue_size: Maximum number of tasks in queue
            retry_delay: Delay in seconds before retrying failed tasks
            task_timeout: Maximum time in seconds for a single task
        """
        self.max_workers = max_workers
        self.retry_delay = retry_delay
        self.task_timeout = task_timeout

        # Priority queue for tasks
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)

        # Track active tasks
        self.active_tasks: Dict[str, QueueTask] = {}
        self.completed_tasks: List[QueueTask] = []
        self.failed_tasks: List[QueueTask] = []

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.shutdown_event = threading.Event()

        # Task processors
        self.processors: Dict[str, Callable] = {}

        # Statistics
        self.stats = {
            'total_enqueued': 0,
            'total_processed': 0,
            'total_failed': 0,
            'total_retried': 0,
            'avg_processing_time': 0.0
        }

        # Lock for thread-safe operations
        self.lock = threading.Lock()

        logger.info(f"MemoryQueueHandler initialized with {max_workers} workers")

    def register_processor(self, task_type: str, processor: Callable):
        """
        Register a processor function for a specific task type.

        Args:
            task_type: Type of task to process
            processor: Function to process the task
        """
        self.processors[task_type] = processor
        logger.info(f"Registered processor for task type: {task_type}")

    def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None
    ) -> str:
        """
        Add a task to the processing queue.

        Args:
            task_type: Type of task to process
            payload: Task data
            priority: Task priority
            task_id: Optional custom task ID

        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"{task_type}_{int(time.time() * 1000)}"

        task = QueueTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority
        )

        try:
            # Use priority value as first element for priority queue
            self.task_queue.put((priority.value, task), timeout=1)

            with self.lock:
                self.stats['total_enqueued'] += 1

            logger.debug(f"Enqueued task {task_id} with priority {priority.name}")
            return task_id

        except queue.Full:
            logger.error(f"Queue is full, cannot enqueue task {task_id}")
            raise RuntimeError("Task queue is full")

    def start(self):
        """Start worker threads to process the queue."""
        if self.workers:
            logger.warning("Workers already started")
            return

        self.shutdown_event.clear()

        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"QueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {self.max_workers} worker threads")

    def stop(self, timeout: float = 10.0):
        """
        Stop all worker threads.

        Args:
            timeout: Maximum time to wait for workers to finish
        """
        if not self.workers:
            return

        logger.info("Shutting down queue handler...")
        self.shutdown_event.set()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)

        self.workers.clear()
        logger.info("Queue handler stopped")

    def _worker_loop(self):
        """Main loop for worker threads."""
        worker_name = threading.current_thread().name
        logger.info(f"{worker_name} started")

        while not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                priority_value, task = self.task_queue.get(timeout=1)

                # Process the task
                self._process_task(task)

                # Mark task as done
                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
                traceback.print_exc()

        logger.info(f"{worker_name} stopped")

    def _process_task(self, task: QueueTask):
        """
        Process a single task.

        Args:
            task: Task to process
        """
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()

        with self.lock:
            self.active_tasks[task.task_id] = task

        try:
            # Get processor for this task type
            processor = self.processors.get(task.task_type)
            if not processor:
                raise ValueError(f"No processor registered for task type: {task.task_type}")

            # Process with timeout
            result = self._execute_with_timeout(processor, task.payload)

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            with self.lock:
                self.active_tasks.pop(task.task_id, None)
                self.completed_tasks.append(task)
                self.stats['total_processed'] += 1

                # Update average processing time
                processing_time = (task.completed_at - task.started_at).total_seconds()
                current_avg = self.stats['avg_processing_time']
                total = self.stats['total_processed']
                self.stats['avg_processing_time'] = (
                    (current_avg * (total - 1) + processing_time) / total
                )

            logger.debug(f"Task {task.task_id} completed successfully")

        except Exception as e:
            task.error_message = str(e)
            self._handle_task_failure(task)

    def _execute_with_timeout(self, func: Callable, payload: Dict[str, Any]) -> Any:
        """
        Execute a function with timeout.

        Args:
            func: Function to execute
            payload: Function arguments

        Returns:
            Function result
        """
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(payload)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.task_timeout)

        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Task exceeded timeout of {self.task_timeout}s")

        if exception[0]:
            raise exception[0]

        return result[0]

    def _handle_task_failure(self, task: QueueTask):
        """
        Handle a failed task, potentially retrying.

        Args:
            task: Failed task
        """
        task.retry_count += 1

        with self.lock:
            self.active_tasks.pop(task.task_id, None)

        if task.retry_count < task.max_retries:
            # Retry the task
            task.status = TaskStatus.RETRYING
            logger.warning(
                f"Task {task.task_id} failed (attempt {task.retry_count}/{task.max_retries}): "
                f"{task.error_message}. Retrying..."
            )

            # Re-enqueue with delay
            time.sleep(self.retry_delay)
            self.task_queue.put((task.priority.value, task))

            with self.lock:
                self.stats['total_retried'] += 1
        else:
            # Max retries exceeded
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

            with self.lock:
                self.failed_tasks.append(task)
                self.stats['total_failed'] += 1

            logger.error(
                f"Task {task.task_id} failed permanently after {task.max_retries} attempts: "
                f"{task.error_message}"
            )

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the queue handler.

        Returns:
            Status dictionary
        """
        with self.lock:
            return {
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'workers': len(self.workers),
                'stats': self.stats.copy()
            }

    def get_task_status(self, task_id: str) -> Optional[QueueTask]:
        """
        Get status of a specific task.

        Args:
            task_id: Task ID to check

        Returns:
            Task object if found
        """
        with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]

            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return task

            # Check failed tasks
            for task in self.failed_tasks:
                if task.task_id == task_id:
                    return task

        return None

    def clear_completed_tasks(self, older_than_minutes: int = 60):
        """
        Clear old completed tasks from memory.

        Args:
            older_than_minutes: Clear tasks completed more than this many minutes ago
        """
        cutoff_time = datetime.now() - timedelta(minutes=older_than_minutes)

        with self.lock:
            self.completed_tasks = [
                task for task in self.completed_tasks
                if task.completed_at and task.completed_at > cutoff_time
            ]

            self.failed_tasks = [
                task for task in self.failed_tasks
                if task.completed_at and task.completed_at > cutoff_time
            ]

        logger.info(f"Cleared completed tasks older than {older_than_minutes} minutes")


# Singleton instance
_queue_handler: Optional[MemoryQueueHandler] = None


def get_queue_handler() -> MemoryQueueHandler:
    """Get or create the singleton queue handler instance."""
    global _queue_handler
    if _queue_handler is None:
        _queue_handler = MemoryQueueHandler()
    return _queue_handler