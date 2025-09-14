"""
Enhanced CLI module for JAM with API server capabilities
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import signal
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import click
import psutil
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.router import MemoryRouter
from agentic_memory.config_manager import ConfigManager
from agentic_memory.server.llama_server_manager import LLMServerManager, EmbeddingServerManager, get_server_manager, get_embedding_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages background processes for servers"""
    
    def __init__(self):
        self.processes = {}
        self.pid_dir = Path.home() / ".jam" / "pids"
        self.pid_dir.mkdir(parents=True, exist_ok=True)
    
    def start_process(self, name: str, command: List[str], cwd: Optional[Path] = None) -> bool:
        """Start a background process"""
        pid_file = self.pid_dir / f"{name}.pid"
        
        # Check if already running
        if self.is_running(name):
            logger.info(f"{name} is already running")
            return True
        
        try:
            # Start process
            if sys.platform == "win32":
                # Windows: Use CREATE_NEW_PROCESS_GROUP for background
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    cwd=cwd
                )
            else:
                # Unix: Use nohup equivalent
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                    cwd=cwd
                )
            
            # Save PID
            pid_file.write_text(str(process.pid))
            self.processes[name] = process
            logger.info(f"Started {name} with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return False
    
    def stop_process(self, name: str) -> bool:
        """Stop a background process"""
        pid_file = self.pid_dir / f"{name}.pid"
        
        if not pid_file.exists():
            logger.info(f"No PID file found for {name}")
            return False
        
        try:
            pid = int(pid_file.read_text())
            
            # Try to terminate process
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                # Force kill if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            
            # Clean up PID file
            pid_file.unlink()
            logger.info(f"Stopped {name} (PID {pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop {name}: {e}")
            return False
    
    def is_running(self, name: str) -> bool:
        """Check if a process is running"""
        pid_file = self.pid_dir / f"{name}.pid"
        
        if not pid_file.exists():
            return False
        
        try:
            pid = int(pid_file.read_text())
            # Check if process exists
            return psutil.pid_exists(pid)
        except:
            return False
    
    def get_status(self, name: str) -> Dict[str, Any]:
        """Get process status"""
        pid_file = self.pid_dir / f"{name}.pid"
        
        if not pid_file.exists():
            return {"running": False}
        
        try:
            pid = int(pid_file.read_text())
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                return {
                    "running": True,
                    "pid": pid,
                    "cpu_percent": proc.cpu_percent(),
                    "memory_mb": proc.memory_info().rss / 1024 / 1024,
                    "create_time": datetime.fromtimestamp(proc.create_time()).isoformat()
                }
        except:
            pass
        
        return {"running": False}


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """JAM - Journalistic Agent Memory CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.group()
def server():
    """Manage servers (LLM, API, Web)"""
    pass


@server.command()
@click.option('--all', 'start_all', is_flag=True, help='Start all servers')
@click.option('--llm', is_flag=True, help='Start LLM server')
@click.option('--embedding', is_flag=True, help='Start embedding server')
@click.option('--api', is_flag=True, help='Start API wrapper')
@click.option('--web', is_flag=True, help='Start web interface')
def start(start_all, llm, embedding, api, web):
    """Start servers"""
    process_manager = ProcessManager()

    if start_all:
        llm = embedding = api = web = True

    if not (llm or embedding or api or web):
        click.echo("Please specify which servers to start (--llm, --embedding, --api, --web, or --all)")
        return

    # Start LLM server
    if llm:
        click.echo("Starting LLM server...")
        manager = get_server_manager()
        if manager.ensure_running():
            click.echo("LLM server started")
        else:
            click.echo("Failed to start LLM server")
            return

    # Start embedding server
    if embedding:
        click.echo("Starting embedding server...")
        emb_manager = get_embedding_manager()
        if emb_manager.ensure_running():
            click.echo("Embedding server started")
        else:
            click.echo("Failed to start embedding server")
            return

    # Start web interface
    if web:
        click.echo("Starting web interface...")
        # Run in foreground
        from agentic_memory.server.flask_app import app
        click.echo("Starting web interface on port 5001")
        app.run(host="0.0.0.0", port=5001)


@server.command()
@click.option('--all', 'stop_all', is_flag=True, help='Stop all servers')
@click.option('--llm', is_flag=True, help='Stop LLM server')
@click.option('--embedding', is_flag=True, help='Stop embedding server')
@click.option('--api', is_flag=True, help='Stop API wrapper')
@click.option('--web', is_flag=True, help='Stop web interface')
def stop(stop_all, llm, embedding, api, web):
    """Stop servers"""
    process_manager = ProcessManager()

    if stop_all:
        llm = embedding = api = web = True

    if not (llm or embedding or api or web):
        click.echo("Please specify which servers to stop (--llm, --embedding, --api, --web, or --all)")
        return

    # Stop web interface
    if web:
        click.echo("Stopping web interface...")
        if process_manager.stop_process("jam-web"):
            click.echo("Web interface stopped")
        else:
            click.echo("Web interface was not running")

    # Stop API wrapper
    if api:
        click.echo("Stopping API wrapper...")
        if process_manager.stop_process("jam-api"):
            click.echo("API wrapper stopped")
        else:
            click.echo("API wrapper was not running")

    # Stop embedding server
    if embedding:
        click.echo("Stopping embedding server...")
        emb_manager = get_embedding_manager()
        if emb_manager.stop():
            click.echo("Embedding server stopped")
        else:
            click.echo("Embedding server was not running")

    # Stop LLM server
    if llm:
        click.echo("Stopping LLM server...")
        manager = get_server_manager()
        if manager.stop():
            click.echo("LLM server stopped")
        else:
            click.echo("LLM server was not running")


@server.command()
def status():
    """Check server status"""
    process_manager = ProcessManager()

    # LLM server status
    manager = get_server_manager()
    llm_status = manager.get_status()

    # Embedding server status
    emb_manager = get_embedding_manager()
    emb_status = emb_manager.get_status()

    click.echo("\nServer Status:")
    click.echo("-" * 40)

    # LLM Server
    if llm_status["running"]:
        click.echo(f"LLM Server: Running")
        click.echo(f"   URL: {llm_status.get('url')}")
        click.echo(f"   Model: {llm_status.get('model')}")
        if 'pid' in llm_status:
            click.echo(f"   PID: {llm_status['pid']}")
    else:
        click.echo("LLM Server: Stopped")

    # Embedding Server
    if emb_status["running"]:
        click.echo(f"Embedding Server: Running")
        click.echo(f"   Port: {emb_status.get('port')}")
        click.echo(f"   Model: {emb_status.get('model')}")
        if 'pid' in emb_status:
            click.echo(f"   PID: {emb_status['pid']}")
    else:
        click.echo("Embedding Server: Stopped")

    # API Wrapper
    api_status = process_manager.get_status("jam-api")
    if api_status["running"]:
        click.echo(f"API Wrapper: Running")
        click.echo(f"   PID: {api_status['pid']}")
        click.echo(f"   CPU: {api_status['cpu_percent']:.1f}%")
        click.echo(f"   Memory: {api_status['memory_mb']:.1f} MB")
    else:
        click.echo("API Wrapper: Stopped")

    # Web Interface
    web_status = process_manager.get_status("jam-web")
    if web_status["running"]:
        click.echo(f"Web Interface: Running")
        click.echo(f"   PID: {web_status['pid']}")
        click.echo(f"   CPU: {web_status['cpu_percent']:.1f}%")
        click.echo(f"   Memory: {web_status['memory_mb']:.1f} MB")
    else:
        click.echo("Web Interface: Stopped")

    click.echo("-" * 40)


@server.command()
def restart():
    """Restart all servers"""
    click.echo("Restarting servers...")

    # Stop all
    ctx = click.get_current_context()
    ctx.invoke(stop, stop_all=True)
    time.sleep(2)

    # Start all
    ctx.invoke(start, start_all=True)


@cli.group()
def memory():
    """Manage memory operations"""
    pass


@cli.group()
def document():
    """Document parsing and ingestion"""
    pass


@document.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--strategy', type=click.Choice(['semantic', 'paragraph', 'sentence']), default='semantic', help='Chunking strategy')
@click.option('--chunk-size', default=2000, help='Maximum chunk size')
@click.option('--overlap', default=200, help='Chunk overlap size')
@click.option('--dry-run', is_flag=True, help='Parse without ingesting into memory')
def parse(file_path, strategy, chunk_size, overlap, dry_run):
    """Parse a document and ingest into memory"""
    from agentic_memory.document_parser import DocumentParser, SemanticChunker, ParagraphChunker, SentenceChunker
    
    click.echo(f"Parsing document: {file_path}")
    
    # Select chunking strategy
    if strategy == 'sentence':
        chunker = SentenceChunker(chunk_size, overlap)
    elif strategy == 'paragraph':
        chunker = ParagraphChunker(chunk_size, overlap)
    else:
        chunker = SemanticChunker(chunk_size, overlap)
    
    # Parse document
    parser = DocumentParser(chunking_strategy=chunker)
    parsed_doc = parser.parse(file_path)
    
    if not parsed_doc.success:
        click.echo(f"❌ Failed to parse document")
        for error in parsed_doc.extraction_errors:
            click.echo(f"   Error: {error}")
        return
    
    # Display results
    click.echo(f"✅ Document parsed successfully")
    click.echo(f"   File type: {parsed_doc.file_type}")
    click.echo(f"   Chunks: {len(parsed_doc.chunks)}")
    click.echo(f"   Words: {parsed_doc.total_words:,}")
    click.echo(f"   Characters: {parsed_doc.total_chars:,}")
    
    if dry_run:
        click.echo("\n--dry-run specified, not ingesting into memory")
        click.echo("\nFirst 3 chunks preview:")
        for i, chunk in enumerate(parsed_doc.chunks[:3], 1):
            click.echo(f"\n--- Chunk {i}/{len(parsed_doc.chunks)} ---")
            preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            click.echo(preview)
        return
    
    # Ingest into memory
    click.echo("\nIngesting chunks into memory...")
    from agentic_memory.storage.sql_store import MemoryStore
    from agentic_memory.storage.faiss_index import FaissIndex

    config = ConfigManager()
    db_path = config.get_value('db_path') or 'data/amemory.sqlite3'
    index_path = config.get_value('index_path') or 'data/faiss.index'

    store = MemoryStore(db_path)
    index = FaissIndex(1024, index_path)  # Assuming 1024 dimensions
    router = MemoryRouter(store, index)
    
    memories_created = 0
    with click.progressbar(parsed_doc.chunks) as chunks:
        for chunk in chunks:
            try:
                memory_text = chunk.to_memory_text()
                memory = router.ingest(memory_text, metadata={
                    'source': 'document',
                    'file_name': Path(file_path).name,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks
                })
                if memory:
                    memories_created += 1
            except Exception as e:
                click.echo(f"\n⚠️  Error ingesting chunk {chunk.chunk_index}: {e}")
    
    click.echo(f"\n✅ Created {memories_created} memories from {len(parsed_doc.chunks)} chunks")


@document.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories')
@click.option('--extensions', '-e', multiple=True, help='File extensions to process (e.g., -e .txt -e .pdf)')
@click.option('--strategy', type=click.Choice(['semantic', 'paragraph', 'sentence']), default='semantic')
@click.option('--chunk-size', default=2000, help='Maximum chunk size')
@click.option('--overlap', default=200, help='Chunk overlap size')
def batch(directory, recursive, extensions, strategy, chunk_size, overlap):
    """Batch process documents in a directory"""
    from agentic_memory.document_parser import DocumentParser, SemanticChunker, ParagraphChunker, SentenceChunker
    
    click.echo(f"Processing directory: {directory}")
    
    # Select chunking strategy
    if strategy == 'sentence':
        chunker = SentenceChunker(chunk_size, overlap)
    elif strategy == 'paragraph':
        chunker = ParagraphChunker(chunk_size, overlap)
    else:
        chunker = SemanticChunker(chunk_size, overlap)
    
    parser = DocumentParser(chunking_strategy=chunker)
    
    # Parse directory
    parsed_docs = parser.parse_directory(directory, recursive=recursive, extensions=list(extensions) if extensions else None)
    
    if not parsed_docs:
        click.echo("No documents found to process")
        return
    
    # Display summary
    total_chunks = sum(len(doc.chunks) for doc in parsed_docs)
    successful = sum(1 for doc in parsed_docs if doc.success)
    
    click.echo(f"\nParsed {len(parsed_docs)} documents:")
    click.echo(f"   Successful: {successful}")
    click.echo(f"   Failed: {len(parsed_docs) - successful}")
    click.echo(f"   Total chunks: {total_chunks}")
    
    # Ingest into memory
    if click.confirm("Ingest all documents into memory?"):
        from agentic_memory.storage.sql_store import MemoryStore
        from agentic_memory.storage.faiss_index import FaissIndex

        config = ConfigManager()
        db_path = config.get_value('db_path') or 'data/amemory.sqlite3'
        index_path = config.get_value('index_path') or 'data/faiss.index'

        store = MemoryStore(db_path)
        index = FaissIndex(1024, index_path)  # Assuming 1024 dimensions
        router = MemoryRouter(store, index)
        
        total_memories = 0
        for doc in parsed_docs:
            if not doc.success:
                continue
            
            click.echo(f"\nProcessing: {Path(doc.file_path).name}")
            memories_created = 0
            
            with click.progressbar(doc.chunks) as chunks:
                for chunk in chunks:
                    try:
                        memory_text = chunk.to_memory_text()
                        memory = router.ingest(memory_text, metadata={
                            'source': 'document',
                            'file_name': Path(doc.file_path).name,
                            'chunk_index': chunk.chunk_index,
                            'total_chunks': chunk.total_chunks
                        })
                        if memory:
                            memories_created += 1
                    except Exception as e:
                        pass
            
            total_memories += memories_created
            click.echo(f"   Created {memories_created} memories")
        
        click.echo(f"\n✅ Total memories created: {total_memories}")


@document.command()
def formats():
    """List supported document formats"""
    from agentic_memory.document_parser import DocumentParser
    
    parser = DocumentParser()
    
    click.echo("\n📄 Supported Document Formats:")
    click.echo("-" * 40)
    
    format_categories = {
        "Text Files": ['.txt', '.md', '.markdown', '.log'],
        "Documents": ['.pdf', '.docx'],
        "Web": ['.html', '.htm', '.xml'],
        "Data": ['.json', '.csv'],
        "Code": ['.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs'],
        "Config": ['.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'],
        "Images (OCR)": ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    }
    
    for category, extensions in format_categories.items():
        click.echo(f"\n{category}:")
        for ext in extensions:
            if ext in parser.parsers:
                click.echo(f"   ✅ {ext}")
            else:
                click.echo(f"   ⚠️  {ext} (limited support)")
    
    click.echo("\n" + "-" * 40)
    click.echo("Note: OCR support requires pytesseract installation")
    click.echo("PDF support requires PyPDF2 installation")
    click.echo("DOCX support requires python-docx installation")


@memory.command()
@click.argument('text')
@click.option('--actor', default='user', help='Actor/source of the memory')
def add(text, actor):
    """Add a memory"""
    from agentic_memory.storage.sql_store import MemoryStore
    from agentic_memory.storage.faiss_index import FaissIndex

    config = ConfigManager()
    db_path = config.get_value('db_path') or 'data/amemory.sqlite3'
    index_path = config.get_value('index_path') or 'data/faiss.index'

    store = MemoryStore(db_path)
    index = FaissIndex(1024, index_path)  # Assuming 1024 dimensions
    router = MemoryRouter(store, index)
    
    result = router.ingest(text, metadata={'actor': actor})
    if result:
        click.echo(f"Memory added: {result.id}")
    else:
        click.echo("Failed to add memory")


@memory.command()
@click.argument('query')
@click.option('--limit', default=5, help='Number of results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text']), default='text')
def search(query, limit, output_format):
    """Search memories"""
    from agentic_memory.storage.sql_store import MemoryStore
    from agentic_memory.storage.faiss_index import FaissIndex

    config = ConfigManager()
    db_path = config.get_value('db_path') or 'data/amemory.sqlite3'
    index_path = config.get_value('index_path') or 'data/faiss.index'

    store = MemoryStore(db_path)
    index = FaissIndex(1024, index_path)  # Assuming 1024 dimensions
    router = MemoryRouter(store, index)
    
    results = router.retrieve(query, limit=limit)
    
    if output_format == 'json':
        click.echo(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. {result.what}")
            click.echo(f"   Who: {result.who}")
            click.echo(f"   When: {result.when}")
            if result.where:
                click.echo(f"   Where: {result.where}")
            click.echo(f"   Score: {result.score:.3f}")


@memory.command()
@click.option('--batch-size', default=10, help='Batch size for processing')
@click.option('--dry-run', is_flag=True, help='Show what would be fixed without making changes')
def fix_embeddings(batch_size, dry_run):
    """Regenerate missing or zero-norm embeddings"""
    from agentic_memory.storage.sql_store import MemoryStore
    from agentic_memory.embedding import get_llama_embedder
    from agentic_memory.server.llama_server_manager import get_embedding_manager
    import numpy as np

    config = ConfigManager()
    db_path = config.get_value('db_path') or 'data/amemory.sqlite3'

    click.echo("Checking for missing or broken embeddings...")
    click.echo("-" * 40)

    store = MemoryStore(db_path)

    # Find embeddings with zero norm
    with store.connect() as con:
        # Get all embeddings - check each one for zero norm
        cursor = con.execute("""
            SELECT e.memory_id, e.vector, m.raw_text
            FROM embeddings e
            LEFT JOIN memories m ON e.memory_id = m.memory_id
            WHERE e.vector IS NOT NULL
        """)

        zero_norm_ids = []
        missing_text_ids = []
        total_checked = 0

        while True:
            batch = cursor.fetchmany(500)
            if not batch:
                break

            for row in batch:
                total_checked += 1
                memory_id = row['memory_id']
                vector_data = row['vector']
                raw_text = row['raw_text']

                if vector_data:
                    vector = np.frombuffer(vector_data, dtype=np.float32)
                    norm = np.linalg.norm(vector)
                    if norm < 0.001:  # Near-zero norm
                        zero_norm_ids.append((memory_id, raw_text))

                if not raw_text:
                    missing_text_ids.append(memory_id)

        # Also find memories without any embeddings
        cursor = con.execute("""
            SELECT m.memory_id, m.raw_text
            FROM memories m
            LEFT JOIN embeddings e ON m.memory_id = e.memory_id
            WHERE e.memory_id IS NULL AND m.raw_text IS NOT NULL
        """)
        missing_embeddings = cursor.fetchall()

    click.echo(f"Checked {total_checked} embeddings")
    click.echo(f"Found {len(zero_norm_ids)} zero-norm embeddings")
    click.echo(f"Found {len(missing_embeddings)} memories without embeddings")
    click.echo(f"Found {len(missing_text_ids)} embeddings without source text")

    if not zero_norm_ids and not missing_embeddings:
        click.echo("\nNo broken embeddings found!")
        return

    if dry_run:
        click.echo("\n--dry-run specified, showing what would be fixed:")

        if zero_norm_ids[:10]:
            click.echo("\nZero-norm embeddings to regenerate:")
            for memory_id, text in zero_norm_ids[:10]:
                if text:
                    preview = text[:100] + "..." if len(text) > 100 else text
                else:
                    preview = "[No source text]"
                click.echo(f"  {memory_id}: {preview}")
            if len(zero_norm_ids) > 10:
                click.echo(f"  ... and {len(zero_norm_ids) - 10} more")

        if missing_embeddings[:10]:
            click.echo("\nMissing embeddings to generate:")
            for row in missing_embeddings[:10]:
                text = row['raw_text']
                preview = text[:100] + "..." if len(text) > 100 else text
                click.echo(f"  {row['memory_id']}: {preview}")
            if len(missing_embeddings) > 10:
                click.echo(f"  ... and {len(missing_embeddings) - 10} more")

        return

    # Start embedding server if needed
    click.echo("\nEnsuring embedding server is running...")
    emb_manager = get_embedding_manager()
    if not emb_manager.ensure_running():
        click.echo("Failed to start embedding server")
        return

    # Wait for server to be ready
    import time
    time.sleep(2)

    # Initialize embedder
    embedder = get_llama_embedder()

    # Process zero-norm embeddings
    if zero_norm_ids:
        click.echo(f"\nRegenerating {len(zero_norm_ids)} zero-norm embeddings...")
        fixed = 0
        failed = 0

        with click.progressbar(zero_norm_ids, label='Fixing zero-norm') as items:
            for memory_id, text in items:
                if not text:
                    failed += 1
                    continue

                try:
                    # Generate new embedding
                    embedding = embedder.encode(text, normalize_embeddings=True)
                    if embedding is not None and len(embedding) > 0:
                        # Update in database
                        embedding_bytes = embedding.astype(np.float32).tobytes()
                        with store.connect() as con:
                            con.execute(
                                "UPDATE embeddings SET vector = ? WHERE memory_id = ?",
                                (embedding_bytes, memory_id)
                            )
                        fixed += 1
                    else:
                        failed += 1
                except Exception as e:
                    click.echo(f"\nError fixing {memory_id}: {e}")
                    failed += 1

        click.echo(f"Fixed {fixed} zero-norm embeddings, {failed} failed")

    # Process missing embeddings
    if missing_embeddings:
        click.echo(f"\nGenerating {len(missing_embeddings)} missing embeddings...")
        created = 0
        failed = 0

        with click.progressbar(missing_embeddings, label='Creating embeddings') as items:
            for row in items:
                memory_id = row['memory_id']
                text = row['raw_text']

                if not text:
                    failed += 1
                    continue

                try:
                    # Generate embedding
                    embedding = embedder.encode(text, normalize_embeddings=True)
                    if embedding is not None and len(embedding) > 0:
                        # Insert into database
                        embedding_bytes = embedding.astype(np.float32).tobytes()
                        dim = len(embedding)

                        with store.connect() as con:
                            con.execute(
                                """INSERT INTO embeddings (memory_id, vector, dimension)
                                   VALUES (?, ?, ?)""",
                                (memory_id, embedding_bytes, dim)
                            )
                        created += 1
                    else:
                        failed += 1
                except Exception as e:
                    click.echo(f"\nError creating embedding for {memory_id}: {e}")
                    failed += 1

        click.echo(f"Created {created} new embeddings, {failed} failed")

    # Summary
    click.echo("\n" + "=" * 40)
    click.echo("SUMMARY")
    click.echo("=" * 40)
    click.echo(f"Zero-norm fixed: {fixed if zero_norm_ids else 0}")
    click.echo(f"Missing created: {created if missing_embeddings else 0}")
    click.echo(f"Total failures: {failed if 'failed' in locals() else 0}")

    if (zero_norm_ids or missing_embeddings) and not dry_run:
        click.echo("\nRun 'memory rebuild-index' to update the FAISS index with the new embeddings")


@memory.command()
def stats():
    """Show memory statistics"""
    from agentic_memory.storage.sql_store import MemoryStore
    from agentic_memory.storage.faiss_index import FaissIndex

    config = ConfigManager()
    db_path = config.get_value('db_path') or 'data/amemory.sqlite3'
    index_path = config.get_value('index_path') or 'data/faiss.index'

    store = MemoryStore(db_path)
    index = FaissIndex(1024, index_path)  # Assuming 1024 dimensions

    # Get statistics directly from store and index
    import os

    with store.connect() as con:
        cursor = con.cursor()

        # Total memories
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]

        # Date range
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM memories")
        date_range = cursor.fetchone()

        # Unique actors (from memories table)
        cursor.execute("SELECT COUNT(DISTINCT who_label) FROM memories")
        unique_actors = cursor.fetchone()[0]

    # File sizes
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0
    index_size_mb = os.path.getsize(index_path) / (1024 * 1024) if os.path.exists(index_path) else 0

    click.echo("\nMemory Statistics:")
    click.echo("-" * 40)
    click.echo(f"Total memories: {total_memories:,}")
    click.echo(f"Unique actors: {unique_actors:,}")
    click.echo(f"Date range: {date_range[0]} to {date_range[1]}")
    click.echo(f"Database size: {db_size_mb:.2f} MB")
    click.echo(f"Index size: {index_size_mb:.2f} MB")
    click.echo(f"Index vectors: {index.index.ntotal:,}")


@memory.command()
@click.option('--backup/--no-backup', default=True, help='Backup existing index before rebuilding')
@click.option('--batch-size', default=500, help='Batch size for processing embeddings')
@click.option('--verify', is_flag=True, help='Verify index after rebuilding')
@click.option('--verbose', is_flag=True, help='Show detailed information about skipped records')
def rebuild_index(backup, batch_size, verify, verbose):
    """Rebuild FAISS index from SQLite embeddings"""
    from agentic_memory.storage.sql_store import MemoryStore
    from agentic_memory.storage.faiss_index import FaissIndex
    from datetime import datetime
    import shutil

    config = ConfigManager()

    click.echo("Starting FAISS index rebuild...")
    click.echo("-" * 40)

    # Paths
    db_path = config.get_value('db_path') or 'data/amemory.sqlite3'
    index_path = config.get_value('index_path') or 'data/faiss.index'

    # Backup existing index if requested
    if backup and os.path.exists(index_path):
        backup_path = f"{index_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        click.echo(f"Backing up existing index to: {backup_path}")
        shutil.copy2(index_path, backup_path)

        # Also backup the map file if it exists
        if os.path.exists(f"{index_path}.map"):
            shutil.copy2(f"{index_path}.map", f"{backup_path}.map")

    # Initialize stores
    click.echo("Loading embeddings from SQLite...")
    store = MemoryStore(db_path)

    # Get embedding dimension from first embedding
    sample_embedding = store.get_sample_embedding()
    if sample_embedding is None:
        click.echo("Error: No embeddings found in SQLite database")
        return

    embedding_dim = len(sample_embedding)
    click.echo(f"Embedding dimension: {embedding_dim}")

    # Create new index
    click.echo("Creating new FAISS index...")

    # Remove old index files
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(f"{index_path}.map"):
        os.remove(f"{index_path}.map")

    # Create new index
    new_index = FaissIndex(embedding_dim, index_path)

    # Get all embeddings from SQLite
    click.echo("Fetching embeddings from database...")
    embeddings_data = store.get_all_embeddings_for_rebuild(batch_size=batch_size)

    total_embeddings = len(embeddings_data)
    click.echo(f"Found {total_embeddings:,} embeddings to process")

    # Track statistics
    processed = 0
    skipped = 0
    duplicates_removed = 0
    seen_hashes = {}  # Map hash to first memory_id that used it

    # Track details for verbose mode
    duplicate_groups = {}  # hash -> list of memory_ids
    zero_norm_ids = []
    error_ids = []

    # Process embeddings with progress bar
    with click.progressbar(embeddings_data, label='Rebuilding index', show_pos=verbose) as embeddings:
        for memory_id, embedding_data in embeddings:
            try:
                # Convert embedding to numpy array
                if isinstance(embedding_data, bytes):
                    embedding = np.frombuffer(embedding_data, dtype=np.float32)
                else:
                    embedding = np.array(embedding_data, dtype=np.float32)

                # Normalize the embedding first (before hashing for consistency)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                else:
                    skipped += 1
                    zero_norm_ids.append(memory_id)
                    continue

                # Check for duplicates (using a hash of the normalized embedding)
                embedding_hash = hash(embedding.tobytes())
                if embedding_hash in seen_hashes:
                    # This is a duplicate - skip it but record it
                    duplicates_removed += 1
                    if embedding_hash not in duplicate_groups:
                        duplicate_groups[embedding_hash] = [seen_hashes[embedding_hash]]
                    duplicate_groups[embedding_hash].append(memory_id)
                    continue

                # First time seeing this embedding - keep it
                seen_hashes[embedding_hash] = memory_id

                # Add to index
                new_index.add(memory_id, embedding)
                processed += 1

            except Exception as e:
                if verbose:
                    click.echo(f"\nError processing embedding for {memory_id}: {e}")
                error_ids.append((memory_id, str(e)))
                skipped += 1

    # Save the new index
    click.echo("\nSaving index...")
    new_index.save()

    # Display statistics
    click.echo("\nRebuild complete!")
    click.echo("-" * 40)
    click.echo(f"Total embeddings in SQLite: {total_embeddings:,}")
    click.echo(f"Successfully indexed: {processed:,}")
    click.echo(f"Duplicates removed: {duplicates_removed:,}")
    click.echo(f"Skipped (errors/zero vectors): {skipped:,}")
    click.echo(f"  - Zero norm vectors: {len(zero_norm_ids):,}")
    click.echo(f"  - Processing errors: {len(error_ids):,}")
    click.echo(f"Final index size: {new_index.index.ntotal:,} vectors")

    # Show verbose details if requested
    if verbose:
        click.echo("\n" + "=" * 60)
        click.echo("DETAILED ANALYSIS")
        click.echo("=" * 60)

        # Show duplicate groups
        if duplicate_groups:
            click.echo(f"\nDuplicate Groups ({len(duplicate_groups)} unique embeddings with duplicates):")
            # Sort by number of duplicates
            sorted_groups = sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (emb_hash, memory_ids) in enumerate(sorted_groups[:10], 1):
                click.echo(f"\n  Group {i}: {len(memory_ids)} identical embeddings")
                click.echo(f"    Kept: {memory_ids[0]}")
                skipped_str = ', '.join(memory_ids[1:5])
                if len(memory_ids) > 5:
                    skipped_str += f" ... and {len(memory_ids)-5} more"
                click.echo(f"    Skipped: {skipped_str}")

            if len(duplicate_groups) > 10:
                click.echo(f"\n  ... and {len(duplicate_groups)-10} more duplicate groups")

        # Show zero norm vectors
        if zero_norm_ids:
            click.echo(f"\nZero Norm Vectors ({len(zero_norm_ids)} total):")
            for i, memory_id in enumerate(zero_norm_ids[:10], 1):
                click.echo(f"  {i}. {memory_id}")
            if len(zero_norm_ids) > 10:
                click.echo(f"  ... and {len(zero_norm_ids)-10} more")

        # Show processing errors
        if error_ids:
            click.echo(f"\nProcessing Errors ({len(error_ids)} total):")
            for i, (memory_id, error) in enumerate(error_ids[:10], 1):
                click.echo(f"  {i}. {memory_id}: {error}")
            if len(error_ids) > 10:
                click.echo(f"  ... and {len(error_ids)-10} more")

        # Summary stats
        click.echo(f"\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60)
        click.echo(f"Unique embeddings: {len(seen_hashes):,}")
        click.echo(f"Duplicate groups: {len(duplicate_groups):,}")
        if duplicate_groups:
            max_dups = max(len(ids) for ids in duplicate_groups.values())
            click.echo(f"Largest duplicate group: {max_dups} identical embeddings")

    # Verify if requested
    if verify:
        click.echo("\nVerifying index...")

        # Sample some embeddings and search for them
        sample_size = min(10, processed)

        verification_passed = 0
        tests_performed = 0

        # Get random memory IDs to test
        for test_id in store.get_random_memory_ids(sample_size * 2):  # Get extra in case some are duplicates
            if tests_performed >= sample_size:
                break

            test_embedding = store.get_embedding_by_memory_id(test_id)
            if test_embedding:
                test_vec = np.frombuffer(test_embedding, dtype=np.float32)
                norm = np.linalg.norm(test_vec)
                if norm > 0:
                    test_vec = test_vec / norm

                    # Check if this exact vector should be in the index
                    # (it might have been deduplicated under a different memory_id)
                    embedding_hash = hash(test_vec.tobytes())

                    results = new_index.search(test_vec, 1)
                    if results and len(results) > 0 and results[0][1] > 0.99:  # High similarity threshold
                        verification_passed += 1

                    tests_performed += 1

        click.echo(f"Verification: {verification_passed}/{tests_performed} test queries successful")

    # Calculate space saved
    old_size = os.path.getsize(backup_path) if backup and os.path.exists(backup_path) else 0
    new_size = os.path.getsize(index_path)

    if old_size > 0:
        space_saved = (old_size - new_size) / (1024 * 1024)  # MB
        reduction_pct = (1 - new_size/old_size) * 100
        click.echo(f"\nSpace saved: {space_saved:.1f} MB ({reduction_pct:.1f}% reduction)")


@cli.group()
def api():
    """API client commands"""
    pass


@api.command()
@click.argument('prompt')
@click.option('--url', default='http://localhost:8001', help='API wrapper URL')
@click.option('--max-tokens', default=100, help='Maximum tokens to generate')
@click.option('--temperature', default=0.3, help='Temperature for sampling')
def complete(prompt, url, max_tokens, temperature):
    """Send completion request to API"""
    import requests
    
    try:
        response = requests.post(
            f"{url}/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and result['choices']:
            click.echo(result['choices'][0]['text'])
        else:
            click.echo(json.dumps(result, indent=2))
            
    except Exception as e:
        click.echo(f"Error: {e}")


@api.command()
@click.argument('message')
@click.option('--url', default='http://localhost:8001', help='API wrapper URL')
@click.option('--system', help='System prompt')
@click.option('--max-tokens', default=100, help='Maximum tokens to generate')
def chat(message, url, system, max_tokens):
    """Send chat request to API"""
    import requests
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    
    try:
        response = requests.post(
            f"{url}/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and result['choices']:
            click.echo(result['choices'][0]['message']['content'])
        else:
            click.echo(json.dumps(result, indent=2))
            
    except Exception as e:
        click.echo(f"Error: {e}")


@api.command()
@click.option('--url', default='http://localhost:8001', help='API wrapper URL')
def health(url):
    """Check API health"""
    import requests
    
    try:
        response = requests.get(f"{url}/health")
        response.raise_for_status()
        result = response.json()
        
        click.echo(f"Status: {result['status']}")
        click.echo(f"LLM Server: {result['llama_server']}")
        click.echo(f"Uptime: {result['api_wrapper']['uptime_seconds']:.0f} seconds")
        click.echo(f"Requests: {result['api_wrapper']['requests_processed']}")
        
    except Exception as e:
        click.echo(f"API is not responding: {e}")


@cli.command()
def config():
    """Show configuration"""
    config = ConfigManager()
    
    click.echo("\nConfiguration:")
    click.echo("-" * 40)
    click.echo(f"LLM Base URL: {config.llm_base_url}")
    click.echo(f"LLM Model: {config.llm_model}")
    click.echo(f"Context Window: {config.context_window}")
    click.echo(f"Database: {config.db_path}")
    click.echo(f"Index: {config.index_path}")
    click.echo("\nRetrieval Weights:")
    click.echo(f"  Semantic: {config.weights['semantic']}")
    click.echo(f"  Lexical: {config.weights['lexical']}")
    click.echo(f"  Recency: {config.weights['recency']}")
    click.echo(f"  Actor: {config.weights['actor']}")
    click.echo(f"  Spatial: {config.weights['spatial']}")
    click.echo(f"  Usage: {config.weights['usage']}")


@cli.command()
def version():
    """Show version information"""
    click.echo("JAM - Journalistic Agent Memory")
    click.echo("Version: 1.0.0")
    click.echo("Python: " + sys.version.split()[0])
    
    # Check component versions
    try:
        import agentic_memory
        click.echo(f"Agentic Memory: {getattr(agentic_memory, '__version__', 'unknown')}")
    except:
        pass
    
    try:
        import fastapi
        click.echo(f"FastAPI: {fastapi.__version__}")
    except:
        pass
    
    try:
        import uvicorn
        click.echo(f"Uvicorn: {uvicorn.__version__}")
    except:
        pass


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()
