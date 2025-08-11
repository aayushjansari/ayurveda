#!/usr/bin/env python3
"""
🌿 Ayurveda Knowledge Bot - Complete Unified Application
Using latest LangChain and ChromaDB patterns from Context7 documentation

This file combines:
- Document loading and chunking
- Embeddings and vector database 
- RAG system with modern chains
- Streamlit chat interface

Run with: streamlit run chat_bot.py
"""

# Fix SQLite compatibility for ChromaDB on Streamlit Cloud
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import os
import time
import shutil
import traceback
import pickle
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Streamlit must be imported first
import streamlit as st

# Document processing imports
try:
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter
    )
    from langchain_core.documents import Document
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Document processing libraries not available: {e}")
    DOCUMENT_PROCESSING_AVAILABLE = False

# Modern LangChain imports for RAG
try:
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_community.llms import FakeListLLM
    MODERN_LANGCHAIN_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Modern LangChain libraries not available: {e}")
    MODERN_LANGCHAIN_AVAILABLE = False

# ChromaDB for advanced features
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    st.warning("⚠️ ChromaDB not available for advanced features")
    CHROMADB_AVAILABLE = False

# Optional experimental features
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

# =============================================================================
# 🚀 CONFIGURATION AND SETUP
# =============================================================================

# Deployment configuration
def get_deployment_config():
    """Get configuration based on deployment environment"""
    
    # Check if running on Streamlit Cloud
    is_streamlit_cloud = (
        "STREAMLIT_SHARING" in os.environ or 
        "STREAMLIT_CLOUD" in os.environ or 
        "streamlit.io" in os.environ.get("STREAMLIT_SERVER_ADDRESS", "") or
        os.path.exists("/mount/src")  # Streamlit Cloud mount point
    )
    
    if is_streamlit_cloud:
        return {
            "environment": "streamlit_cloud",
            "vector_db_path": "./chroma_db",
            "persistent_state_path": "./bot_state",
            "knowledge_base_folder": "./knowledge_base_files",
            "storage_type": "ephemeral"  # Will reset on restart
        }
    else:
        return {
            "environment": "local",
            "vector_db_path": "./chroma_db",
            "persistent_state_path": "./bot_state",
            "knowledge_base_folder": "./knowledge_base_files",
            "storage_type": "persistent"
        }

# Get deployment configuration
DEPLOYMENT_CONFIG = get_deployment_config()

# App configuration
APP_CONFIG = {
    "title": "🌿 Ayurveda Knowledge Bot",
    "description": "Your AI consultant for traditional Ayurvedic medicine",
    "version": "2.0.0 (Cloud Ready)",
    "knowledge_base_folder": DEPLOYMENT_CONFIG["knowledge_base_folder"],
    "vector_db_path": DEPLOYMENT_CONFIG["vector_db_path"],
    "persistent_state_path": DEPLOYMENT_CONFIG["persistent_state_path"],
    "max_chunk_size": 1000,
    "chunk_overlap": 200,
    "max_retrieval_docs": 5,
    "deployment_environment": DEPLOYMENT_CONFIG["environment"]
}

# Persistent storage paths
PERSISTENCE_PATHS = {
    "documents": f"{APP_CONFIG['persistent_state_path']}/documents.pkl",
    "chunks": f"{APP_CONFIG['persistent_state_path']}/chunks.pkl",
    "embedding_config": f"{APP_CONFIG['persistent_state_path']}/embedding_config.json",
    "processing_stats": f"{APP_CONFIG['persistent_state_path']}/processing_stats.json",
    "last_update": f"{APP_CONFIG['persistent_state_path']}/last_update.txt"
}

# Page configuration with cloud deployment optimizations
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-username/ayurveda-bot',
        'Report a bug': 'https://github.com/your-username/ayurveda-bot/issues',
        'About': """
        # 🌿 Ayurveda Knowledge Bot
        
        An AI-powered chatbot for traditional Ayurvedic medicine consultation.
        
        **Features:**
        - 📄 PDF document processing  
        - 🔍 Vector similarity search
        - 🤖 LangChain RAG system
        - 💬 Modern chat interface
        
        **Environment:** {env}
        
        Built with ❤️ using Streamlit, ChromaDB, and LangChain.
        
        *For educational purposes only. Consult qualified practitioners for medical advice.*
        """.format(env=APP_CONFIG["deployment_environment"].title())
    }
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .step-container {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .info-box {
        background-color: #e2f3ff;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-green { background-color: #28a745; }
    .status-yellow { background-color: #ffc107; }
    .status-red { background-color: #dc3545; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 💾 PERSISTENCE UTILITIES
# =============================================================================

def ensure_state_directory():
    """Ensure the persistent state directory exists"""
    Path(APP_CONFIG['persistent_state_path']).mkdir(parents=True, exist_ok=True)
    
    # Show cloud deployment notice if on ephemeral storage
    if DEPLOYMENT_CONFIG.get("storage_type") == "ephemeral":
        if "cloud_notice_shown" not in st.session_state:
            st.info("""
            🌐 **Cloud Deployment Active!** 
            
            Your Ayurveda bot is running on Streamlit Cloud. Data will be processed fresh on each restart, 
            ensuring you always get the latest version. Perfect for beta testing!
            
            ✨ **Features available:**
            - Full PDF processing pipeline
            - Vector search with ChromaDB  
            - AI-powered responses
            - Real-time chat interface
            """)
            st.session_state.cloud_notice_shown = True

def check_existing_state() -> Dict[str, bool]:
    """Check what components have been previously initialized"""
    state_status = {}
    
    # Check for saved documents and chunks
    state_status['documents_exist'] = os.path.exists(PERSISTENCE_PATHS['documents'])
    state_status['chunks_exist'] = os.path.exists(PERSISTENCE_PATHS['chunks'])
    state_status['embedding_config_exist'] = os.path.exists(PERSISTENCE_PATHS['embedding_config'])
    state_status['vector_db_exist'] = os.path.exists(APP_CONFIG['vector_db_path'])
    state_status['processing_stats_exist'] = os.path.exists(PERSISTENCE_PATHS['processing_stats'])
    
    # Check if all components exist
    state_status['complete_state_exist'] = all([
        state_status['documents_exist'],
        state_status['chunks_exist'], 
        state_status['embedding_config_exist'],
        state_status['vector_db_exist']
    ])
    
    # Check last update time
    if os.path.exists(PERSISTENCE_PATHS['last_update']):
        try:
            with open(PERSISTENCE_PATHS['last_update'], 'r') as f:
                state_status['last_update'] = f.read().strip()
        except:
            state_status['last_update'] = "Unknown"
    else:
        state_status['last_update'] = None
    
    return state_status

def get_knowledge_base_hash() -> str:
    """Get a hash of the knowledge base files to detect changes"""
    import hashlib
    
    kb_folder = Path(APP_CONFIG['knowledge_base_folder'])
    if not kb_folder.exists():
        return "no_folder"
    
    pdf_files = list(kb_folder.glob("*.pdf"))
    if not pdf_files:
        return "no_files"
    
    # Create hash from file names, sizes, and modification times
    file_info = []
    for pdf_file in sorted(pdf_files):
        stat = pdf_file.stat()
        file_info.append(f"{pdf_file.name}:{stat.st_size}:{stat.st_mtime}")
    
    combined_info = "|".join(file_info)
    return hashlib.md5(combined_info.encode()).hexdigest()

def save_documents_and_chunks(documents: List[Document], chunks: List[Document], stats: Dict[str, Any]):
    """Save processed documents and chunks to disk"""
    ensure_state_directory()
    
    try:
        # Save documents
        with open(PERSISTENCE_PATHS['documents'], 'wb') as f:
            pickle.dump(documents, f)
        
        # Save chunks
        with open(PERSISTENCE_PATHS['chunks'], 'wb') as f:
            pickle.dump(chunks, f)
        
        # Save processing stats
        with open(PERSISTENCE_PATHS['processing_stats'], 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save current knowledge base hash and timestamp
        kb_hash = get_knowledge_base_hash()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(PERSISTENCE_PATHS['last_update'], 'w') as f:
            f.write(f"{timestamp}|{kb_hash}")
        
        return True
        
    except Exception as e:
        st.error(f"Failed to save documents and chunks: {e}")
        return False

def load_documents_and_chunks() -> tuple:
    """Load processed documents and chunks from disk"""
    try:
        # Load documents
        with open(PERSISTENCE_PATHS['documents'], 'rb') as f:
            documents = pickle.load(f)
        
        # Load chunks
        with open(PERSISTENCE_PATHS['chunks'], 'rb') as f:
            chunks = pickle.load(f)
        
        # Load processing stats
        stats = {}
        if os.path.exists(PERSISTENCE_PATHS['processing_stats']):
            with open(PERSISTENCE_PATHS['processing_stats'], 'r') as f:
                stats = json.load(f)
        
        return documents, chunks, stats
        
    except Exception as e:
        st.error(f"Failed to load documents and chunks: {e}")
        return None, None, {}

def save_embedding_config(strategy_name: str, model_info: Dict[str, Any]):
    """Save embedding configuration"""
    ensure_state_directory()
    
    config = {
        "strategy_name": strategy_name,
        "model_info": model_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app_version": APP_CONFIG["version"]
    }
    
    try:
        with open(PERSISTENCE_PATHS['embedding_config'], 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save embedding config: {e}")
        return False

def load_embedding_config() -> Dict[str, Any]:
    """Load embedding configuration"""
    try:
        with open(PERSISTENCE_PATHS['embedding_config'], 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load embedding config: {e}")
        return {}

def check_knowledge_base_changes() -> bool:
    """Check if knowledge base has changed since last initialization"""
    current_hash = get_knowledge_base_hash()
    
    if not os.path.exists(PERSISTENCE_PATHS['last_update']):
        return True  # No previous state, assume changed
    
    try:
        with open(PERSISTENCE_PATHS['last_update'], 'r') as f:
            content = f.read().strip()
            if '|' in content:
                _, saved_hash = content.split('|', 1)
                return current_hash != saved_hash
            else:
                return True  # Old format, assume changed
    except:
        return True  # Error reading, assume changed

def clear_persistent_state():
    """Clear all persistent state"""
    try:
        # Remove state directory
        if os.path.exists(APP_CONFIG['persistent_state_path']):
            shutil.rmtree(APP_CONFIG['persistent_state_path'])
        
        # Remove vector database
        if os.path.exists(APP_CONFIG['vector_db_path']):
            shutil.rmtree(APP_CONFIG['vector_db_path'])
        
        return True
    except Exception as e:
        st.error(f"Failed to clear persistent state: {e}")
        return False

# =============================================================================
# 📂 STEP 1: DOCUMENT LOADING AND PROCESSING
# =============================================================================

class AdvancedDocumentProcessor:
    """Modern document processing using latest LangChain patterns"""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.documents = []
        self.chunks = []
        
    def load_documents(self) -> List[Document]:
        """Load documents with enhanced metadata following Context7 patterns"""
        if not DOCUMENT_PROCESSING_AVAILABLE:
            raise RuntimeError("Document processing libraries not available")
            
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")
        
        st.write("📂 **Loading PDF documents...**")
        
        # Modern DirectoryLoader pattern from Context7
        loader = DirectoryLoader(
            self.folder_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        
        with st.spinner("Loading documents..."):
            documents = loader.load()
        
        st.success(f"✅ Loaded {len(documents)} document pages")
        
        # Enhanced metadata processing
        for doc in documents:
            self._enhance_document_metadata(doc)
            doc.page_content = self._clean_text(doc.page_content)
        
        self.documents = documents
        return documents
    
    def _enhance_document_metadata(self, doc: Document):
        """Add Ayurveda-specific metadata"""
        source_path = doc.metadata.get('source', '')
        filename = os.path.basename(source_path).lower()
        
        # Document type classification
        if 'research' in filename or '809' in filename:
            doc.metadata.update({
                'doc_type': 'research_paper',
                'priority': 'high'
            })
        elif 'remedy' in filename or 'home' in filename:
            doc.metadata.update({
                'doc_type': 'remedies',
                'priority': 'high'
            })
        elif 'ayurvedic' in filename:
            doc.metadata.update({
                'doc_type': 'ayurvedic_guide',
                'priority': 'medium'
            })
        else:
            doc.metadata.update({
                'doc_type': 'general',
                'priority': 'medium'
            })
        
        # Content analysis
        content_lower = doc.page_content.lower()
        doc.metadata['contains_doshas'] = any(
            keyword in content_lower 
            for keyword in ['vata', 'pitta', 'kapha']
        )
        doc.metadata['contains_treatments'] = any(
            keyword in content_lower 
            for keyword in ['herb', 'remedy', 'treatment']
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        import re
        if not text or len(text.strip()) == 0:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\'\"\n]', '', text)
        
        # Remove very short lines
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
        return '\n'.join(lines)
    
    def create_chunks(self, strategy: str = 'auto') -> List[Document]:
        """Create chunks using modern LangChain patterns"""
        if not self.documents:
            raise RuntimeError("No documents loaded. Call load_documents() first.")
        
        st.write("📝 **Creating intelligent chunks...**")
        
        # Initialize text splitters following Context7 patterns
        splitters = {}
        
        # RecursiveCharacterTextSplitter (most recommended)
        splitters['recursive'] = RecursiveCharacterTextSplitter(
            chunk_size=APP_CONFIG["max_chunk_size"],
            chunk_overlap=APP_CONFIG["chunk_overlap"],
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
        
        # TikToken-based splitter for precise token control
        try:
            splitters['tiktoken'] = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=400,
                chunk_overlap=50
            )
        except Exception:
            pass
        
        # Semantic chunker (experimental)
        if SEMANTIC_CHUNKING_AVAILABLE:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
                splitters['semantic'] = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type="percentile"
                )
            except Exception:
                pass
        
        # Auto-select strategy
        if strategy == 'auto':
            if 'semantic' in splitters:
                strategy = 'semantic'
                st.info("🧠 Using semantic chunking (highest quality)")
            elif 'tiktoken' in splitters:
                strategy = 'tiktoken'
                st.info("🎯 Using TikToken-based chunking")
            else:
                strategy = 'recursive'
                st.info("🔄 Using recursive character chunking")
        
        # Create chunks
        splitter = splitters.get(strategy, splitters['recursive'])
        
        with st.spinner(f"Creating chunks with {strategy} strategy..."):
            chunks = splitter.split_documents(self.documents)
        
        # Enhanced chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_strategy': strategy,
                'chunk_length': len(chunk.page_content),
                'chunk_tokens': len(chunk.page_content.split())
            })
        
        self.chunks = chunks
        st.success(f"✅ Created {len(chunks)} intelligent chunks")
        
        return chunks
    
    def analyze_processing(self):
        """Display processing analytics"""
        if not self.documents or not self.chunks:
            return
        
        st.write("📊 **Processing Analytics**")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents", len(self.documents))
        
        with col2:
            st.metric("Chunks", len(self.chunks))
        
        with col3:
            total_chars = sum(len(doc.page_content) for doc in self.documents)
            st.metric("Total Characters", f"{total_chars:,}")
        
        with col4:
            avg_chunk_size = sum(len(chunk.page_content) for chunk in self.chunks) // len(self.chunks)
            st.metric("Avg Chunk Size", f"{avg_chunk_size} chars")
        
        # Document type distribution
        doc_types = {}
        for doc in self.documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        st.write("**Document Types:**")
        for doc_type, count in doc_types.items():
            st.write(f"- {doc_type}: {count} documents")
    
    def save_state(self):
        """Save current processing state to disk"""
        if not self.documents or not self.chunks:
            return False
        
        # Prepare stats for saving
        stats = {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "total_chars": sum(len(doc.page_content) for doc in self.documents),
            "avg_chunk_size": sum(len(chunk.page_content) for chunk in self.chunks) // len(self.chunks),
            "doc_types": {},
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Document type distribution
        for doc in self.documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            stats["doc_types"][doc_type] = stats["doc_types"].get(doc_type, 0) + 1
        
        return save_documents_and_chunks(self.documents, self.chunks, stats)
    
    def load_state(self):
        """Load processing state from disk"""
        documents, chunks, stats = load_documents_and_chunks()
        
        if documents is not None and chunks is not None:
            self.documents = documents
            self.chunks = chunks
            st.success(f"✅ Loaded {len(documents)} documents and {len(chunks)} chunks from saved state")
            return True
        
        return False

# =============================================================================
# 🧠 STEP 2: EMBEDDINGS AND VECTOR DATABASE
# =============================================================================

class ModernEmbeddingSystem:
    """Modern embedding system using latest Context7 patterns"""
    
    def __init__(self):
        self.embedder = None
        self.vector_store = None
        self.strategy_name = None
        
    def initialize_embeddings(self) -> tuple:
        """Initialize embeddings with priority fallback system"""
        st.write("🧠 **Initializing embeddings...**")
        
        strategies = []
        
        # Strategy 1: OpenAI (if API key available)
        if os.getenv("OPENAI_API_KEY") and MODERN_LANGCHAIN_AVAILABLE:
            try:
                embedder = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    dimensions=1536
                )
                # Test the embedder
                embedder.embed_query("test")
                strategies.append(('openai', embedder, "OpenAI text-embedding-3-small"))
            except Exception as e:
                st.warning(f"OpenAI embeddings failed: {e}")
        
        # Strategy 2: HuggingFace
        try:
            embedder = HuggingFaceEmbeddings(
                model_name='all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu', 'trust_remote_code': False},
                encode_kwargs={'normalize_embeddings': True}
            )
            strategies.append(('huggingface', embedder, "HuggingFace all-MiniLM-L6-v2"))
        except Exception as e:
            st.warning(f"HuggingFace embeddings failed: {e}")
        
        if not strategies:
            raise RuntimeError("No embedding strategies available")
        
        # Use the first available strategy
        self.strategy_name, self.embedder, description = strategies[0]
        st.success(f"✅ Using {description}")
        
        # Save embedding configuration
        model_info = {
            "strategy": self.strategy_name,
            "description": description,
            "model_name": getattr(self.embedder, 'model_name', 'unknown'),
            "available_strategies": [s[0] for s in strategies]
        }
        save_embedding_config(self.strategy_name, model_info)
        
        return self.embedder, self.strategy_name
    
    def load_embeddings_from_config(self) -> tuple:
        """Load embeddings based on saved configuration"""
        config = load_embedding_config()
        
        if not config:
            return None, None
        
        strategy_name = config.get("strategy_name")
        
        st.write(f"🧠 **Loading embeddings from saved config...**")
        st.info(f"Previous strategy: {config.get('model_info', {}).get('description', 'Unknown')}")
        
        # For cloud deployment, prioritize HuggingFace for compatibility
        if DEPLOYMENT_CONFIG.get("environment") == "streamlit_cloud":
            st.info("🌐 Cloud deployment detected - using HuggingFace embeddings for compatibility")
            try:
                embedder = HuggingFaceEmbeddings(
                    model_name='all-MiniLM-L6-v2',
                    model_kwargs={'device': 'cpu', 'trust_remote_code': False},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.embedder = embedder
                self.strategy_name = 'huggingface'
                st.success("✅ Using HuggingFace embeddings for cloud compatibility")
                return embedder, 'huggingface'
            except Exception as e:
                st.warning(f"HuggingFace embeddings failed: {e}")
        
        # Try to recreate the same embedder for local development
        if strategy_name == 'openai' and os.getenv("OPENAI_API_KEY") and MODERN_LANGCHAIN_AVAILABLE:
            try:
                embedder = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    dimensions=1536
                )
                embedder.embed_query("test")  # Test
                self.embedder = embedder
                self.strategy_name = strategy_name
                st.success("✅ Reloaded OpenAI embeddings")
                return embedder, strategy_name
            except Exception as e:
                st.warning(f"Failed to reload OpenAI embeddings: {e}")
        
        if strategy_name == 'huggingface':
            try:
                embedder = HuggingFaceEmbeddings(
                    model_name='all-MiniLM-L6-v2',
                    model_kwargs={'device': 'cpu', 'trust_remote_code': False},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.embedder = embedder
                self.strategy_name = strategy_name
                st.success("✅ Reloaded HuggingFace embeddings")
                return embedder, strategy_name
            except Exception as e:
                st.warning(f"Failed to reload HuggingFace embeddings: {e}")
        
        # Fallback to normal initialization
        st.warning("⚠️ Could not reload saved embeddings, using HuggingFace fallback")
        return self.initialize_embeddings()
    
    def create_vector_store(self, chunks: List[Document]) -> Any:
        """Create vector store using modern ChromaDB patterns"""
        if not self.embedder:
            raise RuntimeError("Embeddings not initialized")
        
        st.write("💾 **Creating vector database...**")
        
        # Setup paths
        persist_directory = APP_CONFIG["vector_db_path"]
        collection_name = f"ayurveda_docs_{self.strategy_name}"
        
        # Handle existing database conflicts
        if os.path.exists(persist_directory):
            st.warning("🗂️ Found existing database - recreating with new settings")
            try:
                shutil.rmtree(persist_directory)
            except Exception as e:
                st.error(f"Failed to clear existing database: {e}")
                # Try with unique name
                import time as time_module
                persist_directory = f"{persist_directory}_{int(time_module.time())}"
        
        # Enhanced document processing
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            import time as time_module
            enhanced_metadata = {
                **chunk.metadata,
                'document_id': f"ayurveda_doc_{i}",
                'content_length': len(chunk.page_content),
                'embedding_strategy': self.strategy_name,
                'processed_at': str(int(time_module.time())),
            }
            
            # Ayurveda-specific enhancements
            content_lower = chunk.page_content.lower()
            if any(dosha in content_lower for dosha in ['vata', 'pitta', 'kapha']):
                enhanced_metadata['ayurveda_category'] = 'constitutional'
            if any(treatment in content_lower for treatment in ['herb', 'remedy', 'treatment']):
                enhanced_metadata['ayurveda_category'] = 'therapeutic'
            
            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata=enhanced_metadata
            )
            processed_chunks.append(enhanced_chunk)
        
        # Create vector store using latest LangChain patterns
        with st.spinner("Creating vector database..."):
            try:
                self.vector_store = Chroma.from_documents(
                    documents=processed_chunks,
                    embedding=self.embedder,
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )
                
                st.success(f"✅ Vector database created with {len(processed_chunks)} documents")
                
            except Exception as e:
                st.error(f"Vector store creation failed: {e}")
                raise
        
        return self.vector_store
    
    def load_existing_vector_store(self) -> Any:
        """Load existing vector store from disk"""
        if not self.embedder:
            raise RuntimeError("Embeddings not initialized")
        
        persist_directory = APP_CONFIG["vector_db_path"]
        collection_name = f"ayurveda_docs_{self.strategy_name}"
        
        if not os.path.exists(persist_directory):
            return None
        
        st.write("💾 **Loading existing vector database...**")
        
        try:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedder,
                persist_directory=persist_directory
            )
            
            # Test the loaded vector store
            doc_count = self.vector_store._collection.count()
            st.success(f"✅ Loaded vector database with {doc_count} documents")
            
            return self.vector_store
            
        except Exception as e:
            st.warning(f"⚠️ Failed to load existing vector store: {e}")
            
            # For cloud deployment, automatically rebuild if corrupted
            if DEPLOYMENT_CONFIG.get("environment") == "streamlit_cloud":
                st.info("🔄 Cloud deployment detected - will rebuild vector database from saved chunks")
                try:
                    # Clear corrupted database
                    if os.path.exists(persist_directory):
                        import shutil
                        shutil.rmtree(persist_directory)
                        st.info("🗑️ Cleared corrupted vector database")
                except Exception:
                    pass
            
            return None
    
    def test_vector_store(self):
        """Test vector store functionality"""
        if not self.vector_store:
            return False
        
        try:
            # Test similarity search
            test_results = self.vector_store.similarity_search(
                "ayurveda doshas vata pitta kapha", k=3
            )
            
            if test_results:
                st.success(f"✅ Vector store test passed - found {len(test_results)} results")
                
                # Show preview
                first_result = test_results[0]
                with st.expander("📝 Test Result Preview"):
                    st.write(f"**Content:** {first_result.page_content[:200]}...")
                    st.write(f"**Metadata:** {first_result.metadata}")
                
                return True
            else:
                st.error("❌ Vector store test failed - no results found")
                return False
                
        except Exception as e:
            st.error(f"❌ Vector store test error: {e}")
            return False

# =============================================================================
# 🤖 STEP 3: RAG SYSTEM WITH MODERN CHAINS
# =============================================================================

class ModernRAGSystem:
    """Modern RAG system using latest LangChain patterns from Context7"""
    
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = None
        self.retriever = None
        self.rag_chain = None
        self.qa_chain = None
        
    def initialize_llm(self):
        """Initialize LLM with fallback strategy"""
        st.write("🤖 **Initializing AI language model...**")
        
        # Priority 1: OpenAI
        if os.getenv("OPENAI_API_KEY") and MODERN_LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-2024-08-06",
                    temperature=0.7,
                    max_tokens=512,
                    streaming=True
                )
                st.success("✅ Using OpenAI GPT-4o-2024-08-06")
                return True
            except Exception as e:
                st.warning(f"OpenAI setup failed: {e}")
        
        # Priority 2: Local Ollama
        try:
            from langchain_community.llms import Ollama
            test_llm = Ollama(model="llama3.2:3b", temperature=0.7)
            test_llm.invoke("test")
            self.llm = test_llm
            st.success("✅ Using Ollama (Local LLM)")
            return True
        except Exception as e:
            st.warning(f"Ollama not available: {e}")
        
        # Fallback: Enhanced mock LLM
        st.info("🔄 Using enhanced Ayurveda knowledge LLM (demonstration mode)")
        self.llm = FakeListLLM(responses=[
            "Based on Ayurvedic principles, the three doshas (Vata, Pitta, Kapha) represent different constitutional types and physiological functions. Vata governs movement and circulation, Pitta controls metabolism and digestion, while Kapha manages structure and immunity.",
            
            "According to traditional Ayurveda, this condition often indicates a dosha imbalance. Treatment typically involves personalized approaches including specific dietary recommendations, herbal formulations, lifestyle modifications, and cleansing practices (Panchakarma) based on your unique constitution.",
            
            "Ayurvedic remedies emphasize natural healing through herbs, oils, and holistic practices. Common approaches include turmeric for inflammation, ginger for digestion, ashwagandha for stress, and pranayama breathing techniques for overall balance.",
            
            "This topic is addressed in classical Ayurvedic texts like Charaka Samhita and Sushruta Samhita. The approach focuses on identifying the root cause (mool karan), restoring balance through natural methods, and strengthening the body's inherent healing capacity (Ojas).",
            
            "From an Ayurvedic perspective, prevention is as important as treatment. This involves understanding your Prakriti (natural constitution), following seasonal routines (Ritucharya), maintaining proper digestion (Agni), and cultivating mental well-being through meditation and yoga."
        ])
        return True
    
    def create_retriever(self):
        """Create retriever with modern configuration"""
        if not self.vector_store:
            raise RuntimeError("Vector store not available")
        
        # Modern retriever configuration following Context7 patterns
        self.retriever = self.vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={
                'k': APP_CONFIG["max_retrieval_docs"]
            }
        )
        
        st.success("✅ Retriever configured")
        return self.retriever
    
    def create_rag_chains(self):
        """Create modern RAG chains using latest Context7 patterns"""
        if not self.llm or not self.retriever:
            raise RuntimeError("LLM or retriever not initialized")
        
        st.write("🔗 **Creating RAG chains...**")
        
        # Modern RAG chain with custom Ayurveda prompt
        ayurveda_prompt = ChatPromptTemplate.from_template("""
You are an expert Ayurvedic consultant with deep knowledge of traditional Indian medicine. 
Use the provided context to answer questions about Ayurveda accurately and helpfully.

Context: {context}

Question: {question}

Instructions:
- Provide accurate information based on the context
- Mention relevant doshas (Vata, Pitta, Kapha) when applicable
- Include practical recommendations when appropriate
- Always suggest consulting a qualified Ayurvedic practitioner for personalized advice
- If the context doesn't contain relevant information, say so clearly

Answer:""")
        
        # Document formatter function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Modern LCEL chain composition following Context7 patterns
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | ayurveda_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Legacy QA chain for compatibility
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            verbose=False
        )
        
        st.success("✅ RAG chains created successfully")
        return True
    
    def test_rag_system(self):
        """Test RAG system with sample queries"""
        if not self.rag_chain:
            return False
        
        st.write("🧪 **Testing RAG system...**")
        
        test_questions = [
            "What are the three doshas in Ayurveda?",
            "How to balance Vata dosha?",
            "What are common Ayurvedic treatments?"
        ]
        
        successful_tests = 0
        
        for question in test_questions:
            try:
                with st.spinner(f"Testing: {question}"):
                    # Test modern chain
                    import time as time_module
                    start_time = time_module.time()
                    response = self.rag_chain.invoke(question)
                    response_time = time_module.time() - start_time
                    
                    # Get source documents
                    source_docs = self.retriever.invoke(question)
                    
                    if response and len(response) > 20:
                        successful_tests += 1
                        st.success(f"✅ Test passed ({response_time:.2f}s, {len(source_docs)} sources)")
                    else:
                        st.warning(f"⚠️ Weak response for: {question}")
                        
            except Exception as e:
                st.error(f"❌ Test failed for '{question}': {e}")
        
        success_rate = (successful_tests / len(test_questions)) * 100
        
        if success_rate >= 66:
            st.success(f"🎉 RAG system ready! Success rate: {success_rate:.1f}%")
            return True
        else:
            st.error(f"❌ RAG system needs attention. Success rate: {success_rate:.1f}%")
            return False
    
    def query(self, question: str, use_modern_chain: bool = True) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            if use_modern_chain and self.rag_chain:
                # Use modern chain
                import time as time_module
                start_time = time_module.time()
                response = self.rag_chain.invoke(question)
                response_time = time_module.time() - start_time
                
                # Get source documents separately
                source_docs = self.retriever.invoke(question)
                
                return {
                    "result": response,
                    "source_documents": source_docs,
                    "response_time": response_time,
                    "chain_type": "modern",
                    "error": False
                }
            else:
                # Use legacy chain
                import time as time_module
                start_time = time_module.time()
                response = self.qa_chain.invoke({"query": question})
                response_time = time_module.time() - start_time
                
                return {
                    **response,
                    "response_time": response_time,
                    "chain_type": "legacy",
                    "error": False
                }
                
        except Exception as e:
            return {
                "result": f"Error processing query: {str(e)}",
                "source_documents": [],
                "response_time": 0,
                "error": True
            }

# =============================================================================
# 💬 STEP 4: STREAMLIT CHAT INTERFACE
# =============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if "pipeline_initialized" not in st.session_state:
        st.session_state.pipeline_initialized = False
    
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = None
    
    if "embedding_system" not in st.session_state:
        st.session_state.embedding_system = None
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Add welcome message
        welcome_msg = """🙏 **Welcome to the Ayurveda Knowledge Bot!**

I'm your AI consultant specializing in traditional Ayurvedic medicine. I can help you with:

🔹 **Constitutional Analysis** - Understanding Vata, Pitta, and Kapha doshas  
🔹 **Natural Remedies** - Herbal treatments and therapeutic practices  
🔹 **Lifestyle Guidance** - Diet, routines, and wellness practices  
🔹 **Classical Wisdom** - Insights from ancient Ayurvedic texts  

*Please note: This information is for educational purposes. Always consult qualified Ayurvedic practitioners for personalized treatment.*"""
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_msg,
            "sources": [],
            "response_time": 0
        })

def run_fast_initialization():
    """Run fast initialization using saved state"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load Documents and Chunks
        status_text.text("📂 Step 1/4: Loading saved documents and chunks...")
        progress_bar.progress(20)
        
        processor = AdvancedDocumentProcessor(APP_CONFIG["knowledge_base_folder"])
        if not processor.load_state():
            st.error("❌ Failed to load saved documents")
            return False
        
        processor.analyze_processing()
        st.session_state.document_processor = processor
        progress_bar.progress(40)
        
        # Step 2: Load Embedding System
        status_text.text("🧠 Step 2/4: Loading embedding configuration...")
        
        embedding_system = ModernEmbeddingSystem()
        embedder, strategy = embedding_system.load_embeddings_from_config()
        
        if not embedder:
            st.error("❌ Failed to load embeddings")
            return False
        
        progress_bar.progress(60)
        
        # Step 3: Load Vector Store
        status_text.text("💾 Step 3/4: Loading vector database...")
        
        vector_store = embedding_system.load_existing_vector_store()
        if not vector_store:
            st.warning("⚠️ Vector database needs rebuilding")
            status_text.text("🔄 Rebuilding vector database from saved chunks...")
            progress_bar.progress(65)
            
            # Rebuild from saved chunks
            try:
                vector_store = embedding_system.create_vector_store(processor.chunks)
                if not vector_store:
                    st.error("❌ Failed to rebuild vector database")
                    return False
                st.success("✅ Vector database rebuilt successfully")
            except Exception as e:
                st.error(f"❌ Failed to rebuild vector database: {e}")
                return False
        
        if not embedding_system.test_vector_store():
            st.warning("⚠️ Vector store test failed but proceeding...")
        
        st.session_state.embedding_system = embedding_system
        progress_bar.progress(80)
        
        # Step 4: Initialize RAG System
        status_text.text("🤖 Step 4/4: Setting up AI system...")
        
        rag_system = ModernRAGSystem(vector_store, embedder)
        
        if not rag_system.initialize_llm():
            st.error("❌ LLM initialization failed")
            return False
        
        rag_system.create_retriever()
        rag_system.create_rag_chains()
        progress_bar.progress(95)
        
        if not rag_system.test_rag_system():
            st.warning("⚠️ RAG system tests showed issues but proceeding...")
        
        st.session_state.rag_system = rag_system
        progress_bar.progress(100)
        
        status_text.text("🚀 Fast initialization complete!")
        st.session_state.pipeline_initialized = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Fast initialization failed: {e}")
        st.exception(e)
        return False

def run_initialization_pipeline():
    """Run the complete initialization pipeline"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Document Processing
        status_text.text("📂 Step 1/4: Loading and processing documents...")
        progress_bar.progress(10)
        
        processor = AdvancedDocumentProcessor(APP_CONFIG["knowledge_base_folder"])
        
        if not os.path.exists(APP_CONFIG["knowledge_base_folder"]):
            st.error(f"❌ Knowledge base folder not found: {APP_CONFIG['knowledge_base_folder']}")
            st.info("Please create the folder and add PDF documents.")
            return False
        
        documents = processor.load_documents()
        progress_bar.progress(25)
        
        chunks = processor.create_chunks()
        progress_bar.progress(40)
        
        processor.analyze_processing()
        st.session_state.document_processor = processor
        
        # Step 2: Embeddings and Vector Store
        status_text.text("🧠 Step 2/4: Creating embeddings and vector database...")
        progress_bar.progress(50)
        
        embedding_system = ModernEmbeddingSystem()
        embedder, strategy = embedding_system.initialize_embeddings()
        progress_bar.progress(60)
        
        vector_store = embedding_system.create_vector_store(chunks)
        progress_bar.progress(75)
        
        if not embedding_system.test_vector_store():
            st.error("❌ Vector store test failed")
            return False
        
        st.session_state.embedding_system = embedding_system
        
        # Step 3: RAG System
        status_text.text("🤖 Step 3/4: Setting up AI reasoning system...")
        progress_bar.progress(80)
        
        rag_system = ModernRAGSystem(vector_store, embedder)
        
        if not rag_system.initialize_llm():
            st.error("❌ LLM initialization failed")
            return False
        
        progress_bar.progress(85)
        
        rag_system.create_retriever()
        rag_system.create_rag_chains()
        progress_bar.progress(95)
        
        # Step 4: Final Testing
        status_text.text("🧪 Step 4/4: Testing complete system...")
        
        if not rag_system.test_rag_system():
            st.warning("⚠️ RAG system tests showed issues but proceeding...")
        
        st.session_state.rag_system = rag_system
        progress_bar.progress(100)
        
        status_text.text("💾 Saving system state for fast reuse...")
        
        # Save the processed state for future fast loading
        if st.session_state.document_processor:
            st.session_state.document_processor.save_state()
        
        status_text.text("🎉 System initialization complete!")
        st.session_state.pipeline_initialized = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        st.exception(e)
        return False

def display_chat_interface():
    """Display the main chat interface"""
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                sources = message["sources"]
                if len(sources) > 0:
                    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                        for i, source in enumerate(sources[:3], 1):
                            content = source.page_content if hasattr(source, 'page_content') else str(source)
                            preview = content[:200] + "..." if len(content) > 200 else content
                            
                            metadata = ""
                            if hasattr(source, 'metadata') and source.metadata:
                                doc_type = source.metadata.get('doc_type', 'unknown')
                                chunk_id = source.metadata.get('chunk_id', 'unknown')
                                metadata = f"*Type: {doc_type}, Chunk: {chunk_id}*"
                            
                            st.markdown(f"**Source {i}:** {preview}\n\n{metadata}")
            
            # Show response time
            if message["role"] == "assistant" and message.get("response_time", 0) > 0:
                st.caption(f"⏱️ Response time: {message['response_time']:.2f}s")
    
    # Chat input
    if prompt := st.chat_input("Ask about Ayurveda (e.g., 'What are the three doshas?')"):
        if not st.session_state.pipeline_initialized:
            st.error("❌ Please initialize the system first using the sidebar.")
            return
        
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "sources": [],
            "response_time": 0
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching Ayurvedic knowledge base..."):
                response = st.session_state.rag_system.query(prompt)
            
            if not response.get("error", True):
                answer = response.get("result", "I couldn't find relevant information.")
                sources = response.get("source_documents", [])
                response_time = response.get("response_time", 0)
                
                st.markdown(answer)
                
                # Display sources
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                        for i, source in enumerate(sources[:3], 1):
                            content = source.page_content if hasattr(source, 'page_content') else str(source)
                            preview = content[:200] + "..." if len(content) > 200 else content
                            
                            metadata = ""
                            if hasattr(source, 'metadata') and source.metadata:
                                doc_type = source.metadata.get('doc_type', 'unknown')
                                chunk_id = source.metadata.get('chunk_id', 'unknown')
                                metadata = f"*Type: {doc_type}, Chunk: {chunk_id}*"
                            
                            st.markdown(f"**Source {i}:** {preview}\n\n{metadata}")
                
                st.caption(f"⏱️ Response time: {response_time:.2f}s")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "response_time": response_time
                })
            else:
                error_msg = response.get("result", "Unknown error occurred")
                st.error(error_msg)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "response_time": 0
                })

def display_sidebar():
    """Display sidebar with controls and information"""
    st.sidebar.markdown(f"# {APP_CONFIG['title']}")
    st.sidebar.markdown(f"*Version {APP_CONFIG['version']}*")
    
    # System status
    st.sidebar.markdown("### 🔧 System Status")
    
    if st.session_state.pipeline_initialized:
        st.sidebar.success("🟢 **System Ready**")
        
        # System info
        if st.session_state.document_processor:
            docs_count = len(st.session_state.document_processor.documents)
            chunks_count = len(st.session_state.document_processor.chunks)
            st.sidebar.markdown(f"📄 **Documents:** {docs_count}")
            st.sidebar.markdown(f"🧩 **Chunks:** {chunks_count}")
        
        if st.session_state.embedding_system:
            strategy = st.session_state.embedding_system.strategy_name
            st.sidebar.markdown(f"🧠 **Embeddings:** {strategy}")
        
        if st.session_state.rag_system:
            llm_type = "OpenAI" if hasattr(st.session_state.rag_system.llm, 'model_name') else "Local/Demo"
            st.sidebar.markdown(f"🤖 **LLM:** {llm_type}")
        
        # Clear system button (local development only)
        if DEPLOYMENT_CONFIG.get("environment") != "streamlit_cloud":
            if st.sidebar.button("🗑️ Clear System", help="Reset and clear all data"):
                for key in ["pipeline_initialized", "document_processor", "embedding_system", "rag_system"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.messages = []
                st.rerun()
    
    else:
        st.sidebar.warning("🟡 **System Not Initialized**")
        
        # Check for existing state
        existing_state = check_existing_state()
        
        if DEPLOYMENT_CONFIG.get("environment") == "streamlit_cloud":
            # Cloud deployment - simplified sidebar
            if existing_state['complete_state_exist']:
                st.sidebar.success("💾 **Ready to Start**")
                if existing_state['last_update']:
                    if '|' in existing_state['last_update']:
                        timestamp, _ = existing_state['last_update'].split('|', 1)
                        st.sidebar.write(f"Knowledge base: {timestamp}")
                
                if st.sidebar.button("⚡ Fast Load", type="primary", help="Start with saved knowledge base"):
                    with st.sidebar:
                        if run_fast_initialization():
                            st.success("✅ Ready to chat!")
                            st.rerun()
                        else:
                            st.error("❌ Initialization failed")
            else:
                st.sidebar.info("🆕 **System Setup**")
                
                if st.sidebar.button("🚀 Initialize System", type="primary", help="Initialize with pre-processed knowledge"):
                    with st.sidebar:
                        if run_initialization_pipeline():
                            st.success("✅ System ready!")
                            st.rerun()
                        else:
                            st.error("❌ Initialization failed")
        else:
            # Local development - show full options
            kb_changed = check_knowledge_base_changes()
            
            if existing_state['complete_state_exist']:
                if kb_changed:
                    st.sidebar.warning("⚠️ **Knowledge base has changed**")
                    st.sidebar.write("Your PDF files have been modified since last initialization.")
                    
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        if st.button("🚀 Fast Load", help="Load existing state (may be outdated)"):
                            with st.sidebar:
                                if run_fast_initialization():
                                    st.success("✅ Fast initialization complete!")
                                    st.rerun()
                                else:
                                    st.error("❌ Fast initialization failed")
                    
                    with col2:
                        if st.button("🔄 Full Rebuild", help="Process documents from scratch"):
                            with st.sidebar:
                                if run_initialization_pipeline():
                                    st.success("✅ System initialized successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Initialization failed")
                else:
                    st.sidebar.success("💾 **Saved state available**")
                    if existing_state['last_update']:
                        if '|' in existing_state['last_update']:
                            timestamp, _ = existing_state['last_update'].split('|', 1)
                            st.sidebar.write(f"Last updated: {timestamp}")
                    
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        if st.button("⚡ Fast Load", type="primary", help="Load from saved state (recommended)"):
                            with st.sidebar:
                                if run_fast_initialization():
                                    st.success("✅ Fast initialization complete!")
                                    st.rerun()
                                else:
                                    st.error("❌ Fast initialization failed")
                    
                    with col2:
                        if st.button("🔄 Rebuild", help="Process documents from scratch"):
                            with st.sidebar:
                                if run_initialization_pipeline():
                                    st.success("✅ System initialized successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Initialization failed")
            else:
                st.sidebar.info("🆕 **First time setup**")
                
                # Full initialization button
                if st.sidebar.button("🚀 Initialize System", type="primary", help="Process documents and create knowledge base"):
                    with st.sidebar:
                        if run_initialization_pipeline():
                            st.success("✅ System initialized successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Initialization failed")
        
        # Clear saved state option (local development only)
        if DEPLOYMENT_CONFIG.get("environment") != "streamlit_cloud" and existing_state['complete_state_exist']:
            st.sidebar.markdown("---")
            if st.sidebar.button("🗑️ Clear Saved State", help="Remove all saved data"):
                if clear_persistent_state():
                    st.sidebar.success("✅ Saved state cleared")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to clear state")
    
    # Sample questions
    st.sidebar.markdown("### 💡 Sample Questions")
    
    sample_questions = [
        "What are the three doshas?",
        "How to balance Vata dosha?",
        "Ayurvedic remedies for digestion",
        "What is Panchakarma treatment?",
        "Benefits of turmeric in Ayurveda"
    ]
    
    for question in sample_questions:
        if st.sidebar.button(f"💬 {question}", key=f"sample_{question}", help="Click to ask this question"):
            if st.session_state.pipeline_initialized:
                # Add to chat directly
                st.session_state.sample_question = question
                st.rerun()
            else:
                st.sidebar.error("Please initialize the system first")
    
    # Clear chat button (local development only)
    if DEPLOYMENT_CONFIG.get("environment") != "streamlit_cloud":
        if st.sidebar.button("🗑️ Clear Chat History"):
            # Keep only the welcome message
            if st.session_state.messages:
                st.session_state.messages = st.session_state.messages[:1]
            st.rerun()
    
    # System information
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("AI-powered Ayurveda consultant for intelligent responses based on traditional knowledge.")

# =============================================================================
# 🚀 MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title(APP_CONFIG["title"])
    st.markdown(APP_CONFIG["description"])
    
    # Check for sample question from sidebar
    if hasattr(st.session_state, 'sample_question'):
        prompt = st.session_state.sample_question
        del st.session_state.sample_question
        
        # Process sample question
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "sources": [],
            "response_time": 0
        })
        
        # Generate response
        if st.session_state.pipeline_initialized and st.session_state.rag_system:
            response = st.session_state.rag_system.query(prompt)
            
            if not response.get("error", True):
                answer = response.get("result", "I couldn't find relevant information.")
                sources = response.get("source_documents", [])
                response_time = response.get("response_time", 0)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "response_time": response_time
                })
        
        st.rerun()
    
    # Create layout
    # Sidebar
    with st.sidebar:
        display_sidebar()
    
    # Main content
    if not st.session_state.pipeline_initialized:
        # Check for saved state to show appropriate instructions
        existing_state = check_existing_state()
        
        if DEPLOYMENT_CONFIG.get("environment") == "streamlit_cloud":
            # Cloud deployment - simplified interface
            if existing_state['complete_state_exist']:
                st.markdown("""
                ## 🌿 Ayurveda Knowledge Bot
                **Your AI consultant for traditional Ayurvedic medicine**
                
                
                **👈 Click "Fast Load" in the sidebar to begin!**
                """)
                
                # Show saved state info
                if existing_state['last_update']:
                    if '|' in existing_state['last_update']:
                        timestamp, _ = existing_state['last_update'].split('|', 1)
                        st.info(f"💾 **Knowledge base last updated:** {timestamp}")
            else:
                st.markdown("""
                ## 🌿 Ayurveda Knowledge Bot
                **Your AI consultant for traditional Ayurvedic medicine**
                
                
                """)
        else:
            # Local development - show full options
            kb_changed = check_knowledge_base_changes()
            
            if existing_state['complete_state_exist'] and not kb_changed:
                # Fast initialization available
                st.markdown("""
                ## ⚡ Welcome Back!
                
                Great news! I found your previously processed knowledge base. You can:
                
                ### 🚀 **Fast Load (Recommended)**
                - ⚡ **Instant startup** in ~5-10 seconds
                - 📦 **Use saved documents** and embeddings  
                - 🤖 **Ready to chat** immediately
                #### **👈 Choose Fast Load in the sidebar!**
                            
                ### ⚠️ Please DO NOT rebuild the knowledge base
                **it will take a lot of time and resources.**
                
                
                """)
                
                # Show saved state info
                if existing_state['last_update']:
                    if '|' in existing_state['last_update']:
                        timestamp, _ = existing_state['last_update'].split('|', 1)
                        st.info(f"💾 **Last processed:** {timestamp}")
            
            elif existing_state['complete_state_exist'] and kb_changed:
                # Knowledge base changed
                st.markdown("""
                ## ⚠️ Knowledge Base Updated
                
                I detected that your PDF documents have been modified since the last initialization.
                
                ### 🚀 **Fast Load**
                - ⚡ **Quick startup** using existing processed data
                - ⚠️ **May miss new content** from updated PDFs
                - 💬 **Ready to chat** with previous knowledge
                
                ### 🔄 **Full Rebuild (Recommended)**
                - 🔍 **Process all current documents** 
                - ✅ **Include latest changes** in your PDFs
                - 💾 **Update saved state** for future fast loading
                
                **👈 Choose your option in the sidebar!**
                """)
            
            else:
                # First time setup
                st.markdown("""
                ## 🚀 First Time Setup
                
                Welcome to the Ayurveda Knowledge Bot! Let's get you started:
                
                1. **📁 Prepare Documents**: Ensure your `knowledge_base_files` folder contains PDF documents about Ayurveda
                2. **🔑 API Keys**: Set your OpenAI API key in environment variables (optional)
                3. **🚀 Initialize**: Click "Initialize System" in the sidebar
                4. **💬 Chat**: Start asking questions about Ayurveda!
                
                ### 📋 System Requirements
                - PDF documents in `knowledge_base_files/` folder
                - Internet connection for downloading models
                - Optional: OpenAI API key for best performance
                
                ### 💡 **After first initialization:**
                - ⚡ **Fast loading** on subsequent runs
                - 💾 **Persistent storage** of processed documents
                - 🔄 **Auto-detection** of document changes
                """)
            
            # Check for knowledge base folder (local only)
            if os.path.exists(APP_CONFIG["knowledge_base_folder"]):
                pdf_files = list(Path(APP_CONFIG["knowledge_base_folder"]).glob("*.pdf"))
                if pdf_files:
                    st.success(f"✅ Found {len(pdf_files)} PDF files in knowledge base")
                    
                    with st.expander("📁 Knowledge Base Files"):
                        for pdf_file in pdf_files:
                            file_size = pdf_file.stat().st_size / 1024  # KB
                            st.write(f"- {pdf_file.name} ({file_size:.1f} KB)")
                else:
                    st.warning("⚠️ No PDF files found in knowledge_base_files folder")
            else:
                st.error(f"❌ Knowledge base folder not found: {APP_CONFIG['knowledge_base_folder']}")
    
    else:
        # Show chat interface
        display_chat_interface()

if __name__ == "__main__":
    main() 