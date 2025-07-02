import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import logging
import PyPDF2
from io import BytesIO

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_path: str = "data/models/faiss_index.bin",
                 metadata_path: str = "data/models/metadata.json"):
        
        self.embedding_model_name = embedding_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.embeddings = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        self.load_embedding_model()
        self.load_or_create_index()
    
    def load_embedding_model(self):
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    def load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load_index()
        else:
            self.create_new_index()
    
    def create_new_index(self):
        """Create a new FAISS index"""
        logger.info("Creating new FAISS index")
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product index
        self.documents = []
        self.embeddings = []
        
    
    def load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(self.index_path)
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.embeddings = np.array(data['embeddings'])
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
        except Exception as e:            
            logger.error(f"Error loading index: {e}")
            self.create_new_index()
    
    def save_index(self):
        """Save FAISS index and metadata"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            faiss.write_index(self.index, self.index_path)
            
            metadata = {
                'documents': self.documents,
                'embeddings': self.embeddings.tolist() if len(self.embeddings) > 0 else []
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def add_documents(self, documents: List[str]):
        try:
            logger.info(f"Adding {len(documents)} documents to index")
            
            # Generate embeddings
            new_embeddings = self.embedding_model.encode(documents, normalize_embeddings=True)
            
            # Add to FAISS index
            self.index.add(new_embeddings.astype(np.float32))
            
            # Store documents and embeddings
            self.documents.extend(documents)
            if len(self.embeddings) == 0:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            self.save_index()
            
            logger.info(f"Successfully added {len(documents)} documents")        
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        try:
            if len(self.documents) == 0:
                return []
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score > 0.1:  # Minimum similarity threshold
                    results.append((self.documents[idx], float(score)))
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_context(self, query: str, k: int = 3) -> str:
        results = self.search(query, k)
        if not results:
            return ""
        
        contexts = [doc for doc, score in results]
        return "\n\n".join(contexts)
    
    def add_knowledge_from_directory(self, directory_path: str):
        documents = []
        
        if not os.path.exists(directory_path):
            logger.warning(f"Directory not found: {directory_path}")
            return
        
        # Get list of already processed files by checking document prefixes
        processed_files = set()
        for doc in self.documents:
            if doc.startswith('[') and ']' in doc:
                filename = doc.split(']')[0][1:]  # Extract filename from [filename] prefix
                processed_files.add(filename)
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            content = ""
            
            # Skip if this file has already been processed
            if filename in processed_files:
                logger.info(f"Skipping already processed file: {filename}")
                continue
            
            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                elif filename.endswith('.pdf'):
                    logger.info(f"Processing PDF file: {filename}")
                    content = self._extract_text_from_pdf(file_path)
                
                if content:
                    # Split long documents into chunks
                    chunks = self._split_document(content)
                    # Add filename prefix to chunks for better context
                    labeled_chunks = [f"[{filename}] {chunk}" for chunk in chunks]
                    documents.extend(labeled_chunks)
                    logger.info(f"Extracted {len(chunks)} chunks from {filename}")
                    
            except Exception as e:
                logger.error(f"Error reading file {filename}: {e}")
        
        if documents:
            self.add_documents(documents)
            logger.info(f"Added {len(documents)} new document chunks from {directory_path}")
        else:
            logger.info(f"No new documents found in {directory_path} (all files already processed)")
    
    def add_pdf_document(self, pdf_path: str):
        """Add a single PDF document to the vector database"""
        try:
            filename = os.path.basename(pdf_path)
            
            # Check if this file has already been processed
            processed_files = set()
            for doc in self.documents:
                if doc.startswith('[') and ']' in doc:
                    existing_filename = doc.split(']')[0][1:]  # Extract filename from [filename] prefix
                    processed_files.add(existing_filename)
            
            if filename in processed_files:
                logger.info(f"PDF already processed: {filename}")
                return
            
            logger.info(f"Processing PDF: {pdf_path}")
            text = self._extract_text_from_pdf(pdf_path)
            
            if text:
                chunks = self._split_document(text)
                labeled_chunks = [f"[{filename}] {chunk}" for chunk in chunks]
                self.add_documents(labeled_chunks)
                logger.info(f"Successfully added {len(chunks)} chunks from {filename}")
            else:
                logger.warning(f"No text extracted from {pdf_path}")
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    def _split_document(self, text: str, chunk_size: int = 500) -> List[str]:
        # Simple chunking by sentences
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.dimension,
            "model_name": self.embedding_model_name,
            "index_exists": self.index is not None
        }
    
    def clear_database(self):
        """Clear all documents and rebuild the index"""
        logger.info("Clearing vector database...")
        self.create_new_index()
        
        # Remove existing files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        
        logger.info("Vector database cleared")
    
    def get_processed_files(self) -> set:
        """Get list of already processed files"""
        processed_files = set()
        for doc in self.documents:
            if doc.startswith('[') and ']' in doc:
                filename = doc.split(']')[0][1:]  # Extract filename from [filename] prefix
                processed_files.add(filename)
        return processed_files
