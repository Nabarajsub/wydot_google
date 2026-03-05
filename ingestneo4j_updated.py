#!/usr/bin/env python3
"""
WYDOT Knowledge Graph Ingestion Pipeline (Advanced)
Implements:
- Phase 1: Google GenAI Embeddings (gemini-embedding-001)
- Phase 2: PyMuPDF parsing + Gemini 2.5 Flash metadata generation
- Phase 3: Hybrid Chunking (Regex structure -> SemanticChunker)
- Phase 4: Hybrid Entity Extraction + Inline ER + Neo4j Graph Writing
"""

import os
import glob
import re
import json
import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any
import concurrent.futures
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_experimental.text_splitter import SemanticChunker
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

import time
from google.api_core.exceptions import ResourceExhausted

def retry_with_backoff(func, *args, **kwargs):
    max_retries = 8
    base_delay = 2
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            is_rate_limit = isinstance(e, ResourceExhausted) or "429" in str(e) or "Quota exceeded" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt)
                print(f"   ⚠️ Rate limit hit. Retrying in {delay}s... ({str(e)[:100]})")
                time.sleep(delay)
            else:
                raise e

class RateLimitedEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            res = retry_with_backoff(super().embed_documents, batch)
            results.extend(res)
            time.sleep(0.5)
        return results

    def embed_query(self, text: str) -> List[float]:
        return retry_with_backoff(super().embed_query, text)

# -------------------------------------------------------------------
# Configuration & Credentials
# -------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Hardcoded Neo4j Credentials as requested
NEO4J_URI = "neo4j+s://1c9edfe6.databases.neo4j.io"
NEO4J_USERNAME = "1c9edfe6"
NEO4J_PASSWORD = "IlZpB7BG3sM34FQ5d_Juv5CidvCHvsMnoLkXHW18CSA"

DATA_FOLDER = "/Users/uw-user/Desktop/WYDOT/Data/Final folder with pdfs only"
TRACKER_FILE = "ingest_updated_tracker.txt"
MEMORY_FILE = "entity_memory.json"

def get_page_number(chunk_text: str, full_text: str, page_map: list) -> int:
    """Finds the approximate page number for a chunk of text."""
    idx = full_text.find(chunk_text)
    if idx == -1: return 0
    for start, end, page in page_map:
        if start <= idx <= end:
            return page
    return 0

# -------------------------------------------------------------------
# LangChain & Models Setup
# -------------------------------------------------------------------
embeddings = RateLimitedEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
    task_type="retrieval_document"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1
)

# -------------------------------------------------------------------
# Pydantic Schemas for Gemini Structured Output
# -------------------------------------------------------------------
class DocumentMetadata(BaseModel):
    display_title: str = Field(description="A clean, human-readable title for the document.")
    document_series: str = Field(description="The core type/series of the document excluding the year (e.g., 'Standard Specifications', 'Right of Way Manual'). Defaults to 'General' if unknown.")
    year: int = Field(description="The year the document was published or became effective (e.g., 2010, 2021). Use 0 if unknown.")
    primary_category: str = Field(description="Must be one of: Regulatory / Manual, Meeting / Administrative, Form / Application, Report / Study, Program / Plan, Public Outreach")
    secondary_category: str = Field(description="Secondary category if applicable, else empty.")

class Entity(BaseModel):
    name: str = Field(description="Canonical entity name")
    label: str = Field(description="Type: Material, Specification, Standard, TestMethod, Form, Committee_Or_Group, Project_Or_Corridor, Concept, Entity")

class Relationship(BaseModel):
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relation_type: str = Field(description="ALL_CAPS relationship type (e.g., REQUIRES, REFERENCES)")

class GraphExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

# -------------------------------------------------------------------
# Phase 2: Document Intelligence
# -------------------------------------------------------------------
def extract_pdf_with_pymupdf(filepath: str) -> tuple[str, list]:
    """Extracts text and markdown tables from PDF using PyMuPDF. Returns (full_text, page_map)."""
    doc = fitz.open(filepath)
    text_blocks = []
    page_map = []
    current_char = 0
    
    for i, page in enumerate(doc):
        page_text_blocks = []
        page_text_blocks.append(page.get_text("text"))
        
        # Extract tables as markdown (if any exist)
        tabs = page.find_tables()
        for tab in tabs:
            if tab.extract():
                try:
                    df = tab.to_pandas()
                    page_text_blocks.append("\n" + df.to_markdown(index=False) + "\n")
                except Exception:
                    pass
                    
        page_text = "".join(page_text_blocks)
        if page_text:
            text_blocks.append(page_text)
            page_map.append((current_char, current_char + len(page_text), i + 1))
            current_char += len(page_text) + 1  # +1 for newline join
            
    return "\n".join(text_blocks), page_map

def get_document_metadata(filename: str, first_page_text: str) -> DocumentMetadata:
    """Uses Gemini 2.5 Flash to intelligently name and categorize the document."""
    prompt = f"""
    Analyze the following filename and the first few pages of a WYDOT (Wyoming Department of Transportation) document.
    Generate a clean 'display_title', 'document_series', 'year', and classify it.
    If the cover page doesn't have the year or series, look further into the provided text snippet.
    
    Filename: {filename}
    Content Snippet: {first_page_text[:20000]}
    """
    structured_llm = llm.with_structured_output(DocumentMetadata)
    try:
        return retry_with_backoff(structured_llm.invoke, prompt)
    except Exception as e:
        print(f"Error generating metadata: {e}")
        return DocumentMetadata(display_title=filename, document_series="General", year=0, primary_category="Regulatory / Manual", secondary_category="")

# -------------------------------------------------------------------
# Phase 3: Hybrid Chunking
# -------------------------------------------------------------------
def hybrid_chunk_document(text: str, source_meta: dict, page_map: list) -> List[Document]:
    """Layer 1: Structural Regex -> Layer 2: Semantic Chunker"""
    print("   > Running Hybrid Chunking...")
    
    # Layer 1: Split into large macro-sections via regex
    section_splits = re.split(r'\n(?=DIVISION\s+\d+|SECTION\s+\d+|CHAPTER\s+\d+)', text, flags=re.IGNORECASE)
    
    # Layer 2: Semantic Chunking inside each section
    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    
    final_chunks = []
    chunk_seq = 0
    for section_text in section_splits:
        if len(section_text.strip()) < 50:
            continue
            
        # Detect section header
        header_match = re.match(r'(DIVISION\s+\d+|SECTION\s+\d+|CHAPTER\s+\d+)', section_text.strip(), re.IGNORECASE)
        section_name = header_match.group(1).upper() if header_match else "General"
        
        # Semantic sub-chunking
        sub_docs = semantic_chunker.create_documents([section_text])
        for doc in sub_docs:
            # Prevent Gemini from crashing on empty or extremely short chunks
            if len(doc.page_content.strip()) < 10:
                continue
                
            chunk_seq += 1
            meta = source_meta.copy()
            meta.update({
                "section": section_name,
                "chunk_seq": chunk_seq,
                "page": get_page_number(doc.page_content, text, page_map)
            })
            final_chunks.append(Document(page_content=doc.page_content, metadata=meta))
            
    return final_chunks

# -------------------------------------------------------------------
# Phase 4: Entity Extraction & Inline ER
# -------------------------------------------------------------------
def llm_graph_extraction(chunk_text: str, existing_entities: List[str]) -> GraphExtraction:
    """LLM Extraction USING Inline Entity Resolution framework"""
    
    prompt = f"""
    You are a WYDOT civil engineering specialized data extractor.
    Extract entities and relationships from the text.
    
    CRITICAL INLINE ENTITY RESOLUTION INSTRUCTIONS:
    Before creating a new entity, check this list of already existing canonical entities:
    {existing_entities}
    
    If the text mentions "Portland Cement" and "Type I Portland Cement" is in the existing list, ALWAYS use the exact string from the existing list to prevent duplicates.
    Do not invent new names if a matching concept exists in the list.
    
    Text:
    {chunk_text}
    """
    structured_llm = llm.with_structured_output(GraphExtraction)
    try:
        return retry_with_backoff(structured_llm.invoke, prompt)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return GraphExtraction(entities=[], relationships=[])

# -------------------------------------------------------------------
# Graph Writer (Cypher)
# -------------------------------------------------------------------
class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.existing_entities = set()
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    self.existing_entities = set(json.load(f))
            except Exception:
                pass
        self._setup_constraints()

    def save_memory(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(list(self.existing_entities), f)

    def _setup_constraints(self):
        with self.driver.session() as session:
            # Create constraints for deduplication
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.source IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            # Create vector index
            try:
                session.run("""
                CREATE VECTOR INDEX wydot_gemini_index IF NOT EXISTS 
                FOR (c:Chunk) ON (c.embedding) 
                OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
                """)
            except Exception as e:
                pass # Index might already exist

    def close(self):
        self.driver.close()

    def check_capacity(self) -> bool:
        """Returns True if the database is near the Aura Free limit (200k nodes / 400k rels)"""
        with self.driver.session() as session:
            try:
                # Approximate counts are fast
                node_count = session.run("MATCH (n) RETURN COUNT(n) AS count").single()["count"]
                rel_count = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS count").single()["count"]
                
                print(f"   📊 DB Capacity: {node_count:,}/200,000 Nodes | {rel_count:,}/400,000 Relationships")
                
                if node_count >= 190000 or rel_count >= 380000:
                    print(f"   🛑 WARNING: Neo4j Aura Free limits almost reached! Stopping ingestion to prevent errors.")
                    return False
                return True
            except Exception as e:
                print(f"   ⚠️ Error checking capacity: {e}")
                return True # Continue on error just in case

    def write_document_and_chunks(self, chunks: List[Document]):
        """Writes the document, chunks, and NEXT_CHUNK sequence to Neo4j."""
        print(f"   > Vectorizing {len(chunks)} chunks (Batch API)...")
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        # This will send a powerful batched request to Gemini instead of 1-by-1
        try:
            chunk_embeddings = embeddings.embed_documents(chunk_texts)
        except Exception as e:
            print(f"   ⚠️ Exception during batch embedding: {e}")
            # Fallback to sequential if batching fails for some anomalous length reason
            chunk_embeddings = [embeddings.embed_query(t) for t in chunk_texts]
            
        with self.driver.session() as session:
            for i, chunk in enumerate(chunks):
                chunk_id = f"{chunk.metadata['source']}_chk_{chunk.metadata['chunk_seq']}"
                
                # Write Document -> Section -> Chunk
                session.run("""
                MERGE (d:Document {source: $source})
                SET d.display_title = $title, d.document_series = $series, d.year = $year, d.primary_category = $cat, d.secondary_category = $scat
                
                MERGE (s:Section {name: $section, doc_source: $source})
                MERGE (d)-[:HAS_SECTION]->(s)
                
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text, c.seq = $seq, c.page = $page
                MERGE (s)-[:HAS_CHUNK]->(c)
                """, 
                source=chunk.metadata['source'], title=chunk.metadata['display_title'], 
                series=chunk.metadata['document_series'], year=chunk.metadata['year'],
                cat=chunk.metadata['primary_category'], scat=chunk.metadata['secondary_category'],
                section=chunk.metadata['section'], chunk_id=chunk_id, 
                text=chunk.page_content, seq=chunk.metadata['chunk_seq'],
                page=chunk.metadata.get('page', 0))

                # Vectorize and save embedding
                embedding_vector = chunk_embeddings[i]
                session.run("""
                MATCH (c:Chunk {id: $chunk_id})
                CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
                """, chunk_id=chunk_id, embedding=embedding_vector)

                # Link chunks sequentially
                if i > 0:
                    prev_chunk_id = f"{chunk.metadata['source']}_chk_{chunks[i-1].metadata['chunk_seq']}"
                    session.run("""
                    MATCH (c1:Chunk {id: $prev_id}), (c2:Chunk {id: $curr_id})
                    MERGE (c1)-[:NEXT_CHUNK]->(c2)
                    """, prev_id=prev_chunk_id, curr_id=chunk_id)

    def write_graph_extraction(self, chunk_id: str, extraction: GraphExtraction) -> tuple[int, int]:
        """Writes entities and links them to the source chunk, returning (nodes_created, rels_created)."""
        nodes_created = 0
        rels_created = 0
        with self.driver.session() as session:
            for ent in extraction.entities:
                # Track in memory for inline ER
                self.existing_entities.add(ent.name)
                
                # MERGE the entity dynamically with apoc or direct label injection (simplified here)
                query = f"""
                MATCH (c:Chunk {{id: $chunk_id}})
                MERGE (e:{ent.label} {{name: $name}})
                MERGE (c)-[:MENTIONS]->(e)
                """
                try:
                    result = session.run(query, chunk_id=chunk_id, name=ent.name)
                    nodes_created += result.consume().counters.nodes_created
                except Exception as e:
                    # Fallback if label is invalid
                    result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (e:Concept {name: $name})
                    MERGE (c)-[:MENTIONS]->(e)
                    """, chunk_id=chunk_id, name=ent.name)
                    nodes_created += result.consume().counters.nodes_created

            for rel in extraction.relationships:
                query = f"""
                MATCH (src {{name: $source}}), (tgt {{name: $target}})
                MERGE (src)-[r:{rel.relation_type}]->(tgt)
                """
                try:
                    result = session.run(query, source=rel.source_entity, target=rel.target_entity)
                    rels_created += result.consume().counters.relationships_created
                except Exception:
                    pass
        return nodes_created, rels_created

    def link_document_versions(self):
        """Phase 4.4: Finds documents of the same series and links them chronologically."""
        print("   > Building Cross-Document Version Chains ([:SUPERSEDES])...")
        with self.driver.session() as session:
            query = """
            MATCH (d:Document)
            WHERE d.year > 0 AND d.document_series IS NOT NULL AND d.document_series <> 'General'
            WITH d.document_series AS series, d
            ORDER BY d.year DESC
            WITH series, collect(d) AS docs
            UNWIND range(0, size(docs)-2) AS i
            WITH docs[i] AS newer, docs[i+1] AS older
            MERGE (newer)-[:SUPERSEDES]->(older)
            """
            result = session.run(query)
            summary = result.consume()
            print(f"     ✅ Created {summary.counters.relationships_created} new version links.")

# -------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------
def main():
    print("🚀 Starting Advanced WYDOT Knowledge Graph Ingestion...")
    
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Data folder not found: {DATA_FOLDER}")
        return

    # Find docs (exclude excel as requested)
    files = []
    for ext in ["*.pdf", "*.docx", "*.doc"]:
        files.extend(glob.glob(os.path.join(DATA_FOLDER, ext)))
        
    PRIORITY_FILES = [
        "Wyoming 2021 Standard Specifications for Road and Bridge Construction.pdf",
        "2010 Standard Specifications.pdf"
    ]
    
    # Sort files so priority files are processed first
    def get_priority(filepath):
        name = os.path.basename(filepath)
        if name in PRIORITY_FILES:
            return PRIORITY_FILES.index(name)
        return len(PRIORITY_FILES)
        
    files.sort(key=get_priority)
        
    print(f"📄 Found {len(files)} files to process.")
    
    processed_files = set()
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            processed_files = set([line.strip() for line in f.readlines() if line.strip()])
    print(f"⏭️  Skipping {len(processed_files)} already processed files.")
    
    neo_writer = Neo4jWriter()
    
    for filepath in files:
        filename = os.path.basename(filepath)
        if filename in processed_files:
            continue
            
        print(f"\nProcessing: {filename}")
        
        # 0. Check Capacity before starting a new file
        if not neo_writer.check_capacity():
            print("🛑 Stopping early due to database capacity limits.")
            break
        
        # 1. Parse File
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            full_text, page_map = extract_pdf_with_pymupdf(filepath)
        else:
            loader = Docx2txtLoader(filepath)
            docs = loader.load()
            full_text = "\n".join([d.page_content for d in docs])
            page_map = [(0, len(full_text), 1)]
            
        if not full_text.strip():
            print("   ⚠️ No text extracted, skipping.")
            continue
            
        # 2. Intelligence: Get Title & Categories
        print("   > Generating Document Metadata (Gemini Flash)...")
        meta_obj = get_document_metadata(filename, full_text)
        base_meta = {
            "source": filename,
            "display_title": meta_obj.display_title,
            "document_series": meta_obj.document_series,
            "year": meta_obj.year,
            "primary_category": meta_obj.primary_category,
            "secondary_category": meta_obj.secondary_category
        }
        
        # 3. Hybrid Chunking
        chunks = hybrid_chunk_document(full_text, base_meta, page_map)
        print(f"   > Generated {len(chunks)} chunks.")
        
        # Write structural tree to Neo4j
        neo_writer.write_document_and_chunks(chunks)
        
        # 4. Graph Extraction per chunk
        print("   > Extracting Entities & Relationships with Inline ER (Parallel)...")
        total_nodes = 0
        total_rels = 0
        
        # We sample the existing entities list once per document to save tokens
        # It's okay if threads don't see each other's live entities immediately as the LLM is robust
        memory_list = list(neo_writer.existing_entities)[-100:]

        def process_chunk(chunk):
            chunk_id = f"{chunk.metadata['source']}_chk_{chunk.metadata['chunk_seq']}"
            graph_data = llm_graph_extraction(chunk.page_content, memory_list)
            return chunk_id, graph_data

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all chunks to the thread pool
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
            
            # As they complete, write them sequentially to avoid Neo4j lock contention
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    chunk_id, graph_data = future.result()
                    n_created, r_created = neo_writer.write_graph_extraction(chunk_id, graph_data)
                    total_nodes += n_created
                    total_rels += r_created
                except Exception as exc:
                    print(f"   ⚠️ Chunk processing generated an exception: {exc}")

        # 5. Mark Document as Completed in Tracker
        with open(TRACKER_FILE, 'a') as f:
            f.write(f"{filename}\n")
        neo_writer.save_memory()
        print(f"   📊 Extraction Summary: {len(chunks)} Chunks | {total_nodes} Nodes | {total_rels} Relationships created.")
        print(f"   ✅ Saved {filename} to tracker.")

    print("\nPhase 4.4: Linking Document Versions...")
    neo_writer.link_document_versions()

    neo_writer.close()
    print("\n✅ Ingestion Complete!")

if __name__ == "__main__":
    main()
