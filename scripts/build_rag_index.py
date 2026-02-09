#!/usr/bin/env python3
"""Build RAG index for CytoAtlas documentation and context.

Creates a LanceDB vector database from:
1. Documentation (docs/*.md)
2. Column definitions (from registry.json)
3. Atlas summaries
4. Biological context (cytokines, cell types, known biology)
5. Data summaries (*_summary.json files)

Usage:
    python scripts/build_rag_index.py --db-path cytoatlas-api/rag_db
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "cytoatlas-api"))

from app.services.chat.embeddings import get_embedding_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

    return chunks


def extract_docs(docs_dir: Path) -> list[dict[str, Any]]:
    """Extract documentation chunks from markdown files.

    Args:
        docs_dir: Path to docs directory

    Returns:
        List of document chunks
    """
    docs = []

    if not docs_dir.exists():
        logger.warning(f"Docs directory not found: {docs_dir}")
        return docs

    md_files = list(docs_dir.glob("**/*.md"))
    logger.info(f"Found {len(md_files)} markdown files")

    for md_file in md_files:
        try:
            text = md_file.read_text()
            chunks = chunk_text(text, chunk_size=500, overlap=50)

            for i, chunk in enumerate(chunks):
                docs.append({
                    "source_id": f"{md_file.name}#chunk{i}",
                    "source_type": "docs",
                    "text": chunk,
                    "atlas": "all",
                    "title": md_file.stem.replace("_", " ").title(),
                    "file": md_file.name,
                })
        except Exception as e:
            logger.error(f"Error reading {md_file}: {e}")

    logger.info(f"Extracted {len(docs)} documentation chunks")
    return docs


def extract_column_definitions(registry_path: Path) -> list[dict[str, Any]]:
    """Extract column definitions from registry.json.

    Args:
        registry_path: Path to registry.json

    Returns:
        List of column definition documents
    """
    docs = []

    if not registry_path.exists():
        logger.warning(f"Registry not found: {registry_path}")
        return docs

    try:
        with open(registry_path) as f:
            registry = json.load(f)

        # Common columns
        common_columns = {
            "cell_type": "Cell type annotation (e.g., CD4+ T, Monocyte)",
            "signature": "Cytokine or protein name (e.g., IL6, TGFB1)",
            "signature_type": "CytoSig (44 cytokines) or SecAct (1,249 proteins)",
            "rho": "Spearman correlation coefficient (-1 to 1)",
            "pvalue": "Statistical p-value (0 to 1)",
            "fdr": "FDR-corrected p-value (Benjamini-Hochberg)",
            "activity_diff": "Activity difference (group1_mean - group2_mean, NOT log2FC)",
            "mean_activity": "Mean activity z-score (typically -3 to +3)",
            "n_cells": "Number of cells in the group",
            "organ": "Organ/tissue name (scAtlas)",
            "disease": "Disease diagnosis (Inflammation Atlas)",
            "comparison": "Comparison label (e.g., 'sex_Male_vs_Female')",
            "neg_log10_pval": "-log10(pvalue) for volcano plots",
        }

        for col, desc in common_columns.items():
            docs.append({
                "source_id": f"column:{col}",
                "source_type": "column",
                "text": f"Column '{col}': {desc}",
                "atlas": "all",
                "column_name": col,
            })

        # Extract from files in registry
        for file_name, file_info in registry.get("files", {}).items():
            schema = file_info.get("schema", {})
            for col, col_type in schema.items():
                if col not in common_columns:
                    docs.append({
                        "source_id": f"column:{col}:{file_name}",
                        "source_type": "column",
                        "text": f"Column '{col}' in {file_name}: type {col_type}",
                        "atlas": file_info.get("atlas", "all"),
                        "column_name": col,
                        "file": file_name,
                    })

        logger.info(f"Extracted {len(docs)} column definitions")
    except Exception as e:
        logger.error(f"Error reading registry: {e}")

    return docs


def extract_atlas_summaries(data_dir: Path) -> list[dict[str, Any]]:
    """Extract atlas summary information.

    Args:
        data_dir: Path to visualization/data directory

    Returns:
        List of atlas summary documents
    """
    docs = []

    atlases = {
        "cima": {
            "name": "CIMA (Chinese Immune Multi-omics Atlas)",
            "description": "6.5 million cells from 421 healthy adults (age 25-85). Includes cytokine/protein activity, age/BMI correlations, and blood biochemistry correlations.",
        },
        "inflammation": {
            "name": "Inflammation Atlas",
            "description": "~5 million cells across 20+ inflammatory diseases. Three cohorts: main, validation, external. Includes disease vs healthy differential activity and treatment response predictions.",
        },
        "scatlas": {
            "name": "scAtlas (Human Tissue Atlas)",
            "description": "6.4 million cells across 35 organs/tissues. Includes both normal tissues and pan-cancer immune profiling. Provides organ-specific activity and tumor vs adjacent comparisons.",
        },
    }

    for atlas_key, atlas_info in atlases.items():
        # Basic summary
        docs.append({
            "source_id": f"atlas:{atlas_key}",
            "source_type": "atlas",
            "text": f"{atlas_info['name']}: {atlas_info['description']}",
            "atlas": atlas_key,
            "title": atlas_info["name"],
        })

        # Try to load summary file
        summary_file = data_dir / f"{atlas_key}_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary_data = json.load(f)
                    summary_text = json.dumps(summary_data, indent=2)
                    docs.append({
                        "source_id": f"atlas:{atlas_key}:summary",
                        "source_type": "data",
                        "text": f"{atlas_info['name']} summary: {summary_text}",
                        "atlas": atlas_key,
                        "title": f"{atlas_info['name']} Summary",
                    })
            except Exception as e:
                logger.error(f"Error reading summary {summary_file}: {e}")

    logger.info(f"Extracted {len(docs)} atlas summaries")
    return docs


def extract_biological_context() -> list[dict[str, Any]]:
    """Extract biological context (key cytokines, cell types, etc.).

    Returns:
        List of biological context documents
    """
    docs = []

    # Key cytokines
    cytokines = {
        "IFNG": "Interferon-gamma (IFNG): Type II interferon produced by T cells and NK cells. Critical for anti-viral immunity and Th1 responses.",
        "TNF": "Tumor Necrosis Factor (TNF): Pro-inflammatory cytokine produced by macrophages and T cells. Central to inflammation and immune regulation.",
        "IL6": "Interleukin-6 (IL-6): Pleiotropic cytokine involved in inflammation, acute phase response, and B cell differentiation.",
        "IL17A": "Interleukin-17A (IL-17A): Key Th17 cytokine involved in autoimmunity and defense against extracellular pathogens.",
        "IL10": "Interleukin-10 (IL-10): Anti-inflammatory cytokine that suppresses immune responses.",
        "IL2": "Interleukin-2 (IL-2): T cell growth factor essential for T cell proliferation and immune homeostasis.",
        "TGFB1": "Transforming Growth Factor Beta 1 (TGFB1): Immunosuppressive cytokine involved in tissue repair and immune regulation.",
        "IL4": "Interleukin-4 (IL-4): Th2 cytokine that promotes B cell activation and antibody production.",
    }

    for sig, desc in cytokines.items():
        docs.append({
            "source_id": f"biology:cytokine:{sig}",
            "source_type": "biology",
            "text": desc,
            "atlas": "all",
            "signature": sig,
        })

    # Key cell types
    cell_types = {
        "CD4 T": "CD4+ T cells (helper T cells): Express CD4 and help other immune cells. Include Th1, Th2, Th17, and Treg subsets.",
        "CD8 T": "CD8+ T cells (cytotoxic T cells): Express CD8 and kill infected or cancerous cells. Major producers of IFNG.",
        "NK": "Natural Killer (NK) cells: Innate lymphocytes that kill infected cells and produce IFNG.",
        "Monocyte": "Monocytes: Circulating myeloid cells that differentiate into macrophages and dendritic cells. Produce TNF and IL-6.",
        "B cell": "B cells: Lymphocytes that produce antibodies. Can present antigens to T cells.",
        "Macrophage": "Macrophages: Tissue-resident phagocytes derived from monocytes. Key producers of inflammatory cytokines.",
        "Dendritic": "Dendritic cells: Professional antigen-presenting cells that bridge innate and adaptive immunity.",
    }

    for ct, desc in cell_types.items():
        docs.append({
            "source_id": f"biology:celltype:{ct}",
            "source_type": "biology",
            "text": desc,
            "atlas": "all",
            "cell_type": ct,
        })

    logger.info(f"Extracted {len(docs)} biological context documents")
    return docs


def build_index(
    documents: list[dict[str, Any]],
    db_path: Path,
    embedding_service,
) -> None:
    """Build LanceDB index from documents.

    Args:
        documents: List of documents to index
        db_path: Path to LanceDB database
        embedding_service: Embedding service
    """
    import lancedb

    logger.info(f"Building index with {len(documents)} documents")

    # Extract texts for embedding
    texts = [doc["text"] for doc in documents]

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedding_service.embed_batch(texts, batch_size=32)

    # Add embeddings to documents
    for doc, emb in zip(documents, embeddings):
        doc["vector"] = emb

    # Create LanceDB
    logger.info(f"Creating LanceDB at {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))

    # Drop existing table if it exists
    if "cytoatlas_docs" in db.table_names():
        db.drop_table("cytoatlas_docs")

    # Create table
    table = db.create_table("cytoatlas_docs", documents)
    logger.info(f"Created table with {len(table)} documents")


def main():
    parser = argparse.ArgumentParser(description="Build RAG index for CytoAtlas")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=project_root / "cytoatlas-api" / "rag_db",
        help="Path to LanceDB database",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=project_root / "docs",
        help="Path to documentation directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "visualization" / "data",
        help="Path to visualization data directory",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "docs" / "registry.json",
        help="Path to registry.json",
    )

    args = parser.parse_args()

    logger.info("=== Building CytoAtlas RAG Index ===")

    # Initialize embedding service
    logger.info("Loading embedding model...")
    embedding_service = get_embedding_service()

    # Collect all documents
    all_docs = []

    # 1. Documentation
    logger.info("\n[1/5] Extracting documentation...")
    all_docs.extend(extract_docs(args.docs_dir))

    # 2. Column definitions
    logger.info("\n[2/5] Extracting column definitions...")
    all_docs.extend(extract_column_definitions(args.registry))

    # 3. Atlas summaries
    logger.info("\n[3/5] Extracting atlas summaries...")
    all_docs.extend(extract_atlas_summaries(args.data_dir))

    # 4. Biological context
    logger.info("\n[4/5] Extracting biological context...")
    all_docs.extend(extract_biological_context())

    # 5. Data summaries (from visualization/data/)
    logger.info("\n[5/5] Extracting data summaries...")
    if args.data_dir.exists():
        for json_file in args.data_dir.glob("*_summary.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    summary_text = json.dumps(data, indent=2)[:1000]  # Limit size
                    all_docs.append({
                        "source_id": f"data:{json_file.name}",
                        "source_type": "data",
                        "text": f"Data summary from {json_file.name}: {summary_text}",
                        "atlas": json_file.name.split("_")[0] if "_" in json_file.name else "all",
                        "file": json_file.name,
                    })
            except Exception as e:
                logger.error(f"Error reading {json_file}: {e}")

    logger.info(f"\nTotal documents: {len(all_docs)}")

    # Build index
    logger.info("\n=== Building Index ===")
    build_index(all_docs, args.db_path, embedding_service)

    logger.info("\n=== Index Build Complete ===")
    logger.info(f"Database location: {args.db_path}")
    logger.info(f"Total documents indexed: {len(all_docs)}")


if __name__ == "__main__":
    main()
