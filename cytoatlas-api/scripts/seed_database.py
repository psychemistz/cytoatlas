#!/usr/bin/env python3
"""Seed the database with atlas metadata and initial data."""

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.core.database import async_session_factory, init_db
from app.models import Atlas, Signature

settings = get_settings()


async def seed_atlases():
    """Seed atlas metadata."""
    atlases = [
        {
            "name": "CIMA",
            "version": "1.0.0",
            "description": "Chinese Immune Multi-omics Atlas - 6.5M healthy donor cells with rich phenotypic data",
            "n_cells": 6484974,
            "n_samples": 421,
            "n_cell_types": 0,  # Will be computed
            "h5ad_path": str(settings.cima_h5ad),
            "results_path": str(settings.cima_results_dir),
            "status": "active",
        },
        {
            "name": "Inflammation",
            "version": "1.0.0",
            "description": "Inflammation Atlas - 6.3M cells across diseases with treatment response data",
            "n_cells": 4918140,
            "n_samples": 817,
            "n_cell_types": 66,
            "h5ad_path": str(settings.inflammation_main_h5ad),
            "results_path": str(settings.inflammation_results_dir),
            "status": "active",
        },
        {
            "name": "scAtlas",
            "version": "1.0.0",
            "description": "scAtlas - 6.4M cells from normal organs and cancer",
            "n_cells": 6400000,
            "n_samples": 0,
            "n_cell_types": 376,
            "h5ad_path": str(settings.scatlas_normal_h5ad),
            "results_path": str(settings.scatlas_results_dir),
            "status": "active",
        },
    ]

    async with async_session_factory() as session:
        for atlas_data in atlases:
            # Check if exists
            result = await session.execute(
                select(Atlas).where(Atlas.name == atlas_data["name"])
            )
            existing = result.scalar_one_or_none()

            if existing:
                print(f"Atlas '{atlas_data['name']}' already exists, updating...")
                for key, value in atlas_data.items():
                    setattr(existing, key, value)
            else:
                print(f"Creating atlas '{atlas_data['name']}'...")
                atlas = Atlas(**atlas_data)
                session.add(atlas)

        await session.commit()
        print("Atlases seeded successfully!")


async def seed_signatures():
    """Seed signature metadata."""
    # Load CytoSig signatures
    cytosig_signatures = [
        "Activin A", "BDNF", "BMP2", "BMP4", "BMP6", "CD40L", "CXCL12",
        "EGF", "FGF2", "GCSF", "GDF11", "GMCSF", "HGF", "IFN1", "IFNG",
        "IFNL", "IL10", "IL12", "IL13", "IL15", "IL17A", "IL1A", "IL1B",
        "IL2", "IL21", "IL22", "IL27", "IL3", "IL36", "IL4", "IL6",
        "LIF", "LTA", "MCSF", "NO", "OSM", "TGFB1", "TGFB3", "TNFA",
        "TRAIL", "TWEAK", "VEGFA", "WNT3A"
    ]

    async with async_session_factory() as session:
        for sig_name in cytosig_signatures:
            result = await session.execute(
                select(Signature).where(
                    Signature.name == sig_name,
                    Signature.signature_type == "CytoSig"
                )
            )
            existing = result.scalar_one_or_none()

            if not existing:
                sig = Signature(
                    name=sig_name,
                    signature_type="CytoSig",
                    description=f"CytoSig signature for {sig_name}",
                    n_genes=50,  # Approximate
                    category="cytokine" if "IL" in sig_name else "growth_factor",
                )
                session.add(sig)

        await session.commit()
        print(f"Seeded {len(cytosig_signatures)} CytoSig signatures")

    # Note: SecAct has ~1249 signatures, would need to load from file
    print("SecAct signatures should be loaded from signature matrix file")


async def main():
    """Run all seeders."""
    print("=" * 50)
    print("  CytoAtlas Database Seeder")
    print("=" * 50)
    print()

    # Initialize database tables
    print("Initializing database tables...")
    await init_db()

    # Seed data
    await seed_atlases()
    await seed_signatures()

    print()
    print("Database seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())
