# CytoAtlas Development Decisions Log

This file tracks critical decisions made during the development of the CytoAtlas visualization platform. Each decision is logged with context, rationale, and review status.

---

## Decision Log Format

Each entry follows this structure:
- **Date:** When the decision was made
- **Agent:** Which review agent evaluated/made the decision
- **Decision:** The specific choice made
- **Rationale:** Why this decision was chosen
- **Alternatives Considered:** Other options that were evaluated
- **Status:** Pending, Implemented, or Rejected
- **Review Needed:** Yes/No - whether human review is required

---

## Phase 1: Foundation Setup

### [2026-01-27] Project Structure
- **Agent:** Orchestrator
- **Decision:** Use modular directory structure with separate agent prompts, visualization panels, and data directories
- **Rationale:** Enables autonomous multi-agent development with clear separation of concerns
- **Alternatives Considered:** Monolithic structure, config-based approach
- **Status:** Implemented
- **Review Needed:** No

---

## Phase 2: CIMA Atlas Panels

### [2026-01-27] CIMA Tab Structure (Updated)
- **Agent:** Orchestrator, Viz Expert
- **Decision:** Implement 10 tabs: Age & BMI, Age/BMI Stratified, Biochemistry, Biochem Scatter, Metabolites, Differential, Cell Types, Multi-omic, Population, eQTL Browser
- **Rationale:** Expanded to include interactive scatter exploration, multi-omic integration, population stratification, and genetic regulation (eQTL)
- **Alternatives Considered:** Keep minimal 6 tabs, separate pages per analysis
- **Status:** Implemented
- **Review Needed:** No

### [2026-01-27] Age/BMI Stratification Bins
- **Agent:** Scientific Reviewer
- **Decision:** Use decade bins for age (<30, 30-39, etc.) and WHO categories for BMI
- **Rationale:** Standard clinical categorizations; matches CIMA metadata structure
- **Alternatives Considered:** Quintiles, custom ranges based on data distribution
- **Status:** Implemented
- **Review Needed:** No

### [2026-01-27] Metabolite Network Visualization
- **Agent:** Viz Expert
- **Decision:** Force-directed graph with D3.js, limited to top 100 correlations above threshold
- **Rationale:** Full network (600K correlations) not renderable; interactive graph shows relationships
- **Alternatives Considered:** Chord diagram, heatmap
- **Status:** Implemented
- **Review Needed:** No

### [2026-01-27] Multi-omic Integration Panel
- **Agent:** Scientific Reviewer, Viz Expert
- **Decision:** D3 force-directed network showing cytokine-biochemistry-metabolite relationships with configurable correlation threshold
- **Rationale:** Integrates three data modalities (cytokines, biochemistry, metabolomics) to reveal coordinated molecular patterns
- **Alternatives Considered:** Separate correlation heatmaps per modality, PCA biplot
- **Status:** Implemented with mock data, awaiting real integrated analysis
- **Review Needed:** Yes - verify biological relevance of cross-omic connections

### [2026-01-27] eQTL Browser Panel
- **Agent:** Scientific Reviewer
- **Decision:** Manhattan plot with interactive table and genotype effect plots
- **Rationale:** Standard eQTL visualization; enables exploration of genetic regulation of cytokine genes
- **Alternatives Considered:** LocusZoom-style regional plots, Miami plot
- **Status:** Implemented with mock data, awaiting CIMA eQTL analysis results
- **Review Needed:** Yes - eQTL data preprocessing required

---

## Phase 3: Inflammation Atlas Panels

### [2026-01-27] Inflammation Tab Structure (Updated)
- **Agent:** Orchestrator, Viz Expert
- **Decision:** Implement 10 tabs: Cell Types, Age & BMI, Disease, Demographics, Treatment Response, Disease Flow, Validation, Longitudinal, Severity, Cell Drivers
- **Rationale:** Expanded to include longitudinal disease progression, severity correlation, and cell type driver analysis
- **Alternatives Considered:** Keep minimal tabs, separate disease-specific pages
- **Status:** Implemented (UI), some panels awaiting longitudinal data
- **Review Needed:** No

### [2026-01-27] Treatment Response Panel Design
- **Agent:** Scientific Reviewer
- **Decision:** Include ROC curves, feature importance, and prediction violin plots
- **Rationale:** Comprehensive ML performance visualization; standard for clinical prediction
- **Alternatives Considered:** Confusion matrix, PR curves
- **Status:** UI Implemented, awaiting analysis pipeline output
- **Review Needed:** No

### [2026-01-27] Longitudinal Trends Panel
- **Agent:** Scientific Reviewer
- **Decision:** Line plots with individual patient trajectories grouped by response status, plus change-from-baseline boxplots
- **Rationale:** Enables identification of early predictive markers by comparing responder vs non-responder trajectories
- **Alternatives Considered:** Spaghetti plots only, slope comparison plots
- **Status:** Implemented with mock data, requires longitudinal sample metadata
- **Review Needed:** Yes - verify timepoint availability in Inflammation Atlas metadata

### [2026-01-27] Severity Correlation Panel
- **Agent:** Scientific Reviewer
- **Decision:** Disease-specific severity scores (DAS28 for RA, Mayo for IBD, PASI for Psoriasis, SLEDAI for SLE)
- **Rationale:** Uses clinically validated severity measures; enables identification of activity-correlated cytokines
- **Alternatives Considered:** Generic severity score, composite indices
- **Status:** Implemented with mock data, awaiting severity score linkage
- **Review Needed:** Yes - verify severity score availability in metadata

### [2026-01-27] Cell Type Drivers Panel
- **Agent:** Scientific Reviewer
- **Decision:** Bar chart of effect sizes, heatmap of cytokines×cell types, and variance explained importance plot
- **Rationale:** Identifies which cell populations drive disease-specific cytokine signatures
- **Alternatives Considered:** Regression coefficients only, random forest feature importance
- **Status:** Implemented, uses existing cell-type stratified analysis
- **Review Needed:** No

---

## Phase 4: scAtlas Panels

### [2026-01-27] scAtlas Tab Structure (Updated)
- **Agent:** Orchestrator, Viz Expert
- **Decision:** Implement 9 tabs: Organ Map, Cell Type Heatmap, Tumor vs Adjacent, Cancer Types, Immune Infiltration, T Cell Exhaustion, CAF Types, Organ-Cancer Matrix, Adjacent Tissue
- **Rationale:** Expanded to include CAF subtype classification, cross-tissue cancer patterns, and pre-malignant field effect analysis
- **Alternatives Considered:** Separate CAF section, keep minimal tabs
- **Status:** Implemented (UI), some panels awaiting data
- **Review Needed:** No

### [2026-01-27] Exhaustion Markers Selection
- **Agent:** Scientific Reviewer
- **Decision:** Focus on PD-1, CTLA-4, TIM-3, LAG-3, TIGIT for exhaustion; GZMB, PRF1, IFNG for cytotoxicity
- **Rationale:** Well-established checkpoint and cytotoxicity markers with clinical relevance
- **Alternatives Considered:** Include TOX, TCF1
- **Status:** Implemented
- **Review Needed:** No

### [2026-01-27] CAF Classification Panel
- **Agent:** Scientific Reviewer
- **Decision:** Three CAF subtypes: myCAF (myofibroblastic), iCAF (inflammatory), apCAF (antigen-presenting)
- **Rationale:** Well-established CAF classification from Öhlund et al. 2017 and subsequent studies; clinically relevant for TME characterization
- **Alternatives Considered:** Additional subtypes (vCAF, mCAF), custom clustering
- **Status:** Implemented with mock data, awaiting CAF subtype annotations
- **Review Needed:** Yes - verify CAF annotation methodology

### [2026-01-27] Organ-Cancer Matrix Panel
- **Agent:** Viz Expert
- **Decision:** Bubble chart with size=sample count, color=cytokine activity, axes=organ×cancer type
- **Rationale:** Enables cross-tissue comparison of cancer-specific cytokine patterns
- **Alternatives Considered:** Heatmap, parallel coordinates
- **Status:** Implemented
- **Review Needed:** No

### [2026-01-27] Adjacent Tissue Analysis Panel
- **Agent:** Scientific Reviewer
- **Decision:** Volcano plot (adjacent vs normal) plus tissue-state comparison (Normal→Adjacent→Tumor)
- **Rationale:** Identifies "field effect" cytokine changes that may precede malignant transformation
- **Alternatives Considered:** Focus on tumor-adjacent only, omit normal tissue comparison
- **Status:** Implemented with mock data, awaiting paired normal/adjacent/tumor samples
- **Review Needed:** Yes - verify sample pairing methodology

---

## Phase 5: Cross-Atlas Integration

### [2026-01-27] Cross-Atlas Section Added (Updated)
- **Agent:** Orchestrator
- **Decision:** Create dedicated Cross-Atlas section with 7 tabs: Overview, Conserved, Atlas Comparison, Cell Type Mapping, Meta-Analysis, Signature Matrix, Pathways
- **Rationale:** Expanded to include cross-cytokine correlation analysis and biological pathway enrichment
- **Alternatives Considered:** Integrated into existing sections, pathway analysis as separate page
- **Status:** Implemented (UI), awaiting integrated analysis
- **Review Needed:** No

### [2026-01-27] Cell Type Harmonization Approach
- **Agent:** Scientific Reviewer
- **Decision:** Use Sankey diagram for cell type mapping visualization
- **Rationale:** Shows flow of cell types across atlases; reveals annotation consistency
- **Alternatives Considered:** Alluvial diagram, confusion matrix
- **Status:** Implemented
- **Review Needed:** Yes - verify mapping methodology with domain expert

### [2026-01-27] Signature Correlation Matrix Panel
- **Agent:** Scientific Reviewer, Viz Expert
- **Decision:** Hierarchically clustered correlation heatmap with module identification
- **Rationale:** Reveals co-regulated cytokine programs; clusters indicate shared upstream regulators
- **Alternatives Considered:** Network visualization, PCA loadings
- **Status:** Implemented with mock clustering
- **Review Needed:** No

### [2026-01-27] Pathway Enrichment Panel
- **Agent:** Scientific Reviewer
- **Decision:** Support multiple databases (KEGG, Reactome, GO, MSigDB Hallmarks) with cytokine module filtering
- **Rationale:** Different databases capture complementary biological knowledge; module filtering enables focused analysis
- **Alternatives Considered:** Single database (KEGG only), automatic pathway selection
- **Status:** Implemented with mock enrichment data, awaiting gene set analysis
- **Review Needed:** Yes - verify enrichment methodology

---

## Critical Review Items

Items flagged for human review will be listed here:

| Date | Decision | Agent | Priority | GitHub Issue |
|------|----------|-------|----------|--------------|
| 2026-01-27 | eQTL data preprocessing | Scientific | Medium | Pending |
| 2026-01-27 | Longitudinal timepoint availability | Scientific | Medium | Pending |
| 2026-01-27 | Severity score metadata linkage | Scientific | Medium | Pending |
| 2026-01-27 | CAF subtype annotation method | Scientific | Medium | Pending |
| 2026-01-27 | Adjacent tissue sample pairing | Scientific | High | Pending |
| 2026-01-27 | Cell type mapping methodology | Scientific | Medium | Pending |
| 2026-01-27 | Multi-omic connection validation | Scientific | Low | Pending |
| 2026-01-27 | Pathway enrichment methodology | Math | Medium | Pending |

---

## Changelog

| Date | Section | Change |
|------|---------|--------|
| 2026-01-27 | All | Initial template created |
| 2026-01-27 | All | Added 6 tabs per atlas + Cross-Atlas section with 5 tabs |
| 2026-01-27 | CIMA | Added Age/BMI Stratified boxplots, Metabolite network panels |
| 2026-01-27 | Inflammation | Added Treatment Response, Disease Sankey, Cohort Validation panels |
| 2026-01-27 | scAtlas | Added Cancer Types, Immune Infiltration, T Cell Exhaustion panels |
| 2026-01-27 | Cross-Atlas | Created new section with Overview, Conserved, Comparison, Mapping, Meta-Analysis |
| 2026-01-27 | CIMA | Fixed metabolite network field mappings (rho vs spearman_rho, feature vs metabolite) |
| 2026-01-27 | Inflammation | Implemented Treatment Response visualizations (ROC, feature importance, violin) |
| 2026-01-27 | Inflammation | Implemented Cohort Validation visualizations (scatter, consistency bar) |
| 2026-01-27 | scAtlas | Implemented Cancer Types visualizations (bar chart, heatmap from cancer_comparison data) |
| 2026-01-27 | scAtlas | Implemented Immune Infiltration visualizations (stacked bar, scatter) |
| 2026-01-27 | scAtlas | Implemented T Cell Exhaustion visualizations (exhaustion/cytotoxicity heatmap, scatter) |
| 2026-01-27 | Preprocessing | Added preprocess_treatment_response() and preprocess_cohort_validation() functions |
| 2026-01-27 | scAtlas | Fixed Tumor vs Adjacent panel - corrected field name mismatches (common_cell_types, difference) |
| 2026-01-27 | All | Added responsive viewport-relative CSS for panel sizing (min/max heights, mobile breakpoints) |
| 2026-01-27 | CIMA | Added 4 new panels: Biochem Scatter, Multi-omic Integration, Population Stratification, eQTL Browser |
| 2026-01-27 | Inflammation | Added 3 new panels: Longitudinal Trends, Severity Correlation, Cell Type Drivers |
| 2026-01-27 | scAtlas | Added 3 new panels: CAF Classification, Organ-Cancer Matrix, Adjacent Tissue Analysis |
| 2026-01-27 | Cross-Atlas | Added 2 new panels: Signature Correlation Matrix, Pathway Enrichment |
| 2026-01-27 | All | Total panels expanded from 24 to 36+ tabs across all sections |
