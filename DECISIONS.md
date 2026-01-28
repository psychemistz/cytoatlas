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

### [2026-01-27] CIMA Tab Structure
- **Agent:** Orchestrator, Viz Expert
- **Decision:** Implement 6 tabs: Age & BMI, Age/BMI Stratified, Biochemistry, Metabolites, Differential, Cell Types
- **Rationale:** Groups related analyses logically; stratified boxplots provide deeper demographic insights
- **Alternatives Considered:** Separate pages per analysis, single-page with accordion
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

---

## Phase 3: Inflammation Atlas Panels

### [2026-01-27] Inflammation Tab Structure
- **Agent:** Orchestrator, Viz Expert
- **Decision:** Implement 6 tabs: Cell Types, Age & BMI, Disease, Treatment Response, Disease Flow, Validation
- **Rationale:** Covers clinical utility (treatment prediction) and scientific validation (cross-cohort)
- **Alternatives Considered:** Separate treatment analysis page
- **Status:** Implemented (UI), Treatment Response pending data
- **Review Needed:** No

### [2026-01-27] Treatment Response Panel Design
- **Agent:** Scientific Reviewer
- **Decision:** Include ROC curves, feature importance, and prediction violin plots
- **Rationale:** Comprehensive ML performance visualization; standard for clinical prediction
- **Alternatives Considered:** Confusion matrix, PR curves
- **Status:** UI Implemented, awaiting analysis pipeline output
- **Review Needed:** No

---

## Phase 4: scAtlas Panels

### [2026-01-27] scAtlas Tab Structure
- **Agent:** Orchestrator, Viz Expert
- **Decision:** Implement 6 tabs: Organ Map, Cell Type Heatmap, Tumor vs Adjacent, Cancer Types, Immune Infiltration, T Cell Exhaustion
- **Rationale:** Covers both normal tissue atlas and cancer-specific analyses
- **Alternatives Considered:** Separate cancer section
- **Status:** Implemented (UI), some panels awaiting data
- **Review Needed:** No

### [2026-01-27] Exhaustion Markers Selection
- **Agent:** Scientific Reviewer
- **Decision:** Focus on PD-1, CTLA-4, TIM-3, LAG-3, TIGIT for exhaustion; GZMB, PRF1, IFNG for cytotoxicity
- **Rationale:** Well-established checkpoint and cytotoxicity markers with clinical relevance
- **Alternatives Considered:** Include TOX, TCF1
- **Status:** Implemented
- **Review Needed:** No

---

## Phase 5: Cross-Atlas Integration

### [2026-01-27] Cross-Atlas Section Added
- **Agent:** Orchestrator
- **Decision:** Create dedicated Cross-Atlas section with 5 tabs: Overview, Conserved, Atlas Comparison, Cell Type Mapping, Meta-Analysis
- **Rationale:** Enables comparison across healthy (CIMA), disease (Inflammation), and tissue (scAtlas) contexts
- **Alternatives Considered:** Integrated into existing sections
- **Status:** Implemented (UI), awaiting integrated analysis
- **Review Needed:** No

### [2026-01-27] Cell Type Harmonization Approach
- **Agent:** Scientific Reviewer
- **Decision:** Use Sankey diagram for cell type mapping visualization
- **Rationale:** Shows flow of cell types across atlases; reveals annotation consistency
- **Alternatives Considered:** Alluvial diagram, confusion matrix
- **Status:** Implemented
- **Review Needed:** Yes - verify mapping methodology with domain expert

---

## Critical Review Items

Items flagged for human review will be listed here:

| Date | Decision | Agent | Priority | GitHub Issue |
|------|----------|-------|----------|--------------|
| - | - | - | - | - |

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
