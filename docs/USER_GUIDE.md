# CytoAtlas User Guide

Step-by-step guide for using the CytoAtlas web portal and REST API to explore cytokine and secreted protein activity signatures across 17+ million immune cells.

**Last Updated**: 2026-02-09

---

## 1. Overview

CytoAtlas provides cytokine and secreted protein activity signatures from three major single-cell atlases:

| Atlas | Cells | Key Data |
|-------|-------|----------|
| **CIMA** | 6.5M | Healthy donors, age/BMI correlations, biochemistry, metabolites |
| **Inflammation** | 6.3M | 20 diseases, treatment response, cohort validation |
| **scAtlas** | 6.4M | 35+ organs, normal vs cancer, immune infiltration, T cell exhaustion |

Each dataset contains:
- **CytoSig** activity: 44 cytokines
- **SecAct** activity: 1,249 secreted proteins
- **Multiple analysis levels**: Sample-level, cell-type level, single-cell

---

## 2. Getting Started

### Accessing CytoAtlas

**Web Portal**: Open in browser
```
http://localhost:8000/  (local development)
https://cytoatlas.example.com/  (production)
```

**REST API**: All data accessible programmatically
```bash
curl https://cytoatlas.example.com/api/v1/cima/summary
```

### Optional Registration

- **Anonymous**: Browse public data, search, 5 chat messages/day
- **Registered**: Data export, advanced queries, 1000 chat messages/day
- **Admin**: Dataset management, audit logs

To register:
1. Click "Sign Up" on the landing page
2. Enter email and password
3. Confirm email (if required)
4. Start exploring!

---

## 3. Exploring Atlas Data

### CIMA Atlas (Healthy Aging & Metabolism)

#### Objective
Understand how age and metabolic factors influence immune cell activity in healthy individuals.

#### Available Data
- **Samples**: 428 healthy donors with rich metadata
- **Cell Types**: 30+ immune subsets
- **Signatures**: CytoSig (44 cytokines) + SecAct (1,249 proteins)
- **Correlations**: Age, BMI, biochemistry (glucose, lipids, etc.), metabolites
- **eQTL**: Genetic regulation of activity signatures

#### Key Questions
1. **How does immune activity change with age?**
   - Go to "Explore" → "CIMA" → "Age Correlation"
   - View scatter plot: Age vs activity for each signature
   - Filter by cell type (e.g., "CD8 T cells")
   - Example finding: IL-7R activity increases with age in naive T cells

2. **Which cytokines correlate with metabolic markers?**
   - Go to "Explore" → "CIMA" → "Biochemistry Correlations"
   - Search for metabolite (e.g., "triglycerides")
   - View correlated cytokines by cell type
   - Example: TNF-alpha negatively correlates with HDL

3. **Compare activity across cell types**
   - Go to "Explore" → "CIMA" → "Cell Type Activity"
   - Select signature (CytoSig or SecAct)
   - View heatmap: Signatures × cell types
   - Hover for detailed statistics

#### API Examples

```bash
# Get age correlations for all signatures
curl -s 'http://localhost:8000/api/v1/cima/correlations/age?signature_type=CytoSig' | jq '.'

# Filter by specific gene
curl -s 'http://localhost:8000/api/v1/cima/correlations/age?gene=IL17A&signature_type=CytoSig' | jq '.'

# Get activity for specific cell type
curl -s 'http://localhost:8000/api/v1/cima/activity/CD8?signature_type=CytoSig' | jq '.[] | {signature, mean_activity, std_activity}'
```

---

### Inflammation Atlas (Disease Response)

#### Objective
Identify disease-specific immune cell activity patterns and predict treatment response.

#### Available Data
- **Diseases**: COVID-19, Influenza, Sepsis, Autoimmune, ~20 total
- **Samples**: 1,047 samples across main, validation, external cohorts
- **Cell Types**: 30+ immune subsets per disease
- **Signatures**: CytoSig (44 cytokines) + SecAct (1,249 proteins)
- **Treatment**: Response prediction (responder vs non-responder)
- **Validation**: Cross-cohort consistency metrics

#### Key Questions
1. **What distinguishes severe from mild disease?**
   - Go to "Explore" → "Inflammation" → "Disease Activity"
   - Select disease (e.g., "COVID-19")
   - View "Severity Comparison" tab
   - See which cytokines differ between severe/mild
   - Example: TNF-alpha, IL-6 elevated in severe COVID-19

2. **Can I predict treatment response?**
   - Go to "Explore" → "Inflammation" → "Treatment Response"
   - Select disease and cell type
   - View model feature importance
   - Check AUC and prediction accuracy
   - Example: Monocyte TNF-alpha predicts IFN-beta response in COVID-19

3. **Are findings validated across cohorts?**
   - Go to "Validate" → "Inflammation"
   - Select "Cross-Cohort Validation"
   - View consistency between main, validation, external cohorts
   - Look for signals with >0.8 correlation across cohorts
   - Example: IL-6 elevation in severe COVID-19 consistently replicated

#### API Examples

```bash
# Get disease activity patterns
curl -s 'http://localhost:8000/api/v1/inflammation/disease-activity?disease=COVID-19&signature_type=CytoSig' | jq '.'

# Get treatment response predictions
curl -s 'http://localhost:8000/api/v1/inflammation/treatment-response?disease=COVID-19' | jq '.'

# Get disease list
curl -s 'http://localhost:8000/api/v1/inflammation/diseases' | jq '.'

# Compare responder vs non-responder
curl -s 'http://localhost:8000/api/v1/inflammation/disease-activity?disease=COVID-19&comparison=treatment_response' | jq '.'
```

---

### scAtlas (Organ-Specific & Cancer)

#### Objective
Map immune cell activity across normal organs and compare with cancer environments.

#### Available Data
- **Organs**: Blood, Bone Marrow, Spleen, Lymph Node, Liver, Lung, Kidney, Brain, etc. (35+)
- **Samples**: Normal tissues from healthy donors + tumor samples
- **Cancer Types**: NSCLC, colorectal, pancreatic, ovarian, melanoma, etc.
- **Signatures**: CytoSig (44 cytokines) + SecAct (1,249 proteins)
- **Analysis**: Organ signatures, cancer infiltration, T cell exhaustion

#### Key Questions
1. **What's the baseline immune activity in healthy organs?**
   - Go to "Explore" → "scAtlas" → "Organ Signatures"
   - Select organ (e.g., "Lung")
   - View cell type × signature heatmap
   - Example: Resident macrophages in lung show high IL-10 (tolerogenic)

2. **How does cancer reshape local immunity?**
   - Go to "Explore" → "scAtlas" → "Cancer Comparison"
   - Select cancer type (e.g., "NSCLC")
   - View "Tumor vs Adjacent" tab
   - See which cytokines are reprogrammed in tumor microenvironment
   - Example: T cells in NSCLC tumors show elevated exhaustion markers (PD-1+, TIM-3+)

3. **Which T cells are exhausted in different cancers?**
   - Go to "Explore" → "scAtlas" → "T Cell Exhaustion"
   - Select cancer type
   - View exhaustion score distribution
   - Compare across CD4 T cells, CD8 T cells, regulatory T cells
   - Example: CD8 T cells in pancreatic cancer show high exhaustion

#### API Examples

```bash
# Get organ signatures
curl -s 'http://localhost:8000/api/v1/scatlas/organ-signatures?organ=Blood&signature_type=CytoSig' | jq '.'

# Get cancer comparison
curl -s 'http://localhost:8000/api/v1/scatlas/cancer-comparison?cancer_type=NSCLC&comparison=tumor_vs_adjacent' | jq '.'

# Get T cell exhaustion scores
curl -s 'http://localhost:8000/api/v1/scatlas/t-cell-exhaustion?cancer_type=NSCLC' | jq '.'

# List available organs
curl -s 'http://localhost:8000/api/v1/scatlas/organs' | jq '.'
```

---

## 4. Cross-Atlas Comparison

Compare signatures and patterns across all three atlases.

### Which Signatures Are Conserved?

1. Go to "Compare" → "Cross-Atlas"
2. Select two atlases to compare (e.g., CIMA vs Inflammation)
3. View "Pairwise Scatter" plot
4. Each point = one signature × cell type combination
5. High diagonal = conserved signature (r > 0.8)
6. Off-diagonal = atlas-specific signature

### Example Questions

**Q: Is IL-6 elevation in COVID-19 (Inflammation) similar to age-related IL-6 increase (CIMA)?**
- Go to "Compare" → "Cross-Atlas" → "CIMA vs Inflammation"
- Find IL-6 on scatter plot
- Correlation ~0.75 = moderately conserved but disease-specific increase

**Q: Are healthy immune signatures in scAtlas organs similar to CIMA baseline?**
- Go to "Compare" → "Cross-Atlas" → "CIMA vs scAtlas"
- View "Blood" organ from scAtlas vs CIMA
- Correlation ~0.90 = highly conserved (expected - same cell types)

### API Examples

```bash
# Get pairwise comparison data
curl -s 'http://localhost:8000/api/v1/cross-atlas/pairwise-scatter?atlas1=CIMA&atlas2=Inflammation&signature_type=CytoSig' | jq '.[] | {signature, cima_activity, inflammation_activity, correlation}'

# Get conserved signatures (r > 0.8 across all atlases)
curl -s 'http://localhost:8000/api/v1/cross-atlas/conserved-signatures?min_correlation=0.8' | jq '.'

# Get cell type mapping across atlases
curl -s 'http://localhost:8000/api/v1/cross-atlas/celltype-sankey?level=coarse' | jq '.'
```

---

## 5. Validation & Credibility Assessment

How confident are the activity predictions? CytoAtlas uses 5-type validation.

### Understanding Validation Grades

**A Grade (85-100)**: Highly confident
- Strong expression correlation
- Consistent across aggregation methods
- Biologically sensible patterns

**B Grade (70-84)**: Good confidence
- Moderate expression correlation
- Some consistency issues
- Mostly sensible patterns

**C Grade (50-69)**: Moderate confidence
- Weak expression correlation
- Method-dependent results
- Some questionable patterns

**F Grade (<50)**: Low confidence
- Poor or no expression correlation
- Method-dependent
- Biologically implausible patterns

### Where to Check

1. Go to "Validate" → Select atlas
2. Select signature type (CytoSig or SecAct)
3. View overall grade in "Summary" tab
4. Drill down into specific cell types in "Details" tab

### Types of Validation

| Type | Question | What It Tests |
|------|----------|---------------|
| **Type 1** | Sample-level | Do pseudobulk expression and activity correlate? |
| **Type 2** | Cell-type level | Do cell-type expression and activity align? |
| **Type 3** | Aggregation | Do pseudobulk and single-cell methods agree? |
| **Type 4** | Single-cell | Does single-cell expression correlate with activity? |
| **Type 5** | Biological | Do results match known biology (e.g., IL-17 in Th17)? |

### API Examples

```bash
# Get validation summary for CIMA
curl -s 'http://localhost:8000/api/v1/validation/summary/CIMA?signature_type=CytoSig' | jq '.'

# Get sample-level validation for IL-17A
curl -s 'http://localhost:8000/api/v1/validation/sample-level/CIMA/IL17A' | jq '.quality_grade, .expression_correlation, .activity_correlation'

# Get biological validation (known markers)
curl -s 'http://localhost:8000/api/v1/validation/biological-associations/CIMA?signature_type=CytoSig' | jq '.[] | {marker, cell_type, expected_association, observed_association}'
```

---

## 6. Using the Chat Interface

Ask questions in plain English. RAG-powered responses are grounded in CytoAtlas data.

### Example Questions

**Q: "What cytokines are elevated in severe COVID-19?"**
- Chat searches validation data for COVID-19
- Retrieves disease activity tables
- Generates response with sources

**Q: "Compare immune activation across all three atlases."**
- Chat fetches cross-atlas comparison data
- Identifies conserved signatures
- Summarizes differences

**Q: "Which cell types show T cell exhaustion markers in pancreatic cancer?"**
- Chat looks up scAtlas cancer data
- Finds exhaustion scores by cell type
- Returns ranked list with confidence scores

### Rate Limits

- **Anonymous**: 5 messages/day
- **Registered**: 1000 messages/day

### Tips

- Be specific: "cytokines in COVID-19" vs "cytokines"
- Ask follow-ups: "Why are those elevated?"
- Request sources: "Show me the validation data for that"

---

## 7. Data Export

Download data as CSV or JSON for analysis.

### Via Web Portal

1. Go to "Explore" or "Validate" page
2. Select data view (e.g., disease activity table)
3. Click "Download" button
4. Choose format (CSV or JSON)
5. File downloads to your computer

### Via API

```bash
# Export CIMA correlations as CSV
curl -s 'http://localhost:8000/api/v1/export/cima/correlations?signature_type=CytoSig' \
  -H "Accept: text/csv" \
  -o correlations.csv

# Export inflammation disease activity as JSON
curl -s 'http://localhost:8000/api/v1/export/inflammation/disease-activity?disease=COVID-19' \
  -H "Accept: application/json" \
  -o covid19_activity.json

# Export scAtlas cancer comparison
curl -s 'http://localhost:8000/api/v1/export/scatlas/cancer-comparison?cancer_type=NSCLC' \
  -H "Accept: text/csv" \
  -o nsclc_comparison.csv
```

### Column Names

Exported tables include:
- `gene` or `protein`: Signature name
- `cell_type`: Immune cell type
- `mean_activity`: Average activity (z-score)
- `std_activity`: Standard deviation
- `p_value`: Statistical significance
- `n_samples`: Number of samples
- `correlation`: Correlation coefficient (if applicable)
- `activity_diff`: Difference between groups (if comparing)

---

## 8. Advanced Usage

### Programmatic Access

#### Python

```python
import requests
import pandas as pd

BASE_URL = "http://localhost:8000/api/v1"

# Get CIMA summary
response = requests.get(f"{BASE_URL}/cima/summary")
summary = response.json()
print(f"CIMA: {summary['cells']} cells, {summary['samples']} samples")

# Get COVID-19 disease activity
response = requests.get(
    f"{BASE_URL}/inflammation/disease-activity",
    params={"disease": "COVID-19", "signature_type": "CytoSig"}
)
data = response.json()
df = pd.DataFrame(data)
print(df.head())

# Export as CSV
df.to_csv("covid19_activity.csv", index=False)
```

#### R

```r
library(httr)
library(jsonlite)

base_url <- "http://localhost:8000/api/v1"

# Get CIMA summary
response <- GET(paste0(base_url, "/cima/summary"))
summary <- fromJSON(content(response, "text"))
cat("CIMA:", summary$cells, "cells,", summary$samples, "samples\n")

# Get disease activity
response <- GET(
  paste0(base_url, "/inflammation/disease-activity"),
  query = list(disease = "COVID-19", signature_type = "CytoSig")
)
data <- fromJSON(content(response, "text"))
df <- as.data.frame(data)
head(df)

# Export as CSV
write.csv(df, "covid19_activity.csv", row.names = FALSE)
```

#### JavaScript/TypeScript

```javascript
const BASE_URL = "http://localhost:8000/api/v1";

// Get CIMA summary
const response = await fetch(`${BASE_URL}/cima/summary`);
const summary = await response.json();
console.log(`CIMA: ${summary.cells} cells, ${summary.samples} samples`);

// Get disease activity with pagination
async function fetchDiseaseActivity() {
  let allData = [];
  let offset = 0;
  const limit = 100;

  while (true) {
    const response = await fetch(
      `${BASE_URL}/inflammation/disease-activity?disease=COVID-19&offset=${offset}&limit=${limit}`
    );
    const data = await response.json();

    if (data.length === 0) break;
    allData = allData.concat(data);
    offset += limit;
  }

  return allData;
}

const allActivity = await fetchDiseaseActivity();
console.log(`Fetched ${allActivity.length} records`);
```

### Batch Processing

```bash
# Process all diseases
for disease in COVID-19 Influenza Sepsis Autoimmune; do
  curl -s "http://localhost:8000/api/v1/inflammation/disease-activity?disease=$disease" \
    -o "disease_${disease}.json"
done

# Process all organs
for organ in Blood "Bone Marrow" Spleen "Lymph Node" Liver; do
  curl -s "http://localhost:8000/api/v1/scatlas/organ-signatures?organ=$organ&signature_type=CytoSig" \
    -o "organ_${organ}.json"
done
```

---

## 9. Common Workflows

### Workflow 1: Identify Disease-Specific Cytokines

1. Go to "Explore" → "Inflammation" → "Disease Activity"
2. Select your disease of interest
3. Sort by "Activity Difference" (descending)
4. Note top 5-10 elevated cytokines
5. Check "Validation" → "Biological Associations" to confirm known markers
6. Export table for downstream analysis

### Workflow 2: Compare Treatment Response

1. Go to "Explore" → "Inflammation" → "Treatment Response"
2. Select disease and treatment type
3. View "Feature Importance" tab
4. Identify top predictive signatures
5. Check ROC curve to assess prediction accuracy
6. Use identified signatures for your own patient cohort

### Workflow 3: Map Organ Immunity

1. Go to "Explore" → "scAtlas" → "Organ Signatures"
2. Select organ of interest
3. View cell type activity heatmap
4. Identify resident vs infiltrating cell types
5. Compare with disease states in "Cancer Comparison" tab

### Workflow 4: Validate New Dataset

1. Go to "Validate"
2. Select atlas and signature type
3. Review 5-type validation summary
4. Focus on signatures with high Type 1-2 correlation
5. Avoid signatures with poor biological concordance (Type 5)
6. Use high-confidence signatures (Grade A-B) for analysis

---

## 10. FAQ

**Q: What's the difference between CytoSig and SecAct?**
- CytoSig: 44 canonical cytokines (IL-6, TNF-alpha, IFN-gamma, etc.)
- SecAct: 1,249 secreted proteins (broader coverage, includes growth factors, chemokines, etc.)

**Q: How are activities calculated?**
- Ridge regression against signature matrices using pseudo-bulk or single-cell expression
- Results are z-scores (can be negative)
- Activity diff = difference between groups (not log2FC)

**Q: Can I use this for my patients?**
- CytoAtlas provides reference signatures for healthy and diseased states
- You can score your own expression data against these signatures
- See scripts/ folder for example workflow

**Q: What if a signature has low validation grade?**
- Low grade means expression-activity mismatch
- Possible causes: Weak biological signal, technical artifact, poor model fit
- Recommendation: Use with caution, prioritize Type 5 (biological) matches

**Q: Can I submit my own dataset?**
- Go to "Submit" page
- Upload H5AD or MTX + metadata CSV
- We'll process and integrate with CytoAtlas
- Timeline: 2-4 weeks for review and QC

---

## 11. Contact & Support

**Questions or Issues?**
1. Check [docs/README.md](README.md) for documentation
2. Review [docs/ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. See [docs/API_REFERENCE.md](API_REFERENCE.md) for endpoint details
4. Email: cytoatlas@example.com

**Report Bugs**
1. Go to "Help" → "Report Issue"
2. Provide error message and steps to reproduce
3. We'll respond within 24 hours

---

Next steps: Try the tutorials on the "Help" page, or check out [docs/API_REFERENCE.md](API_REFERENCE.md) for programmatic access!
