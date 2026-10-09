# Tract‑Level Fixed Broadband Reliability Modeling & Interpretability Project

## Overview

This project demonstrates a complete, end‑to‑end data science workflow focused on modeling tract‑level fixed broadband reliability across Wisconsin. It integrates multi‑source feature engineering, tract‑quarter aggregation, supervised modeling, a multi‑model interpretability stack, and actionable business insights. The goal is to show how a data scientist can build a production‑grade modeling pipeline, explain model behavior with rigor, and translate technical findings into operational strategy.

The project centers on a custom reliability_index derived from Ookla fixed broadband measurements, enriched with demographic, socioeconomic, geographic, housing, and environmental features. Multiple models are trained and evaluated, and their interpretability artifacts are synthesized into a coherent narrative about what drives reliability and how those drivers can inform real‑world decision‑making.

This repository contains all code, documentation, and artifacts needed to understand, reproduce, and evaluate the project.

---

## Key Deliverables

- **Executive Report** — Full narrative of the modeling pipeline, interpretability analysis, and business insights.  
- **Findings & Recommendations** — A concise, decision‑ready summary of the interpretability results and operational implications.  
- **Feature Engineering & Data Sources** — Detailed documentation of all datasets, aggregation logic, feature engineering, and sanitization.  
- **Model Evaluation Results (Appendix)** — Training and cross‑validated metrics for all models.  
- **Interpretability Artifacts** — SHAP, PDP, permutation importance, and GAM smooth‑effect plots.  
- **Unified Tract‑Quarter Feature Matrix** — Final dataset used for modeling and interpretability.

---

## Data Sources

This project integrates four independent datasets, each contributing a distinct dimension of information:

### **Ookla Fixed Broadband Service**
Tile‑level download, upload, latency, tests, and devices across multiple quarters. Aggregated to tract‑quarter and used to compute the reliability_index.

### **ACS Demographic & Socioeconomic Data**
Tract‑level demographic, socioeconomic, housing, education, language, race, poverty, unemployment, and vehicle‑availability variables.

### **TIGER Geography**
Tract boundaries and geometries used to compute tract area, perimeter, and compactness, and to spatially join Ookla tiles to tracts.

### **LCDv2 Elevation & Weather**
Hourly tract‑level environmental conditions (temperature, humidity, wind, precipitation, visibility, pressure), aggregated to tract‑quarter.

---

## Feature Engineering Pipeline

The feature engineering pipeline transforms raw multi‑source data into a unified tract‑quarter feature matrix:

### **Ingestion**
- Ookla tiles filtered to Wisconsin and spatially joined to TIGER tract polygons.  
- LCDv2 hourly data loaded from DuckDB.  
- ACS + TIGER tract‑level data loaded and decoded.

### **Aggregation**
- Ookla: tile → tract‑quarter.  
- LCDv2: hourly → tract‑quarter.  
- ACS + TIGER: tract‑level (no temporal aggregation).

### **Feature Engineering**
- Numeric summaries, percentiles, ranges, CVs.  
- Weather segmentation buckets and extreme‑condition booleans.  
- Compound and storm indicators.  
- Cluster detection for sustained weather events.  
- Geometry features (area, perimeter, compactness).  
- ACS demographic, socioeconomic, housing, education, and language ratios.  
- Sampling density and threshold flags for broadband performance.

### **Sanitization**
- Dataset‑specific null handling and metadata fields.  
- Unified filtering: population_total > 0 and reliability_index not null.

### **Assembly**
LCDv2 + Ookla joined on tract‑year‑quarter, then merged with tract‑level ACS + TIGER features.

---

## Reliability Modeling

The modeling objective is to predict the tract‑quarter reliability_index using the engineered feature matrix. Five models were trained:

- Elastic Net  
- Ridge  
- Random Forest  
- Histogram‑Based Gradient Boosting  
- Generalized Additive Model  

Models were evaluated using RMSE, MAE, and R², providing a balanced view of error magnitude, error severity, and explanatory power. Full results are available in the Model Evaluation Results appendix.

---

## Interpretability Stack

A multi‑model interpretability stack was used to understand structural drivers of reliability:

- Permutation Importance  
- SHAP (ranking + beeswarm)  
- Partial Dependence Plots  
- GAM smooth effects and significance  

Across all models, the interpretability narrative is consistent:

- **Tract geometry is the dominant driver** (tract_area).  
- **Education is the most stable socioeconomic predictor** (edu_bachelors).  
- **Demographic effects are nonlinear and mediated** by geometry and education.  
- **Housing structure meaningfully modulates reliability.**  
- **Geography, weather, and economics contribute secondary nonlinear signals.**

These insights validate the modeling approach and support the business recommendations.

---

## Business Insights

The interpretability results translate into actionable strategies:

- **Geometry‑driven prioritization:** Large tracts require confirmation testing, segmentation, densification, and proactive maintenance.  
- **Education‑based targeting:** Low‑education tracts are high‑ROI candidates for support and infrastructure upgrades.  
- **Nonlinear demographic guidance:** Demographic effects should be bundled with structural signals, not used alone.  
- **Housing‑based planning:** Occupancy and structure influence reliability and should guide upgrade cycles.  
- **Geographic and weather modifiers:** Secondary but meaningful factors for forecasting and risk scoring.  
- **Economic extremes:** High‑value and low‑value tracts show distinct reliability patterns.

---

## Repository Structure

```
fixed_broadband_tract_reliability_model/
data/
    processed/
        modeling/
            (place final_feature_set.parquet here to run the modeling pipeline)
model_store/
    baseline/
    tier1/
    tier2/
modeling_outputs/
    baseline/
    tier1/
    tier2/
notebooks/
    eda/
    feature_selection/
src/
    fixed_broadband_tract_relability_model/
        feature_engineering/
        ingestion/
        modeling/
            main.py (modeling entry point)
.gitattributes
.gitignore
EXECUTIVE_REPORT.md
FEATURE_ENGINEERING_&_DATA_SOURCES.md
final_feature_set.parquet
FINDINGS_&_RECOMMENDATIONS.md
pyproject.toml
README.md
requirements.txt


```
---

## Reproducibility

To reproduce the project end‑to‑end, follow these steps:

1. **Clone the repository**  
   - `git clone <repo-url>`  
   - `cd <repo-directory>`

2. **Create and activate a virtual environment**  
   - `python -m venv venv`  
   - `source venv/bin/activate` (macOS/Linux)  
   - `venv\Scripts\activate` (Windows)

3. **Install dependencies and register the project as an editable package**  
   This step is required for the module imports used throughout the modeling and interpretability pipeline.  
   - `pip install -e .`  
   Installing in editable mode ensures that imports resolve correctly.

4. **Data availability note**  
   The raw datasets used in this project (LCDv2 hourly weather, ACS, TIGER) are not included in the repository due to size and licensing constraints.  
   To ensure reproducibility of the modeling and interpretability pipeline, the repository includes the **final_feature_set.parquet** generated after feature engineering. This allows users to run the full modeling workflow without needing access to the raw data, however it will need to be placed as indicated on the Repository Structure above.

5. **Run the full modeling pipeline**  
   After setup, all modeling, evaluation, and interpretability artifacts can be generated by running:  
   - `python main.py`  
   This script trains all models, computes evaluation metrics, and produces interpretability outputs in the `model_outputs/` directory.

The project is fully reproducible from the unified feature matrix forward. The editable install (`pip install -e .`) is required for the project to run successfully.

---

## Author

Created by **Brett** — demonstrating end‑to‑end data science capability through tract‑level reliability modeling, interpretability, and operational insight generation.

