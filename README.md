# Fixed Broadband Tract Reliability Model  
**Status:** *Active Development — Early Stage*

The **Fixed Broadband Tract Reliability Model** is a data science and machine learning project focused on building a census tract‑level reliability index and predictive model. The project integrates multiple public datasets including Ookla network performance tiles, FCC advertised service data, and tract‑level weather climatology to quantify and model broadband reliability at a granular geographic level.

This project is the third in a series of applied data engineering and data science builds, following prior work on large‑scale weather ingestion pipelines and tract‑level climatology data modeling. It continues that progression by combining geospatial analysis, feature engineering, and predictive modeling into a unified reliability assessment framework.

---

## Project Goal

The primary objective is to develop a **tract‑level broadband reliability index** and a corresponding **machine learning model** that explains and predicts reliability using:

- real‑world network performance (download, upload, latency)  
- advertised service availability and speed tiers  
- weather patterns and climatology  
- demographic and geographic context  

The final output will include:

- a composite reliability index  
- tract‑level feature matrix  
- exploratory analysis of reliability drivers  
- an interpretable ML model  
- tract‑level reliability insights and visualizations  

---

## High‑Level Architecture

The project follows a structured, modular workflow:

### 1. Ingestion & Filtering
Ingestion of warehoused and manually downloaded datasets, including:
- Ookla fixed broadband tiles  
- FCC advertised service summaries  
- tract‑level weather climatology (from prior project)  
- ACS demographics + TIGER tract geometry  

All data is filtered and aggregated to the **census tract** level.

### 2. Feature Engineering
Creation of tract‑level features capturing:
- network performance  
- technology availability  
- speed tier penetration  
- weather conditions  
- demographic and geographic context  

### 3. Reliability Index Construction
Development of a composite reliability metric combining:
- normalized download/upload speeds  
- latency  
- test/device density  
- stability indicators  

### 4. Exploratory Data Analysis
Spatial and statistical analysis of reliability patterns across the targeted geographical area.

### 5. Modeling & Evaluation
Training and evaluating an ML model to explain and predict tract‑level reliability.

### 6. Reporting
Final synthesis of insights, visualizations, and recommendations.

---

## Development Status

This project is currently in **active development**.  
Early scaffolding, ingestion planning, and environment setup are complete.  
Feature engineering, modeling, and reporting stages will be added iteratively.

Visitors can expect regular updates as the project progresses toward a full tract‑level reliability modeling pipeline.
