# Findings & Recommendations

## Context

This document summarizes the core interpretability findings and business recommendations from the tract‑level broadband reliability modeling project. These findings reflect patterns that were stable across Random Forest, Histogram‑Based Gradient Boosting, and Generalized Additive Models, and are grounded in SHAP, permutation importance, PDP curves, and GAM smooth effects. The goal is to provide a clear, decision‑ready view of what drives fixed broadband reliability and how those drivers translate into operational strategy. 

Modeling was performed for, and its results apply to, the US state of Wisconsin.

---

## Key Findings

### 1. Geometry Is the Dominant Driver of Reliability
Across all interpretability methods, **tract_area** is the strongest predictor of reliability. Smaller tracts consistently exhibit higher reliability, while larger tracts show sharp thresholded declines and persistently lower performance. Geometry interacts with nearly every other feature, shaping the reliability landscape more strongly than socioeconomic, demographic, or housing variables.

Housing structure also contributes meaningful nonlinear effects. Multi‑unit environments tend to correlate with more stable reliability, while single‑family‑dominant tracts show greater variability.

### 2. Education Is the Most Stable Socioeconomic Predictor
**pct_bachelors** shows a clean, monotonic positive relationship with reliability across all models. Higher educational attainment consistently corresponds to more reliable broadband performance. Other education variables follow similar patterns with diminishing returns or thresholded declines.

### 3. Demographic Effects Are Nonlinear and Mediated
Demographic variables matter, but their effects are nonlinear and mediated by geometry, education, and housing.

- **pct_black** shows a complex nonlinear pattern.  
- **pct_65_plus** exhibits a U‑shaped relationship: mid‑range elderly density corresponds to lower reliability.  

These effects are real but secondary to geometry and education.

### 4. Housing Composition Meaningfully Modulates Reliability
Housing variables show smooth nonlinear effects, especially in GAM:

- **housing_structure_universe** shows a wave‑like positive trend.  
- **pct_housing_occupied** shows a positive relationship.  
- **pct_single_family** shows reversals depending on density.

These patterns indicate that built environment characteristics influence reliability in structured ways.

### 5. Geography, Weather, and Economics Are Secondary Drivers
Geographic features contribute modest nonlinear effects.  
Weather variables show weak oscillatory patterns due to coarse temporal resolution.  
Economic variables matter only at extremes and do not serve as reliable prioritization signals.

---

## Recommendations

### 1. Tract Geometry: Two‑Step Strategy for Large‑Area Tracts

#### Step 1 — Confirmation Phase (Mobile Testing & Incentivized Sampling)
Before committing infrastructure resources, verify whether modeled underperformance reflects true reliability issues or sparse sampling. For candidate tracts—especially large, sparse tracts—deploy:

- Mobile testing units  
- Incentivized resident testing  
- Short‑term measurement campaigns  
- Device‑based testing kits or app diagnostics  

This strengthens sampling confidence and validates whether modeled risk is real.

#### Step 2 — Post‑Confirmation Operational Actions
Once risk is confirmed:

- Segment large tracts into operational sub‑regions  
- Increase proactive maintenance  
- Prioritize densification projects  
- Apply targeted upgrades based on housing and demographic context  

---

### 2. Education: Targeted Reliability Programs for Low‑Education Tracts
- Deploy enhanced customer‑support programs in low‑education tracts.  
- Prioritize infrastructure upgrades in these areas due to structurally lower reliability.  
- Use education as a stable segmentation variable for operational planning.

### 3. Demographics: Bundled Interventions for Nonlinear, Mediated Effects
- Avoid standalone demographic targeting; effects are mediated by geometry, education, and housing.  
- Focus on **mid‑range pct_65_plus** tracts, which show the lowest reliability.  
- Monitor high‑pct_black tracts for interaction‑driven reliability risks, paired with structural signals.

### 4. Built Environment: Infrastructure Planning Based on Housing Composition
- Increase node density in high‑occupancy tracts.  
- Investigate low‑pct_single_family tracts for structural reliability issues.  
- Use housing composition as a planning dimension for upgrade cycles.

### 5. Geography: Incorporate Regional Nonlinear Patterns
- Integrate geographic segmentation into reliability forecasting.  
- Prioritize infrastructure hardening in geographic regions with lower modeled reliability.

### 6. Weather: Secondary Modifiers for Reliability Risk
- Use weather patterns as secondary modifiers in reliability forecasting.  
- Ensure increased monitoring during extreme heat periods (e.g., temp_above_90_hours).

### 7. Economics: Focus on Extreme‑Value Tracts
- Target high‑value housing tracts for premium reliability offerings.  
- Monitor low‑value tracts for reliability dips.  
- Treat economic variables as meaningful only at extremes.

---

## Reference

For full methodological detail, interpretability artifacts, and reliability_index construction, see EXECUTIVE_REPORT.md.
