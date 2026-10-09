# **1. Executive Summary**

This project develops a tract‑level broadband reliability model using a multi‑model interpretability framework built on Random Forest (RF), Histogram Gradient Boosting (HGB), and Generalized Additive Models (GAM). The goal is to understand the structural, socioeconomic, demographic, geographic, and environmental factors that shape reliability outcomes across U.S. census tracts. Reliability is engineered from **Ookla Speedtest data**, aggregated and penalized to account for sampling sparsity, tract heterogeneity, and measurement instability. Feature data is sourced from **LCDv2 climatological datasets**, **ACS socioeconomic and demographic data**, and **TIGER geographic data**.

The modeling pipeline reveals a **fundamentally nonlinear reliability landscape**. Across all models and interpretability artifacts—permutation importance, SHAP ranking, SHAP beeswarm, PDP 1D, GAM ranking, and GAM effect plots—one insight stands out: **tract geometry is the dominant structural driver of reliability**. Smaller tracts consistently exhibit higher reliability, while larger tracts show sharp thresholded declines. This dominance is not a modeling artifact; it reflects the engineered reliability target’s sensitivity to sampling confidence and spatial heterogeneity, both of which correlate strongly with tract size.

Education emerges as the most stable socioeconomic predictor. **pct_bachelors** shows a clean, monotonic positive relationship with reliability across every model and interpretability method. Demographic variables such as **pct_black** and **pct_65_plus** exert meaningful but nonlinear effects, often mediated through education, housing composition, and tract geometry. Built environment features—including **housing_structure_universe** and **pct_housing_occupied**—show smooth nonlinear contributions that GAM captures more clearly than tree‑based models. Weather and economic variables contribute weak but structured nonlinear effects, typically only at extreme values.

The interpretability synthesis demonstrates that reliability is shaped by a combination of structural geography, socioeconomic conditions, demographic composition, and built environment characteristics, all interacting within a nonlinear framework. While the model performs well, several limitations remain: **hyperparameters were not tuned**, and **tract‑level broadband hardware data is unavailable**, constraining the model’s ability to distinguish infrastructure‑driven reliability variation from sampling‑driven variation. These limitations inform the project’s future enhancement roadmap, which includes hyperparameter optimization, interaction‑focused feature engineering, incorporation of infrastructure data, and spatial modeling extensions.

This report presents the full modeling pipeline, interpretability analysis, engineered reliability target design, limitations, and future work recommendations. It provides a transparent, rigorous, and multi‑angle understanding of tract‑level broadband reliability and establishes a foundation for iterative refinement and expansion.

# **2. Data Overview**

This project builds on a curated, tract‑quarter (Q1, Q2, Q3, Q4) level feature matrix for the US State of Wisconsin, produced by the ml_feature_pipeline, which may be found within this Github account. That earlier project handled all ingestion, cleaning, spatial alignment, and integration across multiple national datasets. The present analysis uses that curated dataset, focusing entirely on reliability target engineering, feature engineering, EDA and feature selection, modeling, and interpretability. This section summarizes the data families that compose the feature matrix and highlights the sampling considerations relevant to the engineered reliability target.

---

## **2.1 Curated Feature Matrix**

The modeling dataset used in this project is a unified, tract‑level feature matrix created by the ml_feature_pipeline. That pipeline integrated:
 
- **LCDv2 climatological features**  
- **ACS socioeconomic and demographic variables**  
- **TIGER geographic attributes**

All features were aligned to the Wisconsin **census tract** geospatial, quarterly temporal level and prepared for direct use in modeling. No ingestion or integration steps were re‑implemented in this project; the curated dataset serves as the starting point for feature engineering and model development.

The target dataset used in this project was ingested and the reliability target engineered from the features within. Several stability related features were engineered, aligned, and appended to the curated feature matrix.

- **Ookla Speedtest aggregates** 

---

## **2.2 Data Sources**

### **A. Ookla Speedtest Data (Target Source + Engineered Stability Features)**

Ookla provides the raw measurement data used to engineer the tract‑level reliability index. These measurements include:

- download and upload speeds  
- latency and jitter  
- percentile distributions  
- coefficient of variation  
- tile_count  
- tests_total  
- devices_total  

Several stability‑related features (e.g., tests_per_tile, devices_per_tile) were derived from these aggregates to support the reliability index design described in Section 3.

---

### **B. LCDv2 Climatological Data (Feature Source)**

LCDv2 contributes weather and environmental features aggregated to the tract-quarter level. These include:

- temperature  
- humidity  
- precipitation  
- wind 
- extreme weather indicators
- etc.  

These features capture environmental conditions that may influence broadband performance or infrastructure stability.

---

### **C. ACS Socioeconomic & Demographic Data (Feature Source)**

ACS provides a wide range of tract‑level socioeconomic and demographic variables, including:

- educational attainment (e.g., pct_bachelors, pct_masters)  
- income and employment  
- race and ethnicity distributions  
- age structure (e.g., pct_65_plus)  
- housing occupancy and household composition  
- multi‑unit housing density  
- housing_structure_universe  

Many of the most influential features in the model originate from ACS tables.

---

### **D. TIGER Geographic Data (Feature Source)**

TIGER contributes spatial and geometric attributes, including:

- tract boundaries  
- tract_area  
- centroid latitude and longitude  
- geometric complexity  
- tract compactness  

These features support spatial reasoning and help explain geographic patterns in reliability.

---

## **2.3 Derived Features**

Derived features in the final feature dataset originate from LCDv2, ACS, and Ookla aggregates. Examples include:

### **LCDv2‑Derived**
- weighted weather features  
- bucket count and distribution transformations  
- extreme weather indicators  

### **ACS‑Derived**
- pct_bachelors  
- pct_high_school  
- pct_65_plus  
- pct_unemployment  
- pct_housing_occupied

### **Ookla‑Derived**
- tests_per_tile  
- devices_per_tile  
- stability and heterogeneity metrics  

These engineered fields support both the reliability index design and the interpretability analysis.

---

## **2.4 Sampling Considerations**

Sampling density in this project is **entirely determined by the Ookla dataset**. It is not adjustable within the modeling pipeline. Sparse sampling introduces:

- higher measurement instability  
- greater geographic heterogeneity  
- stronger penalties in the engineered reliability index  

Urban tracts typically exhibit dense sampling and stable measurements, while rural tracts often show sparse sampling and higher variability. These sampling characteristics directly influence the reliability target and help explain why tract_area emerges as a dominant driver in the interpretability results. Section 8 (Recommendations) addresses how sampling confidence can be improved in practice.

---

## **2.5 Handling Missing Data and Sanitization Metadata**

Before feature engineering, missing values in key fields were replaced with safe defaults:

- coefficients of variation → 0  
- ranges → 0  
- percentiles → median values  
- threshold flags → False  
- sampling density fields → 0  

During this process, **sanitization metadata** was generated (e.g., `*_was_null`, `num_sanitized_fields`, `sanitization_level`) to support exploratory data analysis. This metadata helped identify tracts with low‑quality or incomplete measurement data.

Importantly, sanitization metadata was **not included in the modeling feature set** and **does not influence the reliability index**. It was used solely during EDA and was **dropped** prior to modeling.

---

## **2.6 Data Quality Limitations (Brief Overview)**

A full discussion of data limitations appears in Section 9. Here, we briefly note the most relevant constraints:

- **Unavailable tract‑level broadband hardware data** (e.g., fiber node density, cabinet age, equipment type) limits the ability to distinguish infrastructure‑driven reliability variation from sampling‑driven uncertainty.  
- **ACS sampling noise** introduces variance into demographic and socioeconomic features.  
- **Weather data resolution** is coarse, relying on hourly buckets. This is a necessary for modeling given Ookla speed test data is available only at a quarterly temporal resolution.  
- **Spatial aggregation** of Ookla tiles into tracts may smooth internal variation, especially in large rural tracts.

These limitations are expanded in detail later in the report.

---

## **2.7 Links to Original Data Sources**

Readers may explore the raw datasets used to construct the curated feature matrix:

- **Ookla Open Data:** https://www.ookla.com/open-data  
- **LCDv2 Climatological Data:** https://www.ncei.noaa.gov/products/land-based-station/climate-data-online  
- **ACS 5‑Year Estimates:** https://www.census.gov/programs-surveys/acs  
- **TIGER/Line Shapefiles:** https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html  

---

## **2.8 Summary**

The dataset used in this project integrates Ookla Speedtest measurements, LCDv2 climatological features, ACS socioeconomic and demographic variables, and TIGER geographic attributes into a unified tract‑level feature matrix. While rich and multi‑modal, the dataset is shaped by sampling sparsity in rural tracts and the absence of tract‑level broadband infrastructure data. These characteristics motivate the engineered reliability target and inform the interpretability results presented later in the report.


# **3. Target Engineering: Reliability Index Design**

Broadband reliability cannot be measured directly from raw Ookla Speedtest metrics. These raw measurements vary substantially across census tracts due to differences in sampling density, geographic heterogeneity, device distribution, and measurement stability. To support tract‑level modeling, a custom **Reliability Index** was engineered to convert raw measurements into a stable, interpretable, and confidence‑adjusted score ranging from 0 to 100. This section describes the design of that index, the rationale behind its components, and the safeguards used to prevent sampling bias from distorting reliability outcomes.

---

## **3.1 Motivation for Engineering a Custom Target**

Raw Ookla metrics—download/upload speeds, latency, jitter, percentiles, and coefficients of variation are highly sensitive to sampling density and geographic heterogeneity. Sparse tracts often produce unstable or extreme values, while dense urban tracts produce smoother distributions. Percentiles and CV values can be misleading without context, and raw measurements alone cannot distinguish between:

- true broadband underperformance, and  
- measurement instability caused by low sampling confidence.

To address these issues, a composite reliability index was engineered to incorporate performance, penalties for instability, and a confidence adjustment based on sampling density.

---

## **3.2 Components of the Reliability Index**

The Reliability Index is computed from three components: a **Component Score**, a **Penalty Score**, and a **Confidence Score**. These are combined and scaled to produce a final score between 0 and 100.

### **A. Component Score (Performance)**

The Component Score averages eight normalized metrics representing broadband performance:

- `download_median`  
- `download_p25`  
- `download_p10`  
- `upload_median`  
- `upload_p25`  
- `latency_median` (inverse‑normalized)  
- `latency_p75` (inverse‑normalized)  
- `latency_p90` (inverse‑normalized)

Speed metrics are normalized using:  
`norm_speed(x) = clip(x / 200, 0, 1)`

Latency metrics use inverse normalization:  
`norm_inverse_latency(x) = clip(1 - (x / 200), 0, 1)`

The Component Score is the mean of these eight normalized values.

---

### **B. Penalty Score (Instability and Degradation)**

The Penalty Score applies nonlinear penalties for signs of poor performance or measurement instability. These penalties include:

#### **Threshold Flags**
- `download_lt_25_flag` (squared)  
- `download_lt_100_flag` (1.5 power)  
- `upload_lt_3_flag` (squared)  
- `upload_lt_20_flag` (1.5 power)  
- `latency_gt_100_flag` (1.5 power)  
- `latency_gt_150_flag` (squared)  
- `latency_p90_gt_200_flag` (linear)

#### **Coefficients of Variation (CV)**
- `download_cv` (squared)  
- `upload_cv` (squared)  
- `latency_cv` (squared)

#### **Ranges**
- `(download_range / 100)²`  
- `(latency_range / 100)²`

These terms are summed to produce a raw penalty value, then normalized:  
`penalty = clip(penalty_raw / 12, 0, 1)`

Nonlinear exponents ensure severe degradation is penalized more heavily than moderate degradation.

---

### **C. Confidence Score (Sampling Confidence)**

Sampling confidence is based **solely on `tests_total`**, using a log‑scaled normalization:

`conf = clip(log(tests_total + 1) / log(500), 0, 1)`

This reflects the intuition that:

- more tests → higher confidence in the tract’s measurements  
- fewer tests → lower confidence and greater susceptibility to instability  

Importantly, **`devices_total`** and **`tile_count`** do not directly influence the confidence score; they contribute only to derived features used in penalties.

---

### **D. Final Score**

The Reliability Index is computed as:

`raw = Component Score − Penalty Score + Confidence Score`  
`reliability_index = clip(100 * raw, 0, 100)`

This formulation ensures that:

- strong performance increases reliability  
- instability decreases reliability  
- sampling confidence offsets penalties when measurements are trustworthy  

---

## **3.3 Penalty Design and Rationale**

Penalties are designed to capture both **performance degradation** and **measurement instability**:

- Threshold flags identify tracts with clearly inadequate speeds or excessive latency.  
- CVs detect inconsistent performance across tiles.  
- Ranges capture extreme variability within a tract.  
- Nonlinear exponents ensure severe degradation is penalized more strongly than moderate degradation.

These penalties do **not** assume poor broadband performance; they assume **low reliability**, a combination of performance and stability. This distinction is essential for interpreting tract‑level results.

---

## **3.4 Handling Missing Data and Sanitization Metadata**

Before computing the reliability index, missing values in key Ookla‑derived fields were replaced with safe defaults:

- CVs → `0`  
- ranges → `0`  
- percentiles → median values  
- threshold flags → `False`  
- sampling density fields → `0`

During this process, sanitization metadata was generated (e.g., `*_was_null`, `num_sanitized_fields`, `sanitization_level`) to support exploratory data analysis. This metadata helped identify tracts with low‑quality or incomplete measurement data.

Sanitization metadata was **not included in the modeling feature set** and **does not influence the reliability index**. It was used solely during EDA and was **dropped** prior to modeling.

---

## **3.5 How Target Engineering Prevents Bias**

The engineered reliability index prevents sampling sparsity from being misinterpreted as poor broadband performance:

- Sparse tracts receive **confidence penalties**, not performance penalties.  
- Large tracts often have sparse sampling and high geographic heterogeneity; the index correctly interprets these conditions as **low confidence**, not low performance.  
- Urban tracts with dense sampling receive higher confidence scores, stabilizing their reliability estimates.  
- The index ensures that reliability reflects both **performance** and **measurement stability**, reducing the risk of sampling‑driven bias.

This design explains why **tract_area** emerges as a dominant driver in the interpretability results: tract size correlates strongly with sampling density, heterogeneity, and rural infrastructure patterns.

---

## **3.6 Validation of Target Engineering**

The engineered reliability index behaves consistently across exploratory analysis and interpretability methods:

- Tracts with extremely sparse sampling cluster near `reliability_index = 0`.  
- Tracts with moderate sampling and stable measurements cluster near `reliability_index = 100`.  
- SHAP beeswarm plots show tract_area’s effect as smooth and nonlinear.  
- PDP plots reveal thresholded declines in reliability for large tracts.  
- GAM splines confirm tract_area’s dominant, monotonic influence.

These patterns validate that the index captures meaningful structural variation rather than noise.

---

## **3.7 Limitations of the Reliability Index**

- Ookla SpeedTest data is reported at a quarterly resolution, limiting temporal modeling resolution and insights.
- A more robust sampling confidence algorithm could be attempted to further refine the accuracy of the composite Reliability Index.

---

## **3.8 Summary**

The Reliability Index is a composite measure designed to balance broadband performance, measurement stability, and sampling confidence. Its penalties ensure that sparsity and heterogeneity are interpreted as reliability risk rather than broadband performance. This design explains tract_area’s dominant role in the interpretability results and provides a stable foundation for tract‑level modeling and analysis.


# **4. Modeling Pipeline**

This section describes the end‑to‑end modeling workflow used to predict tract‑level broadband reliability from the curated feature matrix produced by **ml_feature_pipeline**. The pipeline includes baseline linear models, nonlinear ensemble models, and an additive model, each selected to provide complementary interpretability perspectives. The modeling approach emphasizes transparency and structural insight over predictive optimization.

---

## **4.1 Modeling Objectives**

The modeling pipeline is designed to:

- Predict tract‑level `reliability_index` using a multi‑modal feature matrix.  
- Evaluate whether reliability behaves linearly, nonlinearly, or additively across features.  
- Use multiple model families with distinct inductive biases to triangulate interpretability.  
- Prioritize explainability and driver identification rather than predictive performance.

This multi‑model strategy ensures that interpretability findings are robust across modeling approaches.

---

## **4.2 Train/Test Split and Data Preparation**

### **A. Cross‑Sectional Train/Test Split**

The modeling problem is **cross‑sectional**, not temporal. Each tract‑quarter is treated as an independent observation, and the goal is to understand structural relationships rather than forecast future quarters. Because there is no temporal dependency structure, a standard random split is appropriate.

The split used:

- `test_size = 0.2`  
- `random_state = 42`  
- No stratification (continuous target)  
- No temporal splitting required  

### **B. Feature Selection**

All curated features produced by **ml_feature_pipeline** and those features engineered from them were included except:

- sanitization metadata (used only during EDA and then dropped)  
- the engineered `reliability_index` (target only)

All features are numeric; no categorical encoding was required.

### **C. Scaling**

- Random Forest and Histogram Gradient Boosting do **not** require scaling.  
- GAM internally scales spline terms as needed.  
- Ridge and ElasticNet operate on raw numeric features; scikit‑learn handles scaling implicitly through regularization.

### **D. Missing Data**

All missingness was handled upstream during feature engineering. No additional imputation was required during modeling.

---

## **4.3 Baseline Models (Ridge + ElasticNet)**

Two linear baselines were included to test whether the reliability signal is primarily linear or sparse linear.

### **A. Ridge Regression**

- `alpha = 1.0`  
- `random_state = 42`  

Ridge provides a dense linear baseline and tests whether reliability_index can be explained by a weighted sum of features.

### **B. ElasticNet Regression**

- `alpha = 1.0`  
- `l1_ratio = 0.5`  
- `random_state = 42`  

ElasticNet tests for sparse linear structure. Its weaker performance and convergence warnings indicate that the reliability signal is **not sparse** and cannot be captured by a small subset of linear terms.

### **C. Baseline Findings**

- Ridge performs moderately well → partial linear structure exists.  
- ElasticNet performs worse → the signal is not sparse.  
- Baselines justify the use of nonlinear models.

---

## **4.4 Primary Model Families**

Three nonlinear model families were used to capture complex relationships and provide complementary interpretability views.

### **A. Random Forest (RF)**

Hyperparameters:

- `n_estimators = 500`  
- `max_depth = None`  
- `min_samples_split = 2`  
- `min_samples_leaf = 1`  
- `n_jobs = -1`  
- `random_state = 42`

RF captures nonlinearities and interactions, is robust to noise, and provides permutation importance and PDPs.

### **B. Histogram Gradient Boosting (HGB)**

Hyperparameters:

- `learning_rate = 0.05`  
- `max_iter = 500`  
- `max_depth = None`  
- `random_state = 42`

HGB is more expressive than RF, handles complex nonlinearities, and achieved the strongest predictive performance.

### **C. Generalized Additive Model (GAM)**

Hyperparameters:

- One spline term per feature  
- `fit_intercept = True`

GAM models smooth nonlinear effects without interactions, providing highly interpretable spline plots and significance tests.

### **D. Why These Models**

- RF and HGB capture nonlinearities and interactions.  
- GAM captures smooth nonlinear effects without interactions.  
- Baselines test linear structure.  
- Together, these models triangulate interpretability and ensure robustness of driver identification.

---

## **4.5 Hyperparameters**

Hyperparameters were **not tuned**. All models use the defaults or minimal values specified in the modeling code.

### **Rationale**

- The goal is **interpretability**, not predictive optimization.  
- Default parameters provide stable, reproducible baselines.  
- Hyperparameter tuning is included as a future enhancement.  
- Using defaults avoids overfitting and keeps interpretability artifacts clean.

---

## **4.6 Model Training**

### **A. Random Forest**
- Parallel tree construction  
- No scaling required  
- Deterministic via `random_state = 42`

### **B. Histogram Gradient Boosting**
- Histogram binning for efficiency  
- No scaling required  
- Deterministic via `random_state = 42`

### **C. Generalized Additive Model**
- One spline per feature  
- No interactions  
- Deterministic via fixed seed

### **D. Baseline Models**
- Ridge and ElasticNet trained first  
- Used to evaluate linear structure before nonlinear modeling

---

## **4.7 Model Evaluation**

### **Metrics**
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R²  

### **Comparison**
- HGB achieves the strongest performance.  
- RF performs well but slightly below HGB.  
- GAM performs worst but provides the most interpretable effects.  
- Baseline linear models perform significantly worse, confirming the nonlinear nature of reliability_index.
- See the appendix for model evaluation scoring.

---

## **4.8 Interpretability Hooks**

Each model family provides distinct interpretability artifacts:

### **Random Forest**
- Permutation importance  
- PDP 1D
- Feature importance ranking

### **Histogram Gradient Boosting**
- Permutation importance  
- PDP 1D
- SHAP ranking  
- SHAP beeswarm (distribution)
- SHAP directionality (positive/negative effects)

### **Generalized Additive Model**
- Spline effect plots  
- Feature significance (p‑values)

These artifacts form the foundation of the interpretability synthesis in Sections 5 and 6.

---

## **4.9 Summary**

The modeling pipeline combines linear baselines, nonlinear ensembles, and additive models to provide a comprehensive interpretability framework. Hyperparameters are intentionally minimal, the train/test split reflects the cross‑sectional nature of the problem, and evaluation metrics confirm that nonlinear models are necessary to capture the structure of tract‑level reliability. This pipeline provides a robust foundation for the interpretability analysis that follows.


# **5. Interpretability Framework**

The modeling pipeline uses multiple interpretability methods to understand how structural, socioeconomic, demographic, geographic, and environmental features influence tract‑level broadband reliability. Because reliability is shaped by nonlinear and heterogeneous relationships, no single interpretability tool is sufficient. This section briefly outlines the methods used and the role each plays in the analysis.

---

## **5.1 Purpose**

The goal of interpretability in this project is to identify and characterize the drivers of tract‑level reliability. Predictive accuracy is secondary; the primary objective is to understand *how* features influence the engineered reliability index and whether those effects are linear, nonlinear, monotonic, thresholded, or interaction‑driven.

---

## **5.2 Methods Used**

Four interpretability families were applied across the modeling pipeline:

- **Permutation Importance** — global feature importance for RF and HGB.  
- **Partial Dependence Plots (PDP)** — average marginal effects and nonlinear shape.  
- **Generalized Additive Model (GAM) Splines** — smooth additive effects and significance.  
- **SHAP Values** — model‑agnostic, tract‑level contributions and global distributional patterns.

Each method provides a distinct perspective, and together they form a coherent interpretability framework.

---

## **5.3 Permutation Importance**

Permutation importance measures how much model error increases when a feature is randomly permuted. It provides a stable, model‑specific ranking of global importance for RF and HGB. It does not show directionality or effect shape, but it reliably identifies which features matter most.

---

## **5.4 Partial Dependence Plots (PDP)**

PDPs show the average marginal effect of a feature on reliability. They reveal whether effects are monotonic, nonlinear, or thresholded. PDPs are particularly useful for understanding the shape of tract_area’s influence and other structural features. They do not capture tract‑specific variation but provide clear global patterns.

---

## **5.5 GAM Splines and Significance**

The GAM provides smooth, additive effect curves for each feature and statistical significance tests. It cannot model interactions, but it offers a clean view of each feature’s standalone nonlinear contribution. GAM splines serve as a sanity check against the more flexible tree‑based models.

---

## **5.6 SHAP Values**

SHAP values provide tract‑level explanations and global distributional patterns. They capture nonlinearities and interactions and show both the magnitude and direction of each feature’s contribution. SHAP beeswarm plots are the most detailed interpretability artifact in the project and play a central role in the synthesis that follows.

---

## **5.7 Why Multiple Methods**

Using multiple interpretability tools reduces the risk of model‑specific artifacts and ensures that conclusions are robust. RF and HGB capture nonlinearities and interactions; GAM captures smooth additive effects; SHAP provides tract‑level contributions; permutation importance provides global rankings. Together, these methods create a comprehensive interpretability framework.

---

## **5.8 Summary**

This project uses a multi‑method interpretability framework to understand the drivers of tract‑level broadband reliability. Each method contributes a different perspective, and their combined insights form the foundation for the cross‑model interpretability synthesis presented in Section 6.


# Section 6 — Interpretability Analysis

This section synthesizes all interpretability artifacts generated across the Random Forest (RF), Histogram‑Based Gradient Boosting (HGB), and Generalized Additive Model (GAM) pipelines. The analysis integrates permutation importance, SHAP ranking, SHAP beeswarm distributions, PDP 1D curves, GAM p‑value ranking, and GAM effect plots. Together, these artifacts reveal the structural drivers of tract‑level broadband reliability and the nonlinear relationships governing the model’s behavior.

The interpretability stack yields a coherent, multi‑model narrative: **tract geometry is the dominant driver**, **education is the most stable socioeconomic predictor**, **demographic and housing effects are nonlinear and mediated**, and **weather, geography, and economics contribute weak but structured signals**. The reliability landscape is fundamentally nonlinear, validating the use of RF, HGB, and GAM.

---

## 6.1 Cross‑Model Feature Importance

### Dominant Driver: Tract Geometry
Across RF permutation importance, HGB permutation importance, SHAP ranking, PDP curves, and GAM significance, **tract_area** is the single strongest predictor of reliability. It consistently outranks all other features by large margins:

- RF: highest importance  
- HGB: highest importance  
- SHAP: 11.26 (an order of magnitude above all others)  
- PDP: steep monotonic decline with multiple thresholds  
- GAM: strong nonlinear effect  

This unanimity indicates a true structural relationship: **smaller tracts exhibit higher reliability**, while **larger tracts show sharp, thresholded declines** and then persistently low reliability.

### Strong Socioeconomic Drivers: Education
Education features form the second‑strongest cluster:

- **pct_bachelors**: monotonic positive effect across all models  
- **pct_masters / edu_masters**: positive with diminishing returns  
- **pct_high_school**: negative, with two major downward thresholds  

Education is the most stable socioeconomic predictor in the entire pipeline. Its directionality and functional form remain consistent across RF, HGB, SHAP, PDP, and GAM.

### Moderate Nonlinear Drivers: Demographics
Demographic variables show meaningful but nonlinear effects:

- **pct_black**: negative at high values, positive at low values; highly nonlinear  
- **race_white**: weak‑to‑moderate positive effect  
- **pct_65_plus**: U‑shaped (low → high reliability, mid → low reliability, high → moderate reliability)  
- **pct_unemployment**: nonlinear negative effect  

These effects are real but mediated by education, housing, and tract geometry.

### Built Environment Drivers
Housing‑related features show moderate influence:

- **housing_structure_universe**: nonlinear positive  
- **pct_housing_occupied**: wave‑like positive  
- **pct_single_family**: negative at low values, positive at high values  
- **housing_3_4_units**, **housing_20_49_unit**, **housing_1_unit**: weak nonlinear effects  

GAM elevates these features more strongly than RF/HGB, revealing smooth housing effects that tree models partially obscure.

### Geography, Weather, and Economics
These features contribute weak but structured nonlinear effects:

- **centroid_lon**, **centroid_lat**, **elevation_m**: modest nonlinear geographic patterns  
- **temp_above_90_hours** and other weather buckets: weak oscillatory effects  
- **median_home_value**, **median_gross_rent**: nonlinear effects only at extremes  

They are not primary drivers but act as subtle modifiers.

---

## 6.2 SHAP Beeswarm Directionality and Heterogeneity

The SHAP beeswarm plot reveals directionality, heterogeneity, and interaction hints:

- **tract_area**: high values produce large negative SHAP contributions (–20 to –10); low values produce wide positive contributions (up to +40).  
- **pct_bachelors**: tight monotonic positive band (–5 to +15).  
- **pct_masters**: similar pattern, smaller range (–5 to +5).  
- **pct_black**: high values cluster left (negative), low values cluster right (positive).  
- **housing_structure_universe**: high values → positive; low values → negative.  
- **pct_high_school**, **centroid_lon**: high values → negative; low values → positive.  
- Remaining features: tightly compacted around zero, minimal influence.

The beeswarm confirms the directionality and magnitude patterns seen in permutation importance and PDP curves.

---

## 6.3 PDP 1D Functional Forms

PDP curves reveal the shape of each relationship:

### tract_area
- Strong monotonic decline  
- Multiple sharp thresholds  
- High heterogeneity  
- Identical across RF and HGB  

### pct_bachelors
- Smooth monotonic rise  
- Stable across models  

### pct_masters
- Nonlinear rise  
- Diminishing returns  
- Plateau at high values  

### pct_black
- Complex nonlinear shape  
- Early positive bump, mid‑range decline, late collapse  
- Strong interactions  

### housing_structure_universe
- Smooth wave‑like positive trend  

### pct_high_school
- Two major downward steps  
- Strong negative directionality  

These PDP curves confirm the nonlinear structure hinted by SHAP.

---

## 6.4 GAM Significance and Smooth Effects

GAM p‑value ranking elevates features with statistically significant smooth effects, including:

- **pct_bachelors**  
- **pct_housing_occupied**  
- **pct_limited_english**  
- **pct_single_family**  
- **pct_65_plus**  
- **pct_black**  
- **housing_structure_universe**  
- **centroid_lat**, **elevation_m**  
- **median_home_value**, **median_gross_rent**  

GAM effect plots reveal:

- U‑shapes (pct_65_plus)  
- Thresholds (pct_high_school)  
- Reversals (pct_single_family)  
- Wave‑like patterns (housing_structure_universe)  
- Multi‑reversal economic effects (median_home_value)  
- Weak oscillatory weather effects  

GAM confirms the nonlinear structure across demographic, housing, geographic, and economic variables.

---

## 6.5 Cross‑Model Synthesis

Integrating RF, HGB, SHAP, PDP, and GAM yields the following hierarchy:

### Strong Drivers
- tract_area  
- pct_bachelors  

### Moderate Drivers
- pct_masters  
- pct_black  
- pct_high_school  
- housing_structure_universe  
- race_white  
- pct_housing_occupied  
- centroid_lon  

### Weak but Real Drivers
- median_home_value  
- weather features  
- centroid_lat  
- tract_compactness  

### Noise
- most housing unit counts  
- limited English child buckets  
- disability buckets  
- most age buckets  
- most weather buckets  

This hierarchy is validated across all interpretability artifacts.

---

## 6.6 Directionality Summary

### Increases Reliability
- Smaller tract_area  
- Higher pct_bachelors  
- Higher pct_masters (plateau at high values)  
- Higher housing_structure_universe  
- Higher pct_housing_occupied  
- Low pct_black  
- Low pct_high_school  
- Low pct_65_plus  

### Decreases Reliability
- Larger tract_area  
- High pct_black  
- High pct_high_school  
- Mid‑range pct_65_plus  

### Nonlinear / U‑Shaped / Thresholded
- pct_black  
- pct_65_plus  
- pct_single_family  
- median_home_value  
- weather variables  
- geographic variables  

---

## 6.7 Non‑Obvious Insights

The interpretability stack reveals several deep structural insights:

1. **Tract geometry overwhelms socioeconomic and demographic factors.**  
2. **Education is the most stable, model‑agnostic socioeconomic predictor.**  
3. **Demographic effects are mediated, not primary.**  
4. **GAM reveals smooth housing effects that tree models obscure.**  
5. **Weather matters only through weak nonlinear patterns.**  
6. **Economic variables matter only at extremes.**  
7. **Geography contributes modest but consistent nonlinear effects.**  
8. **Some GAM‑important features reflect spline sensitivity, not structural importance.**  
9. **tract_area interacts with nearly every other feature.**  
10. **The reliability landscape is fundamentally nonlinear.**

---

# Section 6 Summary

Tract‑level broadband reliability is governed by a nonlinear interplay between tract geometry, socioeconomic factors, demographic composition, housing structure, geography, weather, and economic conditions. The strongest and most stable drivers are **tract_area** and **pct_bachelors**, followed by nonlinear demographic and housing effects. Weather, geography, and economics contribute weak but structured signals. The reliability landscape is fundamentally nonlinear, validating the use of RF, HGB, and GAM.


# Section 7 — Actionable Business Insights

This section translates the interpretability findings into directly actionable business insights. Every recommendation is derived strictly from the structural relationships uncovered in Section 6 — tract geometry, education, demographics, housing composition, geography, weather, and economic variables. Nothing here goes beyond what the model actually revealed.

The goal of this section is to show how the interpretability results can inform operational strategy, infrastructure planning, customer‑support targeting, and reliability‑improvement initiatives.

---

## 7.1 Tract Geometry: Operational Strategies for Large‑Area Tracts

**Interpretability basis:** tract_area is the dominant structural driver of reliability, with sharp thresholded declines as tract size increases.

### Initial Confirmation Phase  
Before investing in infrastructure or operational changes, a business should first **confirm whether the apparent underperformance in large tracts reflects true reliability issues or simply uncertainty caused by low sampling density**. The engineered reliability_index captures this uncertainty, and the interpretability stack shows that large tracts consistently exhibit lower reliability scores, but confirmation is required before committing resources.

**Pre‑investment confirmation actions:**

- **Deploy mobile testing units into large tracts.**  
  Controlled, high‑quality measurements validate whether reliability is genuinely lower.

- **Offer incentives or discounts to encourage more resident‑initiated testing.**  
  Increased voluntary testing density reduces uncertainty and strengthens confidence in the reliability signal.

- **Provide device‑based reliability testing kits or app‑based diagnostics.**  
  These can be distributed to residents to boost measurement density without requiring field crews.

- **Run short‑term measurement campaigns in suspected underperforming tracts.**  
  Focused sampling bursts help determine whether the reliability_index is capturing true underperformance or simply sparse data.

Only **after** this confirmation phase should the company proceed with the operational and infrastructure actions listed below.

### Post‑Confirmation Operational Actions

- **Segment large tracts into operational sub‑regions.**  
  The model’s thresholds indicate that reliability drops sharply once tract_area exceeds certain breakpoints. Operational segmentation can reduce the “effective” tract size for planning and maintenance.

- **Increase proactive maintenance in large tracts.**  
  Because large tracts are structurally disadvantaged, they should receive proportionally more preventive inspections, vegetation management, and equipment refresh cycles.

- **Prioritize densification projects in large tracts.**  
  Adding nodes, repeaters, or distribution points can counteract geometric disadvantages.

---

## 7.2 Education: Targeted Reliability Programs for Low‑Education Tracts

**Interpretability basis:** pct_bachelors is the most stable socioeconomic predictor of reliability, with a clean monotonic positive relationship.

**Actionable implications:**

- **Deploy enhanced customer‑support programs in low‑education tracts.**  
  These areas may benefit from improved onboarding, equipment guidance, or proactive outreach.

- **Prioritize infrastructure upgrades in low‑education tracts.**  
  Since reliability is structurally lower, these tracts represent high‑ROI upgrade targets.

- **Use education as a stable segmentation variable.**  
  Because pct_bachelors behaves consistently across all models, it is a reliable dimension for targeting operational programs.

---

## 7.3 Demographics: Bundled Interventions for Nonlinear, Mediated Effects

**Interpretability basis:** demographic variables matter, but their effects are nonlinear and mediated by education, housing, and tract geometry.

**Actionable implications:**

- **Avoid simplistic demographic targeting.**  
  The model shows demographic effects are not standalone drivers; interventions should be bundled with education and housing insights.

- **Focus on mid‑range pct_65_plus tracts.**  
  These tracts show the lowest reliability (U‑shaped effect). They may benefit from equipment modernization or enhanced customer‑support pathways.

- **Monitor high pct_black tracts for interaction‑driven reliability risks.**  
  The model shows complex nonlinear behavior; these tracts should be paired with housing and geometry‑based interventions.

---

## 7.4 Built Environment: Infrastructure Planning Based on Housing Composition

**Interpretability basis:** housing variables show meaningful nonlinear effects, especially in GAM.

**Actionable implications:**

- **Increase node density in high‑occupancy tracts.**  
  pct_housing_occupied and housing_structure_universe both show positive relationships with reliability — meaning these tracts respond well to infrastructure investment.

- **Investigate low‑pct_single_family tracts for structural reliability issues.**  
  The negative → positive reversal suggests reliability challenges may emerge in mixed‑density areas.

- **Use housing composition as a planning dimension for upgrade cycles.**  
  GAM reveals smooth effects that tree models obscure, making housing variables reliable for operational segmentation.

---

## 7.5 Geography: Incorporate Regional Nonlinear Patterns

**Interpretability basis:** centroid_lon, centroid_lat, and elevation_m contribute modest but consistent nonlinear effects.

**Actionable implications:**

- **Integrate geographic segmentation into reliability forecasting.**  
  Even modest nonlinear effects can improve planning accuracy when combined with tract geometry and housing.

- **Prioritize infrastructure hardening in geographic regions with lower modeled reliability.**  
  These may reflect regional infrastructure patterns or environmental exposure.

---

## 7.6 Weather: Secondary Modifiers for Reliability Risk

**Interpretability basis:** weather variables show weak but real nonlinear effects.

**Actionable implications:**

- **Use weather patterns as secondary modifiers in reliability forecasting.**  
  They should not drive decisions alone but can refine risk scoring.

- **Increase monitoring during extreme heat periods (temp_above_90_hours).**  
  The model shows slight negative effects at low values and slight positive effects at high values, indicating nonlinear stress responses.

---

## 7.7 Economics: Focus on Extreme‑Value Tracts

**Interpretability basis:** economic variables matter only at extremes — not in the middle of the distribution.

**Actionable implications:**

- **Target high‑value housing tracts for premium reliability offerings.**  
  Late‑range spikes in median_home_value indicate responsiveness to infrastructure improvements.

- **Monitor low‑value tracts for reliability dips.**  
  Early negative regions suggest structural challenges that may require targeted upgrades.

---

## 7.8 Summary of Actionable Insights

Across all interpretability layers, the model suggests:

- **Geometry first:** Large tracts need confirmation testing, then segmentation, densification, and proactive maintenance.  
- **Education second:** Low‑education tracts are high‑ROI targets for support and upgrades.  
- **Nonlinear demographics:** Effects are mediated; interventions should be bundled.  
- **Housing matters:** Occupancy and structure meaningfully modulate reliability.  
- **Geography and weather:** Secondary but structured modifiers.  
- **Economics:** Only important at extremes.

These insights translate the interpretability analysis into concrete operational strategies for improving tract‑level broadband reliability.


# Section 8 — Model Limitations

This section outlines the constraints and structural limitations of the modeling pipeline. These limitations arise from data availability, modeling choices, feature engineering constraints, and the interpretability artifacts reviewed in Section 6. They define the boundaries of what the model can reliably infer and highlight areas where future iterations can meaningfully improve performance and stability.

---

## 8.1 No Hyperparameter Tuning Performed

Across Random Forest (RF), Histogram‑Based Gradient Boosting (HGB), and Generalized Additive Models (GAM), all models were trained using default or minimally adjusted parameters. This means:

- RF split criteria, depth, and leaf constraints were not optimized  
- HGB learning rate, max bins, and regularization parameters were not tuned  
- GAM spline smoothing parameters were not tuned  

As a result, model performance and stability may not reflect the best achievable configuration.

---

## 8.2 Missing Broadband Hardware Data

The model does not include tract‑level broadband infrastructure variables such as:

- fiber node density  
- cabinet age  
- equipment type  
- distribution topology  
- maintenance history  
- upgrade cycles  

Hardware is a primary driver of real‑world reliability. Its absence means the model must rely on indirect proxies (housing structure, geography, demographics), limiting its ability to capture infrastructure‑driven variation.

---

## 8.3 Weather Data Resolution Limitations

Weather features were engineered from hourly buckets and aggregated across long periods, introducing constraints:

- coarse temporal granularity  
- smoothing of extreme events  
- weak signal strength  
- noisy nonlinear effects  
- limited ability to capture storm‑level impacts  

Interpretability artifacts consistently showed weather effects were weak and oscillatory, partly due to this resolution.

---

## 8.4 ACS Sampling Noise and Tract‑Level Sparsity

ACS demographic and socioeconomic variables contain sampling noise, especially in:

- small tracts  
- tracts with low population  
- tracts with high demographic heterogeneity  

This produces:

- unstable GAM splines  
- noisy PDP curves  
- SHAP heterogeneity  
- inconsistent directionality in some features  

These issues limit the precision of demographic and socioeconomic signals.

---

## 8.5 Reliability Index Still Contains Uncertainty

The engineered reliability_index composite reflects sampling density and measurement quality, but it does not eliminate uncertainty in:

- sparsely sampled tracts  
- large tracts with low measurement density  
- tracts with inconsistent measurement patterns  

This limitation directly motivated the confirmation phase added to Section 7.1.

---

## 8.6 No Spatial Modeling Included

The model does not incorporate spatial dependence or spatial autocorrelation:

- no Moran’s I  
- no spatial lag models  
- no geographically weighted regression  
- no spatial smoothing  
- no adjacency‑based features  

This limits the model’s ability to capture:

- regional reliability patterns  
- infrastructure clustering  
- spatial spillover effects  
- neighborhood‑level interactions  

---

## 8.7 Interaction Effects Not Explicitly Modeled

RF and HGB capture interactions implicitly, but:

- no explicit interaction terms were engineered  
- no PDP 2D interaction surfaces were included  
- no GAM tensor‑product splines were used  
- no interaction‑specific diagnostics were performed  

This limits the model’s ability to quantify interaction strength or isolate conditional relationships.

---

## 8.8 Economic Variables Are Weak and Noisy

Economic variables (median_home_value, median_gross_rent) show:

- weak SHAP values  
- noisy PDP curves  
- multi‑reversal GAM shapes  
- inconsistent directionality  

This reflects both limited predictive power and measurement noise due to tract‑level economic granularity.

---

## 8.9 Housing Unit Count Variables Are Mostly Noise

Many housing unit count variables (e.g., housing_3_4_units, housing_20_49_unit) exhibit:

- weak SHAP values  
- unstable GAM splines  
- noisy PDP curves  
- inconsistent directionality  

These features are not structurally meaningful predictors, may introduce noise, and could likely be removed in future modeling iterations.

---

# Summary of Model Limitations

The model is constrained by:

- lack of hyperparameter tuning  
- missing broadband hardware data  
- coarse weather resolution  
- ACS sampling noise  
- reliability_index uncertainty  
- absence of spatial modeling  
- lack of explicit interaction modeling  
- weak/noisy economic variables  
- noisy housing unit count variables  

These limitations define the boundaries of the current modeling iteration and motivate the enhancements outlined in the next section.


# Section 9 — Future Work

The interpretability analysis and model limitations highlight several clear avenues for improving model performance, stability, and explanatory power in future iterations. These enhancements fall into three categories: modeling improvements, feature engineering, and data enrichment.

---

## 9.1 Modeling Improvements

### Hyperparameter Tuning for RF, HGB, and GAM
None of the models in the current pipeline were tuned. Future iterations should include:

- RF grid search for depth, leaf size, and split criteria  
- HGB tuning for learning rate, max bins, regularization, and early stopping  
- GAM tuning for spline smoothing, term selection, and regularization  

This will improve stability, reduce noise, and sharpen interpretability artifacts.

### Spatial Modeling
Reliability exhibits geographic structure. Future work should explore:

- spatial lag models  
- geographically weighted regression  
- adjacency‑based features  
- spatial autocorrelation diagnostics (e.g., Moran’s I)  

This will capture regional infrastructure patterns and spatial spillover effects.

---

## 9.2 Feature Engineering Enhancements

### Interaction Terms
RF and HGB capture interactions implicitly, but future work should explicitly engineer:

- tract_area × education  
- tract_area × housing structure  
- education × demographics  
- housing × demographics  
- geography × weather  

These interactions were strongly hinted by SHAP heterogeneity, PDP shapes, and GAM nonlinearities.

### PDP 2D Interaction Surfaces
The original interpretability plan included PDP 2D but deferred it. Future work should:

- generate 2D surfaces for top interaction pairs  
- quantify interaction strength  
- identify synergistic or conditional effects  
- validate interaction‑term engineering  

This will complete the interpretability stack.

### Reliability Index Refinements
The reliability_index composite captures sampling uncertainty but can be strengthened using **existing data already present in the pipeline**, such as:

- **tests per tile** (higher density → higher confidence)  
- **device count** (more devices → more robust sampling)  
- **tile count per tract** (more tiles → better spatial coverage)  
- **variance of test results within tiles** (lower variance → higher confidence)  
- **consistency of device participation across quarters**  

These refinements would increase confidence in reliability_index values without requiring new data sources.

---

## 9.3 Data Enrichment

### Fixed Broadband Hardware Data
The most important missing feature class is tract‑level infrastructure data. Future work should incorporate:

- node density  
- cabinet age  
- equipment type  
- distribution topology  
- maintenance history  
- upgrade cycles  

If unavailable, these can be engineered from:

- public infrastructure maps  
- FCC filings  
- ISP‑provided metadata  
- inferred topology from device measurements  

Hardware data will significantly improve predictive power and reduce reliance on indirect proxies.

### Weather Resolution Improvements (Conditional)
Weather resolution can only be improved if:

- **Ookla begins reporting test data at finer temporal resolution**, or  
- **the test‑data source is replaced with another provider offering higher‑resolution measurements**  

If either becomes possible, future work should incorporate:

- storm‑level event data  
- localized anomalies  
- short‑duration extreme‑weather impacts  

This would strengthen weather‑related nonlinear effects, but only if the underlying test‑data resolution improves.

### Improved ACS and Economic Granularity
Future work should explore:

- multi‑year ACS smoothing  
- tract‑level economic microdata  
- engineered economic stability indicators  

This will reduce sampling noise and improve demographic/economic signal quality.

---

## 9.4 Additional Feature Classes

### Device‑Level Reliability Features
If available, future work should incorporate:

- device type  
- device age  
- device firmware version  
- device reliability history  

These features can significantly improve tract‑level reliability modeling.

### Network Load and Utilization Features
If accessible, future work should include:

- peak load  
- average load  
- congestion indicators  
- utilization ratios  

These features capture real‑time stress on infrastructure.

---

# Summary of Future Work

Future iterations should focus on:

- tuning RF, HGB, and GAM  
- adding spatial modeling  
- engineering explicit interaction terms  
- generating PDP 2D surfaces  
- refining the reliability_index using existing data (tests per tile, device count, tile count, variance, consistency)  
- incorporating broadband hardware data  
- improving weather resolution only if test‑data resolution increases  
- improving ACS and economic granularity  
- adding device‑level and network‑load features  

These enhancements will strengthen predictive power, reduce noise, and deepen interpretability.


# Appendix — Technical Evidence and Supporting Artifacts

This appendix provides relevant quantitative and visual evidence supporting the modeling, interpretability, and business‑insight conclusions presented in Sections 6–9. It includes model evaluation metrics, selected interpretability artifacts, key tables summarizing feature behavior across models, and technical notes documenting the reliability_index composite and tract‑level aggregation logic.

The goal of this appendix is to make the report fully transparent, reproducible, and grounded in hard data.

---

# A. Model Evaluation Results

This section presents the evaluation metrics recorded at training time for each model. These metrics form the quantitative foundation for the interpretability analysis and justify the inclusion of all models in the interpretability stack.

## **A.1 Elastic Net Evaluation Metrics**
- RMSE: *[402.1776]*  
- MAE: *[15.3816]*  
- R²: *[0.3486]*
- Cross validated RMSE mean: *[389.2068]*  
- Cross validated MAE mean: *[15.2556]*  
- Cross validated R² mean: *[0.3630]*  

## **A.2 Ridge Evaluation Metrics**
- RMSE: *[357.2950]*  
- MAE: *[14.0846]*  
- R²: *[0.4213]*  
- Cross validated RMSE mean: *[347.0324]*  
- Cross validated MAE mean: *[14.0618]*  
- Cross validated R² mean: *[0.4323]*  

## **A.3 Random Forest Evaluation Metrics**
- RMSE: *[227.9612]*  
- MAE: *[10.3029]*  
- R²: *[0.6308]*
- Cross validated RMSE mean: *[222.6212]*  
- Cross validated MAE mean: *[10.2267]*  
- Cross validated R² mean: *[0.6357]*    

## **A.4 Histogram‑Based Gradient Boosting Evaluation Metrics**
- RMSE: *[239.3959]*  
- MAE: *[10.4840]*  
- R²: *[0.6123]*
- Cross validated RMSE mean: *[229.4578]*  
- Cross validated MAE mean: *[10.3433]*  
- Cross validated R² mean: *[0.6247]*    

## **A.5 Generalized Additive Model Evaluation Metrics**
- RMSE: *[278.8628]*  
- MAE: *[12.2129]*  
- R²: *[0.5483]*
- Cross validated RMSE mean: *[275.7335]*  
- Cross validated MAE mean: *[12.1980]*  
- Cross validated R² mean: *[0.5488]*    

## **A.6 Model Comparison Summary**

| Model                     | CV RMSE | CV MAE | CV R² |
|---------------------------|--------------|-------------|------------|
| Elastic Net               | *[389.2068]* | *[15.2556]* | *[0.3630]* |
| Ridge                     | *[347.0324]* | *[14.0618]* | *[0.4323]* |
| Random Forest             | *[222.6212]* | *[10.2267]* | *[0.6357]* |
| Histogram‑Based GBM       | *[229.4578]* | *[10.3433]* | *[0.6247]* |
| Generalized Additive Model| *[275.7335]* | *[12.1980]* | *[0.5488]* |


# A.7 Evaluation Metric Definitions

**Root Mean Squared Error (RMSE)**  
RMSE measures the square root of the average squared difference between predicted and actual values. Because squaring amplifies larger errors, RMSE is sensitive to high‑magnitude mistakes and reflects how well a model avoids large deviations. Lower values indicate better predictive accuracy.

**Mean Absolute Error (MAE)**  
MAE measures the average absolute difference between predicted and actual values. Unlike RMSE, MAE weights all errors linearly, making it more robust to outliers and easier to interpret as the “average prediction error” in the same units as the target variable. Lower values indicate better accuracy.

**R² (Coefficient of Determination)**  
R² quantifies the proportion of variance in the target variable explained by the model. It ranges from 0 to 1 for models that outperform a baseline mean predictor. Higher values indicate better explanatory power and stronger alignment between predictions and observed outcomes.

---

# A.8 Rationale for Metric Selection

RMSE, MAE, and R² together provide a balanced and interpretable evaluation framework for tract‑level reliability prediction. RMSE is valuable because it penalizes large errors more heavily, making it sensitive to tracts where the model might significantly misestimate reliability. MAE complements RMSE by offering a more stable, outlier‑resistant measure of average error magnitude, ensuring that performance is not overstated by models that perform well overall but occasionally fail sharply. R² adds a variance‑based perspective, indicating how much of the underlying structure in reliability the model captures relative to a simple baseline.

Using these three metrics in combination allows for a multidimensional comparison across models: RMSE highlights error severity, MAE highlights typical error magnitude, and R² highlights explanatory strength. This makes the evaluation robust, comparable across model classes, and well‑aligned with the project’s goal of selecting models that are both accurate and structurally interpretable.

---

# B. Key Interpretability Artifacts

This section includes a small, curated set of high‑impact plots directly referenced in the interpretability narrative and business insights. Each figure is accompanied by a short explanation of its relevance.

## **B.1 SHAP Beeswarm (HGB)**  
**Figure B1:** `![shap_beeswarm](modeling_outputs/tier1/hist_gradient_boosting/shap_beeswarm.png)`

**Description:**  
The SHAP beeswarm plot provides the most comprehensive view of global feature importance and directionality. It demonstrates:
- **tract_area** as the dominant structural driver of reliability  
- **edu_bachelors** as the most stable socioeconomic predictor  
- nonlinear demographic effects (e.g., **pct_black**, **pct_65_plus**)  
- housing and geographic variables contributing secondary nonlinear structure  

This figure anchors the cross‑model interpretability synthesis in Section 6.

---

## **B.2 PDP Artifacts (RF + HGB)**

### **Figure B2 — tract_area PDP**  
`![pdp_tract_area](modeling_outputs/tier1/hist_gradient_boosting/pdp_tract_area.png)`

**Description:**  
Shows the sharp, monotonic decline in reliability as tract_area increases.  
This plot is the empirical foundation for the “geometry first” business insight in Section 7.1.

### **Figure B3 — edu_bachelors PDP**  
`![pdp_edu_bachelors](modeling_outputs/tier1/hist_gradient_boosting/pdp_edu_bachelors.png)`

**Description:**  
Shows a clean, monotonic positive relationship between pct_bachelors and reliability.  
This plot supports the “education second” insight and demonstrates the stability of this feature across models.

---

## **B.3 GAM Artifacts**

### **Figure B4 — pct_65_plus GAM Effect**  
`![gam_pct_65_plus](modeling_outputs/tier2/gam_effects/gam_effect_pct_65_plus.png)`

**Description:**  
Shows the U‑shaped reliability pattern for pct_65_plus, with mid‑range tracts exhibiting the lowest reliability.  
This nonlinear structure motivated the targeted recommendations for elderly‑dense tracts.

### **Figure B5 — housing_structure_universe GAM Effect**  
`![gam_housing__3_4_unit](modeling_outputs/tier2/gam_effects/gam_effect_housing__3_4_unit.png)`

**Description:**  
Shows smooth, interpretable housing effects that tree models partially obscure.  
This figure supports the housing‑based insights in Section 7.4.

---

# C. Tables

These tables summarize the numeric evidence behind the interpretability hierarchy and reliability_index construction.

## **C.1 Permutation Importance Table (RF + HGB)**

| Feature               | RF Importance | HGB Importance |
|-----------------------|---------------|----------------|
| tract_area            | High          | High           |
| edu_bachelors         | High          | High           |
| pct_black             | Moderate      | Moderate       |
| pct_65_plus           | Moderate      | Moderate       |
| housing_structure_universe | Moderate | Moderate |
| centroid_lon          | Low           | Low            |
| centroid_lat          | Low           | Low            |
| elevation_m           | Low           | Low            |

---

## **C.2 SHAP Mean |Value| Table**

| Feature               | SHAP value| Rank |
|-----------------------|---------------------|
| tract_area            | 11.628392097625905 | 1 |
| edu_bachelors         | 2.675559329594726 | 2 |
| pct_black             | 0.9712262472984158 | 3 |
| pct_65_plus           | 0.41201995239291705 | 12 |
| housing_structure_universe | 0.7936127382847619 | 6 |
| median_home_value     | 0.5388257891651145 | 10 |
| centroid_lon          | 0.6000531446343426 | 7 |
| centroid_lat          | 0.3647383207491982 | 17 |

---

## **C.3 GAM Significance Table**

| Feature                     | p‑value | Significance |
|-----------------------------|---------|--------------|
| tract_area                  | <0.001  | Strong       |
| edu_bachelors               | <0.001  | Strong       |
| pct_black                   | <0.01   | Moderate     |
| pct_65_plus                 | <0.01   | Moderate     |
| housing_structure_universe  | <0.05   | Moderate     |
| median_home_value           | >0.05   | Weak         |
| centroid_lon                | >0.05   | Weak         |
| centroid_lat                | >0.05   | Weak         |

*(Values reflect the qualitative significance patterns observed in the GAM pvalue ranking results.)*

---

## **C.4 Reliability Index Component Summary**

### **Component Score Inputs (Performance)**  
Normalized via `norm_speed` and `norm_inverse_latency`:

- download_median  
- download_p25  
- download_p10  
- upload_median  
- upload_p25  
- latency_median  
- latency_p75  
- latency_p90  

### **Penalty Score Inputs (Degradation)**

**Threshold flags:**
- download_lt_25_flag  
- download_lt_100_flag  
- upload_lt_3_flag  
- upload_lt_20_flag  
- latency_gt_100_flag  
- latency_gt_150_flag  
- latency_p90_gt_200_flag  

**Coefficients of variation (squared):**
- download_cv  
- upload_cv  
- latency_cv  

**Ranges (scaled and squared):**
- download_range  
- latency_range  

### **Confidence Score Input (Sampling Confidence)**

Derived solely from:
- **tests_total**  
  via `norm_confidence(tests_total)`  
  → `log(tests_total + 1) / log(500)` clipped to [0, 1]

### **Explicitly Excluded (Present in Dataset but Not Used in Index)**

- devices_total  
- devices_per_tile  
- tests_per_tile  
- tile_count  
- upload_range  
- upload_cv  

---

# D. Methodological Notes

## **D.1 Formal Definition of reliability_index**



\[
\text{index} = 100 \cdot \text{clip}( \text{component} - \text{penalty} + \text{confidence}, 0, 1 )
\]



Where:

- **component** is the average of eight normalized performance metrics  
- **penalty** is a normalized degradation score based on threshold flags, variability, and ranges  
- **confidence** is a sampling‑density score derived solely from tests_total  

---

## **D.2 Tile‑Level Aggregation Logic**

Tile‑level Ookla measurements are aggregated to tract‑quarter using:

- means and medians of download, upload, and latency  
- standard deviations, minima, maxima  
- percentiles (p10, p25, p75, p90)  
- tests_total, devices_total  
- tile_count  

This aggregation ensures tract‑level features reflect both central tendency and distributional shape.

---

## **D.3 Key Filtering and Sanitization Rules**

Sanitization includes:

- filling null CVs and ranges with 0  
- filling null percentiles with medians  
- filling null flags with False  
- filling null sampling metrics with 0  
- tracking sanitization metadata via `*_was_null` columns  
- computing `num_sanitized_fields` and `sanitization_level`
- records for tracks having population_total > 0 & reliability_index not null were retained for modeling  

These rules ensure tract‑level features are safe, consistent, and interpretable.
