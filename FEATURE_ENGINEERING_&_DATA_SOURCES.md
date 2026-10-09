# Feature Engineering & Data Sources

## 1. Purpose

This document summarizes the data sources, aggregation logic, feature‑engineering pipelines, and sanitization rules used to construct the unified tract‑quarter feature matrix for the broadband reliability modeling project. Each dataset is treated with equal weight, and all transformations are described at a high level to provide a clear, technical reference. The reliability_index is computed separately and is not detailed here, though that information may be found in EXECUTIVE_REPORT.md .

---

## 2. Data Sources

### Ookla Fixed Broadband Service  
Tile‑level broadband performance measurements (download, upload, latency, tests, devices) collected across multiple quarters. These data provide the tract‑quarter reliability target and sampling confidence metrics.

### ACS Demographic & Socioeconomic Data  
Tract‑level demographic, socioeconomic, housing, education, language, race, poverty, unemployment, and vehicle‑availability variables. These data supply structural and contextual features for each tract.

### TIGER Geography  
Tract boundaries and geometries used to compute tract area, perimeter, and compactness. TIGER also provides the polygons used for spatially joining Ookla tiles to tracts.

### LCDv2 Elevation & Weather  
Hourly tract‑level environmental and weather conditions, including temperature, humidity, wind, precipitation, visibility, and pressure. These data are aggregated to tract‑quarter and provide detailed environmental context.

---

## 3. Aggregation Logic

### 3.1 Ookla → Tract‑Quarter  
- Filter tiles to Wisconsin bounding box.  
- Normalize tile_x/tile_y to centroid_lon/centroid_lat.  
- Spatially join tiles to TIGER tract polygons.  
- Convert units (kbps → Mbps).  
- Extract year and quarter from filenames.  
- Aggregate tile‑level metrics to tract‑quarter:  
  - Means, medians, std, min, max  
  - Percentiles (p10, p25, p75, p90)  
  - tests_total, devices_total, tile_count  

### 3.2 LCDv2 → Tract‑Quarter  
- Add year and quarter from hourly timestamps.  
- Compute hourly extreme‑condition booleans (temperature, wind gusts, precipitation, visibility, pressure).  
- Compute compound booleans (wet_windy, cold_windy, hot_humid).  
- Compute storm_event booleans.  
- Compute circular wind‑direction features (resultant length, circular variance).  
- Detect sustained high‑wind periods and severe weather clusters.  
- Aggregate hourly data to tract‑quarter:  
  - Means, medians, std, min, max, range, CV  
  - Percentiles (p10, p25, p75, p90)  
  - Segmentation bucket counts and proportions  
  - Boolean counts  
  - observed_hours, expected_hours, coverage_ratio  

### 3.3 ACS → Tract  
- ACS data are already tract‑level; no temporal aggregation.
- Compute ratios from raw count data.  
- Used directly after feature engineering.

### 3.4 TIGER → Tract  
- Geometry decoded from WKB to polygons.  
- Compute tract_area, tract_perimeter, tract_compactness.  
- No temporal aggregation.

---

## 4. Feature Engineering

### 4.1 LCDv2 Derived Features  
- Numeric summaries (mean, median, std, min, max, range, CV).  
- Percentiles for all numeric weather variables.  
- Segmentation buckets for temperature, humidity, wind speed, wind gusts, precipitation, pressure, visibility.  
- Extreme‑condition booleans (temp_below_freezing, temp_above_90, gust tiers, precipitation tiers, visibility_lt_1, low_pressure).  
- Compound booleans (wet_windy, cold_windy, hot_humid).  
- Storm event indicator.  
- Cluster features (sustained_high_wind_periods, severe_weather_clusters).  
- Circular wind‑direction features (resultant length, circular variance).  
- Coverage metrics (observed_hours, expected_hours, coverage_ratio).

### 4.2 Tract (ACS + TIGER) Derived Features  
- Geometry features: tract_area, tract_perimeter, tract_compactness.  
- ACS totals: population_65_plus, population_under_5, disability_total, edu_total, limited_english_total, no_english_total.  
- ACS ratios:  
  - Sex ratios (pct_male, pct_female, sex_ratio)  
  - Age ratios (pct_65_plus, pct_under_5)  
  - Disability ratio (pct_disability)  
  - Education ratios (pct_high_school, pct_bachelors, pct_grad_degree)  
  - Poverty ratio (pct_poverty)  
  - Unemployment ratio (pct_unemployment)  
  - Housing occupancy (pct_housing_occupied, pct_housing_vacant)  
  - Race ratios (pct_white, pct_black, pct_asian, pct_hispanic, pct_american_indian, pct_pacific_islander, pct_race_other)  
  - Language isolation (pct_limited_english, pct_no_english)  
  - Vehicle availability (pct_vehicle_none)  
  - Housing structure (pct_single_family, pct_small_multi_family, pct_large_multi_family, pct_mobile_home)  
- Sentinel negative replacements for median_age, median_household_income, median_home_value, median_gross_rent.  
- Drop static fields not used in modeling.

### 4.3 Ookla Derived Features  
- Ranges (download_range, upload_range, latency_range).  
- Coefficients of variation (download_cv, upload_cv, latency_cv).  
- Sampling density (tests_per_tile, devices_per_tile).  
- Threshold flags (download_lt_25_flag, download_lt_100_flag, upload_lt_3_flag, upload_lt_20_flag, latency_gt_100_flag, latency_gt_150_flag, latency_p90_gt_200_flag).

### 4.4 Unified Feature Set  
- LCDv2 joined with Ookla on tract‑year‑quarter.  
- Tract (ACS + TIGER) joined on tract.  
- Filter out tracts with population_total = 0.  
- Filter out rows with null reliability_index.  
- Result is the unified tract‑quarter feature matrix.

---

## 5. Sanitization

### 5.1 LCDv2 Sanitization  
- Null detection for all numeric summaries and percentiles.  
- lcdv2_num_sanitized_fields, lcdv2_is_sanitized, lcdv2_sanitization_level.

### 5.2 Tract Sanitization  
- Null detection for all numeric ACS and TIGER fields.  
- tract_num_sanitized_fields, tract_is_sanitized, tract_sanitization_level.

### 5.3 Ookla Sanitization  
- CVs → 0  
- Ranges → 0  
- Percentiles → median  
- Flags → False  
- Sampling density → 0  
- tests_total → 0  
- num_sanitized_fields, is_sanitized, sanitization_level.

### 5.4 Unified Sanitization  
- population_total > 0 filter.  
- reliability_index not null filter.

---

## 6. Reference

For full methodological detail, interpretability artifacts, and reliability_index construction, see the Executive Report and Appendix.
