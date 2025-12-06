# Manifest Change Log

## 2025-09-28: Dataset Alignment & External Drive Configuration

### Dataset Scope Adjustment
**Discovery**: Outage data from Figshare only available through 2020 (no 2021-2022 files exist)
**Decision**: Align both datasets to 2014-2020 for consistent temporal coverage

### Final Configuration: 2014-2020 (7 years)
Both manifests now configured for **7 years of matched data**:

**daily_grids_manifest.json:**
```json
"years": ["2014","2015","2016","2017","2018","2019","2020"]
```

**outages_manifest.json:**
```json  
"years": ["2014","2015","2016","2017","2018","2019","2020"]
```

**Rationale:**
- Ensures temporal alignment between weather and outage datasets
- Maximizes available data (7 years vs original 1 year)
- Eliminates missing data issues for ML model training
- Covers major weather events: 2017 (Harvey, Irma, Maria), 2018 (Florence, Michael), 2020 (active hurricane season)

### External Drive Configuration

**External Drive Structure:**
```
/Volumes/Academia/AI-Studio-Project/
├── data/
│   └── raw/
│       ├── weather/
│       │   └── daily_grids/
│       │       ├── 2014/ → 2020/
│       └── outages/
│           ├── 2014/ → 2020/
```

**Team Members:** If you don't have an external drive called "Academia", you'll need to:
1. Update the paths in both fetch scripts
2. Change `/Volumes/Academia/AI-Studio-Project/` to your preferred storage location
3. Consider using a separate branch for your local modifications

**Estimated Storage Requirements:**
- Weather data: ~252MB (36MB × 7 years)  
- Outage data: ~5.3GB (based on file sizes from Figshare API)
- Total: ~5.6GB for complete 7-year dataset

---

## Previous Changes

## outages_manifest.json

**Date**: 2025-09-26  
**Change**: Restricted years from full range to 2014 only

**Previous configuration:**
```json
"years": ["2014","2015","2016","2017","2018","2019","2020","2021","2022"]
```

**New configuration:**
```json
"years": ["2014"]
```

**Reason**: 
- Weather data (daily_grids_manifest.json) only has 2014 downloaded
- Chose 2014 for weather data due to faster download times  
- Need dataset alignment for Milestone 1 development
- Can expand both datasets to more years once pipeline is working

**Impact**: 
- Reduces outage data from 9 years to 1 year
- Aligns temporal coverage between weather and outage datasets
- Enables proper data integration testing with matched timeframes

**Future**: Can expand both manifests to include 2015, 2016, etc. once initial pipeline is validated