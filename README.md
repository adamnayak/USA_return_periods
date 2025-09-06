# USA_return_periods
This public repository contains all code for the project entitled: "Financial losses associated with US floods occur with frequent, low return period precipitation" soon to be published. Specifically, the repository contains all code and associated data for USA return period analysis for financial losses. We refer to the Hyperclusters repository for clustered claims data.

NOTE: This repository and associated code (including the README) is regularly updated and being refined to enhance user experience for easy implementation. For inquiries about access or code use, please reach out directly to an3232@columbia.edu to get the most up-to-date files. Thank you!

# Abstract
Flooding in the U.S. leads to billions of dollars in financial losses annually, with projected increases due to escalating extreme precipitation, population growth, and deteriorating flood infrastructure. While federal regulation mandates flood insurance purchase within 100-year floodplains, our analysis of millions of federal insurance claims reveals that most flood losses arise from frequent, low-intensity precipitation events relative to regional climatology, with average regional precipitation return periods of under five years. Similarly, precipitation linked to disaster aid and property buyouts has return periods averaging less than 20 years. Using unsupervised learning, we identify that space-time precipitation clusters associated with major storms dominate losses, emphasizing the need for flood risk assessments and mitigation strategies that account for recurrent spatiotemporal compound events. The findings underscore the importance of flood risk management that extends beyond solely fluvial impacts into pluvial flood preparedness and assessment.

## Publication Link
Forthcoming...

# Repository Contents

## Python Source Files
Our 'src' folder contains our helper functions for analysis and statistical and unsupervised machine learning implementations.

#### ST_Cluster.py
This contains all functions for our spatiotemporal clustering analysis and subsequent plotting.

#### Nonstationary_MK.py
This file contains our primary function for county-level nonstationary Mann-Kendall tests and Sen's slope derivations.

#### MSWEP_Preprocess.py
Preprocessing functions for MSWEP gridded data, county aggregation, GEV fitting, and return period and event derivation.

#### ERA5_Preprocess.py
Preprocessing functions for ERA5 gridded data, county aggregation, GEV fitting, and return period and event derivation.

#### PRISM_Preprocess.py
Preprocessing functions for PRISM gridded data monthly events.

## Implementation Notebooks

### Analysis

### Figures

#### F1_3, F2, F4-F7
Jupyter notebooks for figure curation from processed and analyzed data.

### Pre-processing

## Final Figures
Here we provide our final figures for publication.
