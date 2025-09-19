# USA_return_periods
This public repository contains all code for the project entitled: "Financial losses associated with US floods occur with frequent, low return period precipitation" soon to be published. Specifically, the repository contains all code for USA return period analysis for financial losses.

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

#### Execute_ST_Cluster_Claims.ipynb
This is our latest code execution of our spatiotemporal clustering analysis for claims. Note the code allows for generation of a range of clusters based on varying parameters. 

#### Execute_ST_Cluster_Disasters.ipynb
This is our latest code execution of our spatiotemporal clustering analysis for disaster declarations (associated with aid and buyouts). Note the code allows for generation of a range of clusters based on varying parameters. 

#### Execute_Nonstationary_MK.ipynb
Our Mann-Kendall test for return periods in R for county-level return periods. 

#### nonstationary_gev.Rmd
Our nonstationary GEV implementation in R for county-level precipitation.

#### Cluster_Sensitivity.ipynb
This script provides analysis plots of the sensitivity analysis output across parameters of interest for varying clustering parameterizations for our ST_DBSCAN implementation.

### Figures

#### {F1_3, F2, F4-F7}.ipynb
Jupyter notebooks for figure curation from processed and analyzed data.

### Pre-processing

#### MSWEP_Execute_Preprocess.ipynb
Execution of our county-level precipitation analysis using MSWEP data, including annual block maxima, financial loss merger, GEV fitting, and return period extraction.

#### ERA5_Execute_Preprocess.ipynb
Execution of our county-level precipitation analysis using ERA5 data, including annual block maxima, financial loss merger, GEV fitting, and return period extraction.

#### PRISM_Execute_Preprocess.ipynb
Execution of our county-level precipitation analysis using PRISM data, including annual block maxima, financial loss merger, GEV fitting, and return period extraction.

#### CPI_Adjust.ipynb
This notebook performs operations to adjust claims for inflation using CPI-U for conducting further cost analysis.

#### Claims_Preprocess.ipynb
This notebook contains the execution of our pre-processing FIMA claims filtering and merging for county-level analysis.

#### Disasters_Preprocess.ipynb
This notebook contains the execution of our pre-processing FEMA Presidential disaster declaration filtering and merging for county-level analysis.

#### Aid_Preprocess.ipynb
This notebook contains the execution of our pre-processing FEMA individual disaster aid disbursement merging for county-level analysis.

#### Buyouts_Preprocess.ipynb
This notebook contains the execution of our pre-processing FEMA HMGP property buyout merging for county-level analysis.


## Final Figures
Here we provide our final figures for publication.

## Data Export
Here we provide the most recent version of the clustered disaster declarations sensitivity analysis files, which we aim to update periodically. We refer to the Hyperclusters repository for the most recent version of clustered claims data, which can be merged with the redacted claims dataset provided by OpenFEMA.

# Contact Me!
If you have general questions about the code or data please feel free to reach out and I am always happy to try to do my best to help out. If you're interested in using similar method or working on a new project, I am always looking to collaborate and am happy to contribute more broadly! Email is always in flux - but try me at adam.nayak@columbia.edu, adam.nayak@alumni.stanford.edu, adamnayak1@gmail.com, or feel free to ping me on LinkedIn.
