# 6862-project: Estimating Critical Biogeochemistry of the Southern Ocean Using Machine Learning Techniques

## Project Overview
Autonomous Biogeochemical-Argo (BGC-Argo) floats offer an opportunity for year-round sampling of remote regions of the Southern Ocean, but are limited in the variables they can directly measure. In this project we use machine learning-based regression methods to predict values of silicate and phosphate from inputs of temperature, salinity, oxygen, nitrate, date, and location. We train our models using ship transect data with the ultimate goal of predicting phosphate and silicate BGC-Argo float data and comparing it to the output from Earth System Models. In our preliminary results we report success using an ordinary least squares regression, with higher accuracy predicting values of phosphate compared to silicate. Future work on the project will incorporate the use of Bayesian methods to improve upon the scores of our baseline estimator and the application of our estimators to BGC-Argo float data.

## Repository Overview
Our repository is structured in the following manner:
- main: contains all of our code for data analysis
  - data: contains directories of data (that is small enough to be uploaded to GitHub) and data processing code (reformatting data, quality control)
  - figures: contains figures produced by our analysis and code
## Description of code
### Main
- ***MILESTONE_CODE***
  - Consolidated analysis (cross-validation, linear regression) of GOSHIP data to predict phosphate and silicate values; uses different methods of encoding position (none, raw, radians) and month (none, thermometer code, sin-cosine pair)
- ***GOSHIPData_map***
  - Creates maps and timeseries of GOSHIP data over different depth ranges

### Data
- ***DP_GOSHIP***
  - Reformats GOSHIP bottle data into csv files for varaibles of interest; Code will open original GO-SHIP bottle data and save data that is in region of interest as a csv file that can easily be read-in and a txt file that saves data notes
  - Things to change for your local machine: Change GitHubDir to path to local copy of this GitHub repo, Change LatN, LatS, LonE, LonW to desired region of interest
  - Note: there are some issues with data types (I may correct this to also update data types to floats from strings)
- ***DP_GOSHIP_QC***
  - Uses GOSHIP quality control flags to crop data to those with only QC flags = 2,6
- ***DP_ArgoDataDownload***
  - Writes bash files and index files to download Argo float data from a specific region of interest from the fttp site
  - Things to change for your local machine: Change GitHubDir to path to local copy of GitHub repo, Change LocalArgoDacDir to path to where on your machine you want to download the Argp data ***DO NOT HAVE THIS BE IN THE GITHUB REPO BECAUSE THE FILES ARE TOO BIG***, Change LatN, LatS, LonE, LonW to desired region of interest
