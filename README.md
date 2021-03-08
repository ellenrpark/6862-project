# 6862-project

## Data Processing Code
- ***DP_GOSGIP***
  - Change GitHubDir to path to local copy of this GitHub repo
  - Change LatN, LatS, LonE, LonW to desired region of interest
  - Code will open original GO-SHIP bottle data and save data that is in region of interest as a csv file that can easily be read-in and a txt file that saves data notes
  - Note: there are some issues with data types (I may correct this to also update data types to floats from strings)
- ***DP_ArgoDataDownload***
  - Change GitHubDir to path to local copy of GitHub repo
  - Change LocalArgoDacDir to path to where on your machine you want to download the Argp data ***DO NOT HAVE THIS BE IN THE GITHUB REPO BECAUSE THE FILES ARE TOO BIG***
  - Change LatN, LatS, LonE, LonW to desired region of interest
