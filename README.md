Thesis Work Roadmap

data/queries contains results from google bigquery
data/inputs contains data we use as inputs for google bigquery or other files
data/pip_json contains json data we downloaded in order to learn about pip packages
data/github_raw contains downlaoded data from github 
data/github_clean contains data cleaned from data/github_raw
data/github_commits contains commit data cleaned from data/clean
repos is a temporary storage location we use when cloning github repos to extract git commit information



1. Obtain download data on pip projects (run query below)
SELECT
  file.project as `project`,
  COUNT(*) AS num_downloads,
  DATE_TRUNC(DATE(timestamp), MONTH) AS `month`
FROM `bigquery-public-data.pypi.file_downloads`
WHERE DATE(timestamp)
  BETWEEN DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 180 MONTH), MONTH)
    AND CURRENT_DATE()
GROUP BY `month`, `project`
ORDER BY `month` DESC
- store in ‘data/queries/package_downloads.csv’

2. Turn ‘data/queries/package_downloads.csv’ into list of python packages (occurs in output_analysis.ipynb)
    1. data/inputs/packages.csv
    2. data/inputs/packages_filtered.csv

3. Run pypi_json.sh in order to download the json’s

4. link_packages.ipynb to get github information about each package

5. run this query and save the table
SELECT *
  FROM `githubarchive.month.20*`
  WHERE (_TABLE_SUFFIX BETWEEN '1809' AND '2308') AND repo.name in 
    (SELECT repo FROM `thesis-402503.python_libraries.github_pip`)

6. export it to google cloud
EXPORT DATA 
OPTIONS (uri='gs://thesis-github/filtered_github_data_large/partitions*.json', format='JSON')
AS 
(SELECT * 
FROM `thesis-402503.python_libraries.filtered_table`)

7. install gsutil, gcloud, then run this command to download to local computer
gsutil -m cp -r \
  "gs://thesis-github/filtered_github_data_large" data/github_raw/

8. run bash clean_github_daha.sh
9. 
