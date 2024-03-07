Thesis Work Roadmap

1.  Select list of packages (python libraries)
    
    a. Run this query on BQ to bbtain monthly pip python library download data from Google
    ```
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
    ```
    b. Download result from above query as csv and move it to `data/queries/package_downloads.csv`

    c. Run `output_analysis.ipynb` to turn produce our selected list of python packages, `data/inputs/packages_filtered.csv`

    d. Run `pypi_json.sh` to download json files for pip packages into `data/pip_json`

    e. Run `link_packages.ipynbs` to obtain github information associated with each pip package, stored in `data/inputs/package_repos.csv`

4. I split up the queries for github data. Prior to 2018/09, pip download data either doesn't exist (prior to 2016) or cannot be compared to data post 2018/09

    a. Import `data/inputs/package_repos.csv` as `github_pip` in your project
    
    b. GitHub Data can be obtained using the following query. Note that `thesis-402503` is my own personal project id and you'll have to change `thesis-402503` to your own id
    
      ```
      SELECT *
        FROM `githubarchive.month.20*`
        WHERE (_TABLE_SUFFIX BETWEEN '1101' AND '2403') AND repo.name in 
          (SELECT repo FROM `thesis-402503.python_libraries.github_pip`)
      ```
      Then, I saved the data as a BQ table called `filtered_table`. Then, I partitioned it and saved it into a table called `partitioned_filtered`
      ```
      CREATE TABLE `thesis-402503.python_libraries.partitioned_filtered`
      PARTITION BY RANGE_BUCKET(export_id, GENERATE_ARRAY(0, 1000, 1))
      CLUSTER BY export_id
      AS (
        SELECT *, CAST(FLOOR(1000*RAND()) AS INT64) AS export_id
        FROM `thesis-402503.python_libraries.filtered_table_pre18`
      );
      ```

    c. Create a folder in a gcloud bucket (mine the folder `filterd_github_data_large` in the gcloud bucket `thesis-github`) and export the data 

    - **Data from 2018/09 to 2023/08**
      ```
      EXPORT DATA 
      OPTIONS (uri='gs://thesis-github/filtered_github_data_large/partitions*.json', format='JSON')AS 
      (SELECT * 
      FROM `thesis-402503.python_libraries.filtered_table`)
      ```
    - **Data from 2011/01 to 2018/08**
      ```
      EXPORT DATA 
      OPTIONS (uri='gs://thesis-github/github_data_pre_18/github_data_pre18*.json', format='json')
      AS (SELECT * FROM `thesis-402503.python_libraries.partition_filtered_table_pre18`)
      ```

    d. Install gsutil/gcloud (follow instructions [here](https://cloud.google.com/sdk/docs/install)), then run the below command to download the raw data to your local machine

    - **Data from 2018/09 to 2023/08**
      ```
      gsutil -m cp -r gs://thesis-github/filtered_github_data_large data/github_raw/
      ```
    - **Data from 2011/01 to 2018/08**
      ```
      gsutil -m cp -r gs://thesis-github/github_data_pre_18 data/github_raw/
      ```

3. Clean downloaded results

    a. Transforming raw JSONs into csv's. Both of the below scripts delete the raw json data and replace it with cleaned data in `data/github_clean/filtered_github_data_large` and `data/github_clean/filtered_github_data_pre_18`, respectively
      - `bash clean_github_daha.sh` (from home directory `undergrad_thesis`) cleans all the data

    b. obtain_commit_data_pr - gets commit data for PRs, commit data for pushes already obtained in part a

    c. collect_commit_data.sh and collect_commit_data_pr.sh collect commit data for pushes and pull requests

    e. match_committers_pr.py and match_committers_push.py link commit authors/committers to github id's
      
    


data/queries contains results from google bigquery
data/inputs contains data we use as inputs for google bigquery or other files
data/pip_json contains json data we downloaded in order to learn about pip packages
data/github_raw contains downlaoded data from github 
data/github_clean contains data cleaned from data/github_raw
data/github_commits contains commit data cleaned from data/clean
data/package_downloads contains pip package downloads with version info at the daily level
repos is a temporary storage location we use when cloning github repos to extract git commit information


d. aggregating_push_data.sh and aggregating_pr_data.sh and clean pull request and push data


undergrad_thesis/data/inputs/free_email_domains.txt is from https://gist.github.com/humphreybc/d17e9215530684d6817ebe197c94a76b