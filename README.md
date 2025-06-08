# Project README

## Overview

This project analyzes the relationship between financial news headlines and S\&P 500 price movements using AWS services. We extract both lexicon-based and sentence-level sentiment features from headlines, aggregate them at the daily level, combine them with closing prices, and build regression models to predict market returns.

## Prerequisites

* AWS account with permissions for S3, IAM, Glue, Athena, SageMaker, and EMR.
* AWS CLI configured locally (`aws configure` with Access Key, Secret, and default region).
* Python 3.8+ environment for Jupyter Notebooks (requires: `boto3`, `pyathena`, `s3fs`, `pandas`, `transformers`, `torch`, `scikit-learn`, `matplotlib`).
* (Optional) Docker for SageMaker Processing containers.

## Data Sources

1. **S\&P 500 with Financial News Headlines (2008–2024)**

   * Contains `date`, `title`, and `cp` (closing price) per headline.
   * Download from Kaggle and upload to S3.
2. **Financial Sentiment Lexicon**

   * Contains `word_or_phrase` and `sentiment_score`.
   * Download from Kaggle and upload to S3.

## AWS Resources and Workflow

### 1. S3 Buckets

* `textdataproject-raw-snpnews-zitian/`: Raw headlines CSV files.
* `textdataproject-raw-lexicon-zitian/`: Raw lexicon CSV files.
* `textdataproject-processed-features-zitian/`: Processed Parquet outputs (daily lexicon features, sentence-level sentiment, merged features).
* `textdataproject-query-results-zitian/`: Athena query results.

### 2. IAM Role

* **Role Name**: `TextDataProjectRole`

  * Trust policy allows: `glue.amazonaws.com`, `sagemaker.amazonaws.com`, `athena.amazonaws.com`, (and optionally `elasticmapreduce.amazonaws.com`).
  * Attached policies: `AmazonS3FullAccess`, `AmazonAthenaFullAccess`, `AmazonSageMakerFullAccess`, `AWSGlueServiceRole`, `AmazonEMRFullAccess`.

### 3. Glue Catalog and Crawlers

1. **news\_db** (raw headlines)

   * Crawler: `crawler_snpnews_raw_2025` scans `s3://textdataproject-raw-snpnews-zitian/`, creates table `sp500_headlines_2008_2024_csv` in database `news_db`.
2. **lexicon\_db** (raw lexicon)

   * Crawler: `crawler_lexicon_raw_2025` scans `s3://textdataproject-raw-lexicon-zitian/`, creates table `financial_sentiment_lexicon_csv` in database `lexicon_db`.

### 4. Glue ETL Job: Lexicon-Based Daily Sentiment

* **Job Name**: `glue_job_news_sentiment_daily`
* **IAM Role**: `TextDataProjectRole`
* **Script Summary**:

  1. Read `news_db.sp500_headlines_2008_2024_csv` (`title`, `date`, `cp`).
  2. Clean text (remove non-alphabetic characters, lowercase). Create `title_clean`.
  3. Tokenize into words, remove English stop words.
  4. Explode tokens, join with `lexicon_db.financial_sentiment_lexicon_csv` on `word_or_phrase` = token.
  5. Assign zero score to tokens not in lexicon; compute `score` column.
  6. Aggregate by `date`:

     * `daily_total_score = SUM(score)`
     * `daily_pos_count = COUNT(score > 0)`
     * `daily_neg_count = COUNT(score < 0)`
     * `daily_cp = first(cp)`
  7. Write Parquet to `s3://textdataproject-processed-features-zitian/daily_sentiment_with_cp/`, partitioned by `date`.

### 5. Athena External Tables (Lexicon Features)

1. **Database**: `sentiment_db` (create if not exists).

   ```sql
   CREATE DATABASE IF NOT EXISTS sentiment_db;
   USE sentiment_db;
   ```
2. **daily\_sentiment\_with\_cp**

   ```sql
   CREATE EXTERNAL TABLE IF NOT EXISTS sentiment_db.daily_sentiment_with_cp (
     daily_total_score DOUBLE,
     daily_pos_count BIGINT,
     daily_neg_count BIGINT,
     daily_cp DOUBLE
   )
   PARTITIONED BY (date STRING)
   STORED AS PARQUET
   LOCATION 's3://textdataproject-processed-features-zitian/daily_sentiment_with_cp/';
   MSCK REPAIR TABLE sentiment_db.daily_sentiment_with_cp;
   ```

### 6. Sentence-Level Sentiment Analysis (FinBERT) on SageMaker

1. **Launch a SageMaker Notebook Instance**

   * Notebook name: `finbert-sentiment-notebook`
   * Instance type: `ml.t3.large` (or larger if full dataset).
   * IAM role: `TextDataProjectRole`.
   * (Optional) Lifecycle script installs: `!pip install pyathena boto3 pandas transformers torch s3fs`.

2. **Notebook Steps**

   1. **Install dependencies**

      ```bash
      !pip install --upgrade pip
      !pip install pyathena boto3 pandas transformers torch s3fs
      ```
   2. **Read `clean_headlines_for_finbert` table from Athena**

      ```python
      from pyathena import connect
      import pandas as pd

      conn = connect(
          s3_staging_dir='s3://textdataproject-query-results-zitian/',
          region_name='eu-north-1'
      )
      query = """
      SELECT date, title_clean
      FROM sentiment_db.clean_headlines_for_finbert
      """
      df_headlines = pd.read_sql(query, conn)
      print("Total headlines:", len(df_headlines))
      df_headlines.head(5)
      ```
   3. **Load FinBERT and predict**

      ```python
      from transformers import AutoTokenizer, AutoModelForSequenceClassification
      import torch

      tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
      model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
      labels = ["negative", "neutral", "positive"]

      def finbert_predict(texts, batch_size=16):
          all_labels = []
          all_scores = []
          for i in range(0, len(texts), batch_size):
              batch = texts[i : i + batch_size]
              enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
              with torch.no_grad():
                  outputs = model(**enc)
                  probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
              for p in probs:
                  idx = p.argmax()
                  all_labels.append(labels[idx])
                  all_scores.append(float(p[idx]))
          return all_labels, all_scores

      df_headlines["sentiment_label"], df_headlines["sentiment_score"] = finbert_predict(df_headlines["title_clean"].tolist())
      df_headlines.head(5)
      ```
   4. **Write sentence-level results to S3 (non-partitioned)**

      ```python
      import boto3

      bucket = "textdataproject-processed-features-zitian"
      prefix = "sentiment_per_headline"
      s3_path = f"s3://{bucket}/{prefix}/all_sentiment_per_headline.parquet"

      df_headlines.to_parquet(s3_path, index=False, engine="pyarrow")
      print("Uploaded single Parquet to:", s3_path)
      ```

3. **Athena: Register `sentiment_per_headline` as external table**

   ```sql
   USE sentiment_db;
   CREATE EXTERNAL TABLE IF NOT EXISTS sentiment_db.sentiment_per_headline (
     date STRING,
     title_clean STRING,
     sentiment_label STRING,
     sentiment_score DOUBLE
   )
   STORED AS PARQUET
   LOCATION 's3://textdataproject-processed-features-zitian/sentiment_per_headline/';
   ```

### 7. Daily Sentence-Level Aggregation (Non-Partitioned)

1. **CTAS to produce `daily_sentiment_finbert_temp` (non-partitioned)**

   ```sql
   USE sentiment_db;
   CREATE TABLE IF NOT EXISTS sentiment_db.daily_sentiment_finbert_temp
   WITH (
     format = 'PARQUET',
     external_location = 's3://textdataproject-processed-features-zitian/daily_sentiment_finbert_temp/'
   ) AS
   SELECT
     h.date                      AS date,
     AVG(h.sentiment_score)      AS avg_sentiment,
     STDDEV(h.sentiment_score)   AS std_sentiment,
     COUNT(*)                    AS headline_count,
     MAX(d.daily_cp)             AS daily_cp
   FROM sentiment_db.sentiment_per_headline h
   JOIN sentiment_db.daily_sentiment_with_cp d
     ON h.date = d.date
   GROUP BY h.date;
   ```

2. **Validate `daily_sentiment_finbert_temp`**

   ```sql
   SELECT COUNT(*) FROM sentiment_db.daily_sentiment_finbert_temp;
   SELECT * FROM sentiment_db.daily_sentiment_finbert_temp ORDER BY date LIMIT 5;
   ```

3. **Register `daily_sentiment_finbert` external table**

   ```sql
   USE sentiment_db;
   CREATE EXTERNAL TABLE IF NOT EXISTS sentiment_db.daily_sentiment_finbert (
     date STRING,
     avg_sentiment DOUBLE,
     std_sentiment DOUBLE,
     headline_count BIGINT,
     daily_cp DOUBLE
   )
   STORED AS PARQUET
   LOCATION 's3://textdataproject-processed-features-zitian/daily_sentiment_finbert_temp/';
   ```

4. **Validate `daily_sentiment_finbert`**

   ```sql
   SELECT date, avg_sentiment, std_sentiment, headline_count, daily_cp
   FROM sentiment_db.daily_sentiment_finbert
   ORDER BY date LIMIT 5;
   ```

### 8. Merge Lexicon and Sentence-Level Features into `daily_full_features`

1. **CTAS to produce `daily_full_features_temp` (non-partitioned)**

   ```sql
   USE sentiment_db;
   CREATE TABLE IF NOT EXISTS sentiment_db.daily_full_features_temp
   WITH (
     format = 'PARQUET',
     external_location = 's3://textdataproject-processed-features-zitian/daily_full_features_temp/'
   ) AS
   SELECT
     l.date                                  AS date,
     l.daily_total_score                     AS daily_total_score,
     l.daily_pos_count                       AS daily_pos_count,
     l.daily_neg_count                       AS daily_neg_count,
     l.daily_cp                              AS daily_cp_from_lexicon,
     f.avg_sentiment                         AS avg_sentiment,
     f.std_sentiment                         AS std_sentiment,
     f.headline_count                        AS headline_count,
     f.daily_cp                              AS daily_cp_from_finbert
   FROM sentiment_db.daily_sentiment_with_cp l
   LEFT JOIN sentiment_db.daily_sentiment_finbert f
     ON l.date = f.date;
   ```

2. **Validate `daily_full_features_temp`**

   ```sql
   SELECT COUNT(*) FROM sentiment_db.daily_full_features_temp;
   SELECT * FROM sentiment_db.daily_full_features_temp ORDER BY date LIMIT 5;
   ```

3. **Register `daily_full_features` external table**

   ```sql
   USE sentiment_db;
   CREATE EXTERNAL TABLE IF NOT EXISTS sentiment_db.daily_full_features (
     date STRING,
     daily_total_score DOUBLE,
     daily_pos_count BIGINT,
     daily_neg_count BIGINT,
     daily_cp_from_lexicon DOUBLE,
     avg_sentiment DOUBLE,
     std_sentiment DOUBLE,
     headline_count BIGINT,
     daily_cp_from_finbert DOUBLE
   )
   STORED AS PARQUET
   LOCATION 's3://textdataproject-processed-features-zitian/daily_full_features_temp/';
   ```

4. **Validate `daily_full_features`**

   ```sql
   SELECT date, daily_total_score, avg_sentiment, daily_cp_from_lexicon, daily_cp_from_finbert
   FROM sentiment_db.daily_full_features
   ORDER BY date LIMIT 5;
   ```

---

## Modeling (Regression)

Now that we have the merged daily features table, we can train regression models to predict next-day returns.

### 1. Example: Linear Regression in SageMaker Notebook

```python
import pandas as pd
from pyathena import connect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Connect to Athena
conn = connect(
    s3_staging_dir='s3://textdataproject-query-results-zitian/',
    region_name='eu-north-1'
)

# 2. Load daily_full_features
query = """
SELECT *
FROM sentiment_db.daily_full_features
ORDER BY date
"""
df = pd.read_sql(query, conn)

# 3. Compute next-day return
#    Shift daily_cp_from_lexicon to get next day's cp
#    next_return = (next_cp - cp_today) / cp_today

df['next_cp'] = df['daily_cp_from_lexicon'].shift(-1)
df['next_return'] = (df['next_cp'] - df['daily_cp_from_lexicon']) / df['daily_cp_from_lexicon']
# Drop last row with NaN
[df.dropna(inplace=True)]

# 4. Define features and target
features = [
    'daily_total_score', 'avg_sentiment', 'std_sentiment', 'headline_count'
]
X = df[features]
y = df['next_return']

# 5. Split train/test (no shuffle to preserve time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 6. Train linear regression
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)

# 7. Evaluate
mse = mean_squared_error(y_test, preds)
print("Linear Regression MSE:", mse)
```

### 2. (Optional) Advanced Models: XGBoost / LightGBM

1. **Install**:

   ```bash
   !pip install xgboost lightgbm
   ```
2. **Example**:

   ```python
   import xgboost as xgb

   dtrain = xgb.DMatrix(X_train, label=y_train)
   dtest = xgb.DMatrix(X_test, label=y_test)
   params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
   bst = xgb.train(params, dtrain, num_boost_round=50)

   preds_xgb = bst.predict(dtest)
   print("XGBoost MSE:", mean_squared_error(y_test, preds_xgb))
   ```

### 3. Save and Document Results

* Plot predicted vs. actual returns over time.
* Compare performance of lexicon-only vs. sentence-level vs. combined features.
* Save plots and metrics in S3 or Notebook outputs for review.

---


## How to Run

1. **Upload raw CSVs to S3**:

   ```bash
   aws s3 cp data/snp_headlines_2008_2024.csv s3://textdataproject-raw-snpnews-zitian/ --recursive
   aws s3 cp data/financial_sentiment_lexicon.csv s3://textdataproject-raw-lexicon-zitian/ --recursive
   ```
2. **Run Glue Crawlers**:

   * `crawler_snpnews_raw_2025` → populates `news_db.sp500_headlines_2008_2024_csv`
   * `crawler_lexicon_raw_2025` → populates `lexicon_db.financial_sentiment_lexicon_csv`
3. **Execute Glue ETL Job**:

   * `glue_job_news_sentiment_daily` → outputs Parquet to `s3://textdataproject-processed-features-zitian/daily_sentiment_with_cp/`
4. **Run SageMaker Notebook**:

   * `finbert_sentiment_analysis.ipynb` → reads from Athena, runs FinBERT, writes to `s3://textdataproject-processed-features-zitian/sentiment_per_headline/`
5. **Athena Aggregation**:

   * Create and validate `daily_sentiment_finbert_temp` and `daily_sentiment_finbert` tables.
   * Create and validate `daily_full_features_temp` and `daily_full_features` tables.
6. **Modeling**:

   * Run `modeling_and_prediction.ipynb` (Linear Regression, XGBoost, etc.) to predict `next_return` and evaluate.

---

### Contacts

* Project Lead: Zitian

For questions or issues, reach out on Slack or email. Thank you!
