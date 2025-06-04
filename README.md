# Project README

## Overview

This project aims to analyze the relationship between financial news headlines and S\&P 500 price movements using AWS services. We will extract sentence-level sentiment and quarterly topic features from headlines, align them with daily closing prices, and build regression models to predict market returns.

## Prerequisites

* An AWS account with appropriate permissions to create and access S3, IAM, Glue, Athena, SageMaker, and EMR resources.
* AWS CLI configured locally (`aws configure` with Access Key, Secret, and default region).
* Python 3.8+ environment for running Jupyter Notebooks (with libraries: boto3, pyathena, s3fs, pandas, transformers, torch, scikit-learn, matplotlib).
* (Optional) Docker installed for running SageMaker Processing containers.

## Data Sources

1. **S\&P 500 with Financial News Headlines (2008–2024)**

   * Contains `date`, `title`, and `cp` (closing price) for each headline.
   * Stored locally after downloading from Kaggle and uploaded to S3.
2. **Financial Sentiment Lexicon**

   * Contains `word_or_phrase` and `sentiment_score` for each financial term.
   * Stored locally after downloading from Kaggle and uploaded to S3.

## AWS Resources and Workflow

1. **S3 Buckets**

   * `textdataproject-raw-snpnews/`: Raw headlines CSV files.
   * `textdataproject-raw-lexicon/`: Raw lexicon CSV files.
   * `textdataproject-processed-features/`: Processed Parquet outputs (sentiment, topic, daily features).
   * `textdataproject-query-results/`: Athena query results.

2. **IAM Role**

   * `TextDataProjectRole`: IAM Role with trust policy allowing AWS Glue, Athena, SageMaker, and EMR to assume it. Attached policies: `AWSGlueServiceRole`, `AmazonS3FullAccess`, `AmazonAthenaFullAccess`, etc.

3. **Glue Catalog and Crawlers**

   * **news\_db**: Database for raw headlines.

     * `crawler_snpnews_raw`: Scans `textdataproject-raw-snpnews/`, creates table `raw_news_<suffix>`.
   * **lexicon\_db**: Database for lexicon.

     * `crawler_lexicon_raw`: Scans `textdataproject-raw-lexicon/`, creates table `raw_lexicon_<suffix>`.

4. **Glue ETL Job: Clean, Tokenize, and Lexicon Scoring**

   * **Job Name**: `glue_job_news_sentiment_daily`
   * **Input**: `news_db.raw_news_<suffix>` and `lexicon_db.raw_lexicon_<suffix>`.
   * **Process**:

     1. Read raw headlines (`title`, `date`, `cp`).
     2. Clean text (remove non-alphabetic chars, lowercasing).
     3. Tokenize and remove stop words.
     4. Explode tokens, join with lexicon on `word_or_phrase` to get `sentiment_score`.
     5. Aggregate per day: calculate `daily_total_score`, `daily_pos_count`, `daily_neg_count`, and `daily_cp`.
     6. Write Parquet to `s3://textdataproject-processed-features/daily_sentiment_with_cp/` partitioned by `date`.

5. **Athena External Tables**

   * **sentiment\_db.daily\_sentiment\_with\_cp**: Points to `s3://textdataproject-processed-features/daily_sentiment_with_cp/`, columns: `date STRING`, `daily_total_score DOUBLE`, `daily_pos_count BIGINT`, `daily_neg_count BIGINT`, `daily_cp DOUBLE`.
   * **daily\_sentiment\_finbert** (created later): Aggregated sentence-level sentiment.
   * **quarterly\_topic\_features**: Outputs from Spark LDA.

6. **Sentence-Level Sentiment Analysis (FinBERT)**

   * **Environment**: SageMaker Notebook or Processing Job.
   * **Process**:

     1. Read raw headlines from S3 or Athena.
     2. Load `yiyanghkust/finbert-tone` model using Hugging Face `transformers`.
     3. For each `title`, compute sentiment label and probability score.
     4. Write results to `s3://textdataproject-processed-features/sentiment_per_headline/` as CSV/Parquet with columns: `date`, `title`, `sentiment_label`, `sentiment_score`.

7. **Daily Sentiment Aggregation**

   * **Athena**: Aggregate `sentiment_per_headline` by `date`:

     * `avg_sentiment = AVG(sentiment_score)`
     * `std_sentiment = STDDEV(sentiment_score)`
     * `headline_count = COUNT(*)`
     * Join `daily_cp` from raw headlines.
   * Write Parquet to `s3://textdataproject-processed-features/daily_sentiment_finbert/`.

8. **Quarterly Topic Modeling (Spark LDA)**

   * **Data Preparation**:

     1. Use Athena or Glue to compile all `title_clean` per quarter into a single record (concatenate or maintain tokens).
     2. Write to `s3://textdataproject-processed-features/quarterly_docs/`.
   * **EMR Job (Spark)**:

     1. Read `quarterly_docs/` Parquet/CSV.
     2. Tokenize and vectorize using `CountVectorizer` and optionally `TF-IDF`.
     3. Fit `LDA(k=10)` to obtain topics.
     4. Extract `topicDistribution` per quarter, write to `s3://textdataproject-processed-features/quarterly_topic_features/`.

9. **Entity and Event Extraction (Optional)**

   * **SageMaker Processing or Glue Notebook**:

     1. Load raw `title_clean`.
     2. Use spaCy or Spark NLP to extract named entities (e.g., `Fed`, `Tesla`).
     3. Apply regex rules for event tags (e.g., `earnings beat`, `rate hike`).
     4. Aggregate daily or quarterly counts of each entity/event.
     5. Write to S3 path: `s3://textdataproject-processed-features/entity_event_counts/`.

10. **Merge Features and Final Dataset**

    * **Athena SQL**:

      1. Join `daily_sentiment_finbert` with `quarterly_topic_features` (mapping each `date` to its `year_quarter`).
      2. (Optional) Join entity/event counts.
      3. Create table `sentiment_db.daily_full_features` with columns:

         * `date STRING`, `avg_sentiment DOUBLE`, `std_sentiment DOUBLE`, `headline_count BIGINT`, `daily_cp DOUBLE`, `topic1_prob_avg DOUBLE`, …, `topic10_prob_avg DOUBLE`, `entity_Fed_count INT`, …, `event_rate_hike_count INT`, etc.

11. **Visualization (QuickSight)**

    * Connect to Athena database `sentiment_db`.
    * Create analyses:

      1. Line chart: `date` vs. `avg_sentiment` and `daily_cp`.
      2. Stacked area chart: quarterly topic probabilities over time.
      3. Scatter plot: `avg_sentiment` vs. daily returns.
      4. Bar chart: entity/event counts vs. returns.

12. **Modeling (SageMaker Notebook)**

    * Read `daily_full_features` via PyAthena.
    * Construct features:

      * Daily features: `avg_sentiment`, `std_sentiment`, `headline_count`, `daily_cp`.
      * Quarterly features: `topic*_prob_avg`, `entity_*_count`, `event_*_count`.
      * Target: `next_day_return = (cp_next / cp_current) - 1`.
    * Split train/test, train regression models (e.g., LinearRegression, XGBoost), evaluate R², MAE.

13. **Optional: Model Deployment**

    * Create SageMaker Training Job with the best model.
    * Deploy to SageMaker Endpoint.
    * Write Lambda function (triggered by EventBridge daily) to:

      1. Query new headlines for the day.
      2. Run FinBERT and topic aggregation for that quarter if updated.
      3. Call Endpoint with daily features to get `predicted_return`.
      4. Write predictions to DynamoDB or S3.

## Directory Structure

```
project-root/
├── notebooks/
│   ├── env_setup.ipynb
│   ├── s3_glue_verification.ipynb
│   ├── finbert_sentiment_analysis.ipynb
│   ├── daily_sentiment_aggregation.ipynb
│   ├── quarterly_lda_emr_script.py
│   ├── entity_extraction.ipynb (optional)
│   └── modeling_and_prediction.ipynb
├── scripts/
│   ├── sentiment_processor.py           # SageMaker Processing script
│   ├── quarterly_lda.py                  # EMR Spark LDA job script
│   └── entity_event_extractor.py         # spaCy entity/event extraction
├── data/
│   └── README.md (this file)
└── README.md                             # High-level project README
```

## How to Run

1. Clone this repository.
2. Configure AWS CLI with your credentials:

   ```bash
   aws configure  # enter Access Key, Secret Key, default region (e.g., eu-west-1)
   ```
3. Upload raw Kaggle CSVs to S3:

   ```bash
   aws s3 cp ./data/snp_news.csv s3://textdataproject-raw-snpnews/ --recursive
   aws s3 cp ./data/financial_sentiment_lexicon.csv s3://textdataproject-raw-lexicon/ --recursive
   ```
4. Create IAM Role `TextDataProjectRole` with trust policy for Glue, Athena, SageMaker, EMR. Attach required managed policies.
5. In AWS Glue Console, run `crawler_snpnews_raw` and `crawler_lexicon_raw` to populate Glue Catalog.
6. Execute Glue ETL Job `glue_job_news_sentiment_daily` to generate daily sentiment Parquet.
7. Run `finbert_sentiment_analysis.ipynb` or `sentiment_processor.py` to produce sentence-level sentiment scores.
8. Use Athena queries in `daily_sentiment_aggregation.ipynb` to create `daily_sentiment_finbert` table.
9. Prepare quarterly documents and submit `quarterly_lda.py` on EMR; output to S3 `quarterly_topic_features`.
10. (Optional) Run `entity_extraction.ipynb` to get entity/event counts.
11. Aggregate all features in Athena to create `daily_full_features`.
12. Visualize in QuickSight and train models in `modeling_and_prediction.ipynb`.
13. (Optional) Deploy final model to SageMaker Endpoint and set up Lambda/ EventBridge.

---

### Contacts

* Project Lead: Zitian

Feel free to reach out for any clarifications or issues.
