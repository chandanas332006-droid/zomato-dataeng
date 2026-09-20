# Zomato Data Engineering Project

A data engineering pipeline for processing Zomato restaurant, order, menu, and customer review data using Airflow, dbt, Snowflake, and Gemini-based review enrichment.

## Technologies

- Python
- Apache Airflow
- dbt
- Snowflake
- Google Gemini API
- PowerShell

## Project Structure

```text
zomato/
├── ai/                 # Review enrichment scripts
├── airflow/            # Airflow DAGs and configuration
├── dbt/                # dbt models and tests
├── data/               # Local data files
├── .gitignore
└── README.md
```

## Pipeline

1. Load source data into Snowflake.
2. Transform and model data using dbt.
3. Orchestrate workflows with Airflow.
4. Enrich customer reviews with sentiment, topic, and issue classification.
5. Store analytical results in Snowflake.

## Setup

```powershell
git clone https://github.com/chandanas332006-droid/zomato-dataeng.git
cd zomato-dataeng

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file. Never commit credentials.

```env
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_DATABASE=ZOMATO
SNOWFLAKE_SCHEMA=STAGING
SNOWFLAKE_WAREHOUSE=ZOMATO_WH
GEMINI_API_KEY=your_api_key
```

## Running dbt

```powershell
dbt debug
dbt run
dbt test
```

## Running the Review Enrichment Script

```powershell
python .\ai\enrich_reviews.py
```

## Security

Do not commit:

- `.env`
- Passwords or API keys
- `profiles.yml`
- `.venv/`
- Large CSV files
- Airflow logs

## License

This project is for educational and portfolio purposes.
