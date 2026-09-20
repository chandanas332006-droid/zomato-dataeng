import os
import pandas as pd
import streamlit as st
import snowflake.connector
from google import genai
import json
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-2.5-flash"

FORBIDDEN_WORDS = [
    "drop",
    "delete",
    "truncate",
    "alter",
    "update",
    "insert",
    "create",
    "replace",
    "grant",
    "revoke"
]

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method"
]


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# DATABASE SCHEMA
# ============================================================

SCHEMA = """
Tables available in Snowflake.

FCT_ORDERS(
    order_id,
    order_date,
    customer_id,
    restaurant_id,
    city,
    cuisine,
    payment_method,
    order_status,
    is_delivered,
    sales_amount,
    discount,
    delivery_fee,
    gst,
    customer_rating,
    delivery_time_min
)

FACT_ORDER_ITEMS(
    order_id,
    food_id,
    quantity,
    price
)

RESTAURANTS(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    rating,
    cost_for_two
)

USERS(
    user_id,
    user_name,
    age,
    gender,
    city
)

REVIEWS(
    review_id,
    restaurant_id,
    user_id,
    rating,
    comment
)

MENU(
    food_id,
    restaurant_id,
    food_name,
    price,
    category
)

ORDERS(
    order_id,
    customer_id,
    restaurant_id,
    order_date,
    order_status
)

Important rules about the tables:

- Use FCT_ORDERS for order-related questions.
- Use RESTAURANTS for restaurant-related questions.
- Use REVIEWS for review-related questions.
- Use MENU for food/menu questions.
- Use USERS for customer questions.
- Use FACT_ORDER_ITEMS for order item questions.

GMV means delivered revenue.

For GMV:
SUM(sales_amount) should be used with
is_delivered = TRUE.

Do NOT use MARTS tables because the MARTS schema
currently has no tables.

Use only the tables listed above.
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert.

Your job is to write ONE SELECT query that answers
the user's question.

Rules:

1. SELECT queries only.
2. Never modify data.
3. You may use WITH queries (CTEs).
4. Use bare table names only.
5. Do NOT use database or schema prefixes.
6. Use only tables listed in the database schema.
7. Add a LIMIT of 100 or less unless the question asks
   for a single total.
8. For GMV, use sales_amount from FCT_ORDERS and
   consider only delivered orders.
9. Return ONLY JSON in this exact format:

{{
    "sql": "your query here"
}}

Database schema:

{SCHEMA}
"""


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

@st.cache_resource
def get_connection():

    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),

        # IMPORTANT:
        # Your tables are not in MARTS.
        # We will use PUBLIC as the default schema.
        schema="STAGING",

        role="DBT_ROLE"
    )


# ============================================================
# GENERATE SQL USING GEMINI
# ============================================================

def generate_sql(question):

    response = client.models.generate_content(
        model=MODEL,

        contents=question,

        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0
        }
    )

    answer = response.text

    # Remove markdown code fences if Gemini adds them
    answer = answer.replace("```json", "")
    answer = answer.replace("```", "")
    answer = answer.strip()

    # Convert JSON into Python dictionary
    data = json.loads(answer)

    sql = data["sql"]

    # Remove database/schema prefixes if Gemini adds them
    sql = sql.replace(
        "ZOMATO.PUBLIC.",
        ""
    )

    sql = sql.replace(
        "ZOMATO.MARTS.",
        ""
    )

    sql = sql.replace(
        "ZOMATO.",
        ""
    )

    return sql.strip().rstrip(";")


# ============================================================
# CHECK SQL SAFETY
# ============================================================

def is_safe(sql):

    lowered = sql.lower().strip()

    # Must start with SELECT or WITH
    if (
        not lowered.startswith("select")
        and not lowered.startswith("with")
    ):
        return False

    # Check forbidden words
    for word in FORBIDDEN_WORDS:

        if word in lowered:
            return False

    return True


# ============================================================
# RUN QUERY IN SNOWFLAKE
# ============================================================

def run_query(sql):

    conn = get_connection()

    cursor = conn.cursor()

    try:

        result = cursor.execute(
            sql
        ).fetch_pandas_all()

        return result

    finally:

        cursor.close()


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Chat with your Zomato Data")

st.caption(
    f"Ask in English, {MODEL} writes the SQL, "
    "Snowflake runs it"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Example Questions")

    for q in EXAMPLE_QUESTIONS:

        st.markdown(
            f"- {q}"
        )


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "Enter your question here",

    placeholder=(
        "e.g. Top 10 restaurants by revenue "
        "in Bangalore"
    )
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    try:

        # Generate SQL
        sql = generate_sql(question)

        # Show generated SQL
        st.subheader("Generated SQL")

        st.code(
            sql,
            language="sql"
        )

        # Check safety
        if not is_safe(sql):

            st.error(
                "The generated SQL is not safe to run. "
                "Please modify your question."
            )

        else:

            # Run SQL
            df = run_query(sql)

            # Show result count
            st.success(
                f"{len(df)} rows returned"
            )

            # Show data
            st.dataframe(
                df,
                hide_index=True
            )

            # Create chart when there are exactly 2 columns
            # and second column is numeric
            if (
                len(df.columns) == 2
                and pd.api.types.is_numeric_dtype(
                    df.iloc[:, 1]
                )
            ):

                st.subheader("Chart")

                st.bar_chart(
                    df,
                    x=df.columns[0],
                    y=df.columns[1]
                )

    except json.JSONDecodeError:

        st.error(
            "Gemini did not return valid JSON. "
            "Please try the question again."
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )