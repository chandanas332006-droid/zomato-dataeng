import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from google import genai
from dotenv import load_dotenv

# Load .env
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

NEW_REVIEWS = 50
TOK_K = 5

CACHE_FILE = "review_embeddings.parquet"


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# CONNECT TO SNOWFLAKE AND READ REVIEWS
# ============================================================

def read_reviews_from_snowflake():

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT
            REVIEW_ID,
            CITY,
            RATING,
            COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """

    cursor = conn.cursor()

    cursor.execute(query)

    df = cursor.fetch_pandas_all()

    cursor.close()
    conn.close()

    # Convert column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    return df


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def embed(texts):

    all_embeddings = []

    # Gemini allows maximum 100 texts in one embedding request
    for i in range(0, len(texts), 100):

        batch = texts[i:i + 100]

        print(
            f"Embedding reviews {i + 1} to "
            f"{i + len(batch)}..."
        )

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch
        )

        batch_embeddings = [
            embedding.values
            for embedding in response.embeddings
        ]

        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# ============================================================
# LOAD REVIEWS
# ============================================================

@st.cache_data()
def load_reviews():

    # If embeddings already exist, use them
    if os.path.exists(CACHE_FILE):

        print("Loading reviews from cache...")

        return pd.read_parquet(CACHE_FILE)

    # Otherwise read from Snowflake
    print("Reading reviews from Snowflake...")

    df = read_reviews_from_snowflake()

    print(f"Loaded {len(df)} reviews.")

    # Create embeddings
    print("Creating embeddings...")

    df["embedding"] = embed(
        df["comment"].tolist()
    )

    # Save embeddings locally
    df.to_parquet(CACHE_FILE)

    print("Embeddings saved to cache.")

    return df


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Chat with your Zomato Reviews")

st.caption(
    f"Searching {NEW_REVIEWS} reviews, "
    f"answering with {CHAT_MODEL}"
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vec_a, vec_b):

    return (
        np.dot(vec_a, vec_b)
        /
        (
            np.linalg.norm(vec_a)
            *
            np.linalg.norm(vec_b)
        )
    )


# ============================================================
# FIND SIMILAR REVIEWS
# ============================================================

def find_similar_reviews(question, df):

    # Create embedding for user's question
    question_vector = embed([question])[0]

    scores = []

    # Compare question with every review
    for review_vector in df["embedding"]:

        score = cosine_similarity(
            question_vector,
            review_vector
        )

        scores.append(score)

    # Copy dataframe
    result = df.copy()

    # Add similarity score
    result["score"] = scores

    # Get top K reviews
    return result.nlargest(
        TOK_K,
        "score"
    )


# ============================================================
# ASK GEMINI
# ============================================================

def ask_llm(question, top_reviews):

    context = ""

    # Build context from top reviews
    for _, row in top_reviews.iterrows():

        context += (
            f"City: {row['city']}\n"
            f"Rating: {row['rating']} stars\n"
            f"Review: {row['comment']}\n\n"
        )

    system_prompt = """
You are a helpful assistant analyzing Zomato customer reviews.

Answer ONLY using the customer reviews provided.

Be concise and clear.

If the provided reviews do not contain enough information
to answer the question, say:
"I don't have enough information in the provided reviews."
"""

    user_prompt = f"""
Question:
{question}

Customer Reviews:
{context}
"""

    response = client.models.generate_content(

        model=CHAT_MODEL,

        contents=user_prompt,

        config={
            "system_instruction": system_prompt,
            "temperature": 0.2
        }
    )

    return response.text


# ============================================================
# LOAD REVIEW DATA
# ============================================================

review_df = load_reviews()


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder=(
        "e.g. What are the most common complaints "
        "about delivery?"
    )
)


# ============================================================
# ANSWER QUESTION
# ============================================================

if question:

    # Find the most relevant reviews
    top_reviews = find_similar_reviews(
        question,
        review_df
    )

    # Ask Gemini using those reviews
    answer = ask_llm(
        question,
        top_reviews
    )

    # Display answer
    st.markdown("### Answer")

    st.write(answer)

    # Display reviews used
    with st.expander(
        "Reviews used to build this answer"
    ):

        st.dataframe(
            top_reviews[
                [
                    "city",
                    "rating",
                    "comment"
                ]
            ],
            hide_index=True
        )