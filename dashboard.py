import pandas as pd
import streamlit as st

# Load the topics from the CSV file
def load_topics():
    try:
        return pd.read_csv('topics.csv')  # Adjust the file name if needed
    except FileNotFoundError:
        st.error("topics.csv file not found.")
        return pd.DataFrame(columns=["topic_num", "top_words"])

# Load topics
topics_df = load_topics()

# Check if DataFrame is empty
if topics_df.empty:
    st.warning("No valid topics found, displaying default message.")
    topics_df = pd.DataFrame({"topic_num": [1], "top_words": ["No topics available"]})

# Streamlit app layout
st.title("Topic Modeling Dashboard")

st.write("Below are the identified topics from your analysis:")

# Display the topics in a table
st.table(topics_df)

# Optionally, if you want to visualize the topics as a list
st.subheader("List of Topics")
for index, row in topics_df.iterrows():
    st.write(f"**Topic {row['topic_num']}:** {row['top_words']}")
