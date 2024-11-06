import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def load_and_display_data():
  
    df_section_total = pd.read_csv("section_totals.csv")  
    df_section_article = pd.read_csv("section_article_counts.csv") 

    
    df_merged = pd.merge(df_section_total, df_section_article, on="section")
    df_merged.columns = df_merged.columns.str.strip()  


    fig = go.Figure()

   
    fig.add_trace(go.Bar(
        x=df_merged['section'],
        y=df_merged['total_count'],
        name='Total Count',
        marker=dict(color='rgb(0, 123, 255)'),
    ))

   
    fig.add_trace(go.Scatter(
        x=df_merged['section'],
        y=df_merged['article_count'],
        name='Article Count',
        mode='lines+markers',
        marker=dict(color='rgb(255, 99, 132)'),
        line=dict(width=2)
    ))

    fig.update_layout(
        title="Total Count and Article Count per Section",
        xaxis_title="Section",
        yaxis_title="Count",
        barmode='group',
        xaxis_tickangle=-45,  
        template="plotly_white",
        showlegend=True
    )

    st.plotly_chart(fig)

 
    try:
        topics = pd.read_csv("topics.csv")  
        st.subheader("Topics from Topic Modeling")
        for _, topic in topics.iterrows():
            st.write(f"**Topic {topic['topic_num']}**: {topic['top_words']}")
    except Exception as e:
        st.error(f"Error loading topics: {e}")


st.title('RTBF Articles Topic Modeling and Visualization')


if st.button('Load Topic Modeling Data'):
    load_and_display_data()
