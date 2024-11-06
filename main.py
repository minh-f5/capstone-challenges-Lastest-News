import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def scrape_rtbf_articles():
    sections = ['elections', 'info', 'sport', 'regions', 'culture', 'environnement', 'bien-etre', 'tech', 'vie-pratique']
    base_url = "https://www.rtbf.be/en-continu/{}?page={}"

    articles = []

    for section in sections:
        for page_number in range(1, 11):  # Pages 1 to 10
            url = base_url.format(section, page_number)
            response = requests.get(url)

            if response.status_code != 200:
                print(f"Failed to retrieve page {page_number} in section {section}: {response.status_code}")
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            article_elements = soup.find_all("article")

            if not article_elements:
                print(f"No more articles found in section {section} on page {page_number}.")
                break

            for article in article_elements:
                title_element = article.find("h3")
                link_element = article.find("a")

                title = title_element.get_text(strip=True) if title_element else "No title found"
                link = link_element["href"] if link_element else "No link found"

                content = None
                if link:
                    full_url = f'https://www.rtbf.be{link}'
                    article_response = requests.get(full_url)
                    if article_response.status_code == 200:
                        article_soup = BeautifulSoup(article_response.content, 'html.parser')

                        content_div = article_soup.find('div', class_='article__body')  
                        if not content_div:
                            content_div = article_soup.find('article')  
                        if not content_div:
                            content_div = article_soup.find('div', class_='content')  

                        content = content_div.get_text(strip=True) if content_div else "No content found"

                articles.append({
                    "title": title,
                    "link": link,
                    "section": section,  # Add the section to each article
                    "content": content  
                })

            print(f"Section {section}, Page {page_number}: {len(article_elements)} articles found.")

            if len(articles) >= 2000:
                print(f"Total articles collected: {len(articles)}")
                break

    df = pd.DataFrame(articles)
    df = df.dropna(subset=['content'])  

    if df.empty:
        print("No articles with valid content were collected.")
    else:
        print(f"Collected {len(df)} articles with content.")

    df.to_csv("articles_rtbf.csv", index=False)
    print("Articles saved to articles_rtbf.csv")
    return df


def perform_topic_modeling(df, n_topics=1000):
    french_stop_words = [
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'en', 'à', 
        'et', 'est', 'dans', 'pour', 'sur', 'avec', 'par', 'que', 'qui', 
        'ce', 'cette', 'ces', 'son', 'sa', 'ses', 'se', 'sont', 'au', 
        'a', 
    ]
    vectorizer = CountVectorizer(stop_words=french_stop_words)
    dtm = vectorizer.fit_transform(df['content'])

    print("Document-Term Matrix shape:", dtm.shape)

    dtm_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
    print(dtm_df.head())  

    nmf_model = NMF(n_components=n_topics, random_state=1)
    W = nmf_model.fit_transform(dtm)  
    H = nmf_model.components_ 

   
    topics = []
    for topic_idx, topic in enumerate(H):
        top_words_indices = topic.argsort()[-15:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_indices]
        topics.append({
            "topic_num": topic_idx + 1,
            "top_words": " ".join(top_words)  
        })
        print(f"\nTopic {topic_idx + 1}:")
        print(" ".join(top_words))
    
   
    df['topic'] = W.argmax(axis=1)

 
    topic_counts = df.groupby(['section', 'topic']).size().unstack(fill_value=0)
    topic_counts.to_csv('topic_counts_per_section.csv')
    print("Topic counts per section saved to topic_counts_per_section.csv")
    return topics, topic_counts

if __name__ == "__main__":
   
    df = scrape_rtbf_articles()

 
    topics, topic_counts = perform_topic_modeling(df, n_topics=1000)
    print(topic_counts)

df_section_total = pd.read_csv("section_totals.csv")  
df_section_article = pd.read_csv("section_article_counts.csv") 
df_merged = pd.merge(df_section_total, df_section_article, on="section")
df_merged.columns = df_merged.columns.str.strip()

fig= go.Figure()

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
fig.show()
