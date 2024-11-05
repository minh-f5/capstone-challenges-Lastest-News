import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

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

def perform_topic_modeling(df, n_topics=8):
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

    # Extract topics and top words
    topics = []
    for topic_idx, topic in enumerate(H):
        top_words_indices = topic.argsort()[-15:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_indices]
        topics.append({
            "topic_num": topic_idx + 1,
            "top_words": ", ".join(top_words)  # Join top words as a single string
        })
        print(f"\nTopic {topic_idx + 1}:")
        print(" ".join(top_words))
    
    # Save topics to CSV
    topics_df = pd.DataFrame(topics)
    topics_df.to_csv('topics.csv', index=False)
    print("Topics saved to topics.csv")
    return topics

if __name__ == "__main__":
    # Scrape articles
    df = scrape_rtbf_articles()

    # Perform topic modeling
    topics = perform_topic_modeling(df, n_topics=8)
