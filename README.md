# capstone-challenges-Lastest-News
📊 RTBF News Scraper & Topic Modeling
📰
Welcome to the RTBF News Scraper & Topic Modeling project! This tool is designed to scrape the latest articles from the RTBF website, extract full article content, and perform topic modeling using Non-negative Matrix Factorization (NMF) to uncover trends and themes across news sections.

🛠️ Project Overview
This project uses Beautiful Soup for web scraping and sklearn for topic modeling, providing insights into the topics prevalent in recent RTBF news articles. It processes multiple sections of the website, including politics, sports, environment, technology, and more.

🏃‍♂️ Quick Start
To get started, follow these steps:

Here's a sample README file with descriptive sections, enhanced with emojis for better readability and engagement!

📊 RTBF News Scraper & Topic Modeling 📰
Welcome to the RTBF News Scraper & Topic Modeling project! This tool is designed to scrape the latest articles from the RTBF website, extract full article content, and perform topic modeling using Non-negative Matrix Factorization (NMF) to uncover trends and themes across news sections.

🛠️ Project Overview
This project uses Beautiful Soup for web scraping and sklearn for topic modeling, providing insights into the topics prevalent in recent RTBF news articles. It processes multiple sections of the website, including politics, sports, environment, technology, and more.

🏃‍♂️ Quick Start
To get started, follow these steps:

1. 📥 Installation
2. 🧑‍💻 Usage

Here's a sample README file with descriptive sections, enhanced with emojis for better readability and engagement!

📊 RTBF News Scraper & Topic Modeling 📰
Welcome to the RTBF News Scraper & Topic Modeling project! This tool is designed to scrape the latest articles from the RTBF website, extract full article content, and perform topic modeling using Non-negative Matrix Factorization (NMF) to uncover trends and themes across news sections.

🛠️ Project Overview
This project uses Beautiful Soup for web scraping and sklearn for topic modeling, providing insights into the topics prevalent in recent RTBF news articles. It processes multiple sections of the website, including politics, sports, environment, technology, and more.

🏃‍♂️ Quick Start
To get started, follow these steps:

1. 📥 Installation
Clone the repository:

bash
Copy code
git clone 
cd RTBF-news-scraper
Install the required packages:

bash
Copy code
pip install -r requirements.txt
2. 🧑‍💻 Usage
Run the main script to scrape articles and perform topic modeling:

bash
Copy code
python main.py

3. 📊 Output
rtbf_articles.csv: Contains the titles and links of scraped articles.
Topics Summary: Prints the top keywords representing each topic discovered via NMF.

🔍 Features
Multi-section Scraping: Automatically scrapes up to 10 pages per RTBF news section.
Full Article Extraction: Captures full text of articles for in-depth topic analysis.
Topic Modeling: Applies NMF to extract prominent themes from article content.

📁 Project Structure
main.py: Main script for scraping and topic modeling.
dashboard.py : deployment
requirements.txt: Lists all dependencies.

🧠 Insights
After scraping, the model identifies 8 main topics from the articles, showing the top keywords per topic for deeper analysis. Modify the n_topics variable in main.py to adjust the number of topics based on your preference.

Here's a sample README file with descriptive sections, enhanced with emojis for better readability and engagement!

📊 RTBF News Scraper & Topic Modeling 📰
Welcome to the RTBF News Scraper & Topic Modeling project! This tool is designed to scrape the latest articles from the RTBF website, extract full article content, and perform topic modeling using Non-negative Matrix Factorization (NMF) to uncover trends and themes across news sections.

🛠️ Project Overview
This project uses Beautiful Soup for web scraping and sklearn for topic modeling, providing insights into the topics prevalent in recent RTBF news articles. It processes multiple sections of the website, including politics, sports, environment, technology, and more.

🏃‍♂️ Quick Start
To get started, follow these steps:

1. 📥 Installation
Clone the repository:

bash
Copy code
git clone
cd RTBF-news-scraper
Install the required packages:

bash
Copy code
pip install -r requirements.txt
2. 🧑‍💻 Usage
Run the main script to scrape articles and perform topic modeling:

bash
Copy code
python main.py
3. 📊 Output
rtbf_articles.csv: Contains the titles and links of scraped articles.
Topics Summary: Prints the top keywords representing each topic discovered via NMF.
🔍 Features
Multi-section Scraping: Automatically scrapes up to 10 pages per RTBF news section.
Full Article Extraction: Captures full text of articles for in-depth topic analysis.
Topic Modeling: Applies NMF to extract prominent themes from article content.
📁 Project Structure
main.py: Main script for scraping and topic modeling.
fetch_article_content.py: Contains functions for web scraping and article content extraction.
requirements.txt: Lists all dependencies.
🧠 Insights
After scraping, the model identifies 5 main topics from the articles, showing the top keywords per topic for deeper analysis. Modify the n_topics variable in main.py to adjust the number of topics based on your preference.

🚀 Future Improvements
Enhance NLP Preprocessing: Add lemmatization and stemming to refine text analysis.
Implement LDA: Add another model (Latent Dirichlet Allocation) for topic modeling comparison.
Automate Scraping Frequency: Enable scheduled scraping for continuous updates.


Check out my app built with [Streamlit 🌐](https://minh-f5-capstone-challenges-lastest-news-app-y0psdd.streamlit.app/).

![Breaking News GIF](https://media.giphy.com/media/6H9Oqec3e7cFuHzr4K/giphy.gif)
