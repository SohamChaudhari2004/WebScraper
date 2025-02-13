# Universal Web Scraper

An AI-powered web scraper that extracts structured data from websites, processes it, and allows querying using NLP. Supports JSON/CSV export and dynamic field selection.

## Features
- Extracts and processes website data using Selenium
- Converts raw HTML into structured Markdown
- Supports multiple AI models for intelligent data formatting
- Token usage tracking and cost estimation
- Export data in JSON or CSV formats
- Query scraped data using NLP

## Installation necessary dependencies
```
# Clone the repository
git clone https://github.com/SohamChaudhari2004/WebScraper.git
```
```
# Create a virtual environment in root 
python -m venv venv
venv/Scripts/activate  # On mac use `source venv/bin/activate`
```
```
# Install dependencies
pip install -r requirements.txt
```

## Run the streamlit app
```
streamlit run streamlit_app.py
```
## Environment Variables
Create a `.env` file and add:
```env
GROQ_API_KEY=your_api_key
```
### How to use the application
1. Enter the website URL.
2. Select the AI model for formatting.
3. Add fields to extract.
4. Click "Scrape" to extract and process data.
5. Download the results in JSON or CSV format.
6. Use the query feature to get insights from the scraped data(to be implemented).

## Requirements
- Python 3.10+
- Selenium
- Streamlit
- Groq API Key (for AI processing)
- Pandas
