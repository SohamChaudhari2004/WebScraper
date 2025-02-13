import streamlit as st
from streamlit_tags import st_tags_sidebar
import pandas as pd
import json
from datetime import datetime
from scraper import (
    fetch_html_selenium, save_raw_data, format_data, save_formatted_data, 
    calculate_price, html_to_markdown_with_readability, create_dynamic_listing_model, 
    create_listings_container_model
)
from assets import PRICING

# Initialize Streamlit app with a wide layout
st.set_page_config(page_title="Universal Web Scraper", layout="wide")
st.title("Universal Web Scraper")

# Inject custom CSS to increase width
st.markdown(
    """
    <style>
        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .dataframe-container {
            width: 100% !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar components
st.sidebar.title("Web Scraper Settings")
model_selection = st.sidebar.selectbox("Select Model", options=list(PRICING.keys()), index=0)
url_input = st.sidebar.text_input("Enter URL")

# Tags input in the sidebar
tags = st.sidebar.empty()
tags = st_tags_sidebar(
    label='Enter Fields to Extract:',
    text='Press enter to add a tag',
    value=[],
    suggestions=[],
    maxtags=-1,
    key='tags_input'
)

st.sidebar.markdown("---")

# Process tags into a list
fields = tags

# Initialize session states
if 'perform_scrape' not in st.session_state:
    st.session_state['perform_scrape'] = False
if 'selected_format' not in st.session_state:
    st.session_state['selected_format'] = "csv"  # Default format

# Scraping function
def perform_scrape():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_html = fetch_html_selenium(url_input)
    markdown = html_to_markdown_with_readability(raw_html)
    save_raw_data(markdown, timestamp)
    DynamicListingModel = create_dynamic_listing_model(fields)
    DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
    formatted_data, tokens_count = format_data(markdown, DynamicListingsContainer, DynamicListingModel, model_selection)
    input_tokens, output_tokens, total_cost = calculate_price(tokens_count, model=model_selection)
    df = save_formatted_data(formatted_data, timestamp)

    return df, formatted_data, markdown, input_tokens, output_tokens, total_cost, timestamp

# Handle button press for scraping
if st.sidebar.button("Scrape"):
    with st.spinner('Scraping data, please wait...'):
        st.session_state['results'] = perform_scrape()
        st.session_state['perform_scrape'] = True

# Display results if scraping is done
if st.session_state.get('perform_scrape'):
    df, formatted_data, markdown, input_tokens, output_tokens, total_cost, timestamp = st.session_state['results']

    # Token usage info
    st.sidebar.markdown("## Token Usage")
    st.sidebar.markdown(f"**Input Tokens:** {input_tokens}")
    st.sidebar.markdown(f"**Output Tokens:** {output_tokens}")
    st.sidebar.markdown(f"**Total Cost:** :green-background[***${total_cost:.4f}***]")

    # Convert formatted data to JSON string
    if isinstance(formatted_data, str):
        data_dict = json.loads(formatted_data)
    else:
        data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

    # Extract main data
    first_key = next(iter(data_dict))
    main_data = data_dict[first_key]  
    df = pd.DataFrame(main_data) 

    # Format selection buttons
    st.markdown("## Select Output Format")
    col1, col2, col3 = st.columns(3)
    if col1.button("JSON", use_container_width=True):
        st.session_state['selected_format'] = "JSON"
    if col2.button("CSV", use_container_width=True):
        st.session_state['selected_format'] = "CSV"

    # Display the selected format
    st.markdown("## Output Preview")
    if st.session_state['selected_format'] == "JSON":
        st.code(json.dumps(data_dict, indent=4), language="json")
    elif st.session_state['selected_format'] == "CSV":
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Create download buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Download JSON", data=json.dumps(data_dict, indent=4), file_name=f"{timestamp}_data.json")
    with col2:
        st.download_button("Download CSV", data=df.to_csv(index=False), file_name=f"{timestamp}_data.csv")
