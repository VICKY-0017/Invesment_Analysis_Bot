# This is the section of code that needs to be modified for news display
# Replace the current news display section with this:

# Inside the main content area, where news is displayed:
with col1:
    st.subheader("Latest News")
    if news_items:
        for item in news_items:
            if len(item) > 10:  # Avoid empty or very short items
                st.markdown(
                    f"""
                    <div style="
                        padding: 1rem;
                        background-color: white;
                        border-radius: 0.5rem;
                        margin-bottom: 1rem;
                        border-left: 4px solid #0066cc;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                    ">
                        <p style="margin: 0; color: #1f1f1f;">{item}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    else:
        st.info("Enter a query to see the latest news.")

# And modify the extract_news_and_table function to better handle news extraction:
def extract_news_and_table(text: str) -> Tuple[list, pd.DataFrame, str]:
    """
    Extract news items and table data from the text
    Returns: (news_items, dataframe, additional_info)
    """
    news_items = []
    table_data = None
    additional_info = ""
    
    # Split text into sections
    sections = text.split('\n\n')
    
    for section in sections:
        if "latest news" in section.lower():
            # Extract news items, handling different formats
            # Remove the "latest news" header
            news_text = section.lower().replace("latest news", "").strip()
            news_text = section.split("latest news")[-1].strip()
            
            # Split by multiple possible delimiters
            for delimiter in ['. ', ', ']:
                if delimiter in news_text:
                    items = news_text.split(delimiter)
                    news_items.extend([item.strip() for item in items if item.strip()])
                    break
            
            if not news_items and news_text:  # If no delimiters found but text exists
                news_items.append(news_text)
                
            # Clean up news items
            news_items = [item.strip(' .,') for item in news_items if item.strip()]
            news_items = [item for item in news_items if len(item) > 10]  # Remove very short items
            
        elif '|' in section:
            # Handle table data...
            [rest of the table processing code remains the same]
    
    return news_items, table_data, additional_info

# Add this CSS to the top of your app
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .element-container {
        margin-bottom: 1rem;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
        gap: 1rem;
    }
    </style>
""", unsafe_allow_html=True)
