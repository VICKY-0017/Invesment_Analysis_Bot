import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq
import os
import streamlit as st
import pandas as pd
from typing import Dict, Any

# Load environment variables
load_dotenv()
phi.api = os.getenv("Phidata_API")
groq_api_key = os.getenv("GROQ_API_KEY")

def parse_table_from_markdown(markdown_text: str) -> pd.DataFrame:
    """Extract table data from markdown text and convert to pandas DataFrame"""
    try:
        # Find table content
        table_lines = [line.strip() for line in markdown_text.split('\n') if '|' in line]
        if not table_lines:
            return None
        
        # Parse headers
        headers = [col.strip() for col in table_lines[0].split('|') if col.strip()]
        
        # Parse data rows
        data = []
        for line in table_lines[2:]:  # Skip header separator line
            row_data = [col.strip() for col in line.split('|') if col.strip()]
            if row_data:
                data.append(row_data)
        
        return pd.DataFrame(data, columns=headers)
    except Exception:
        return None

def display_results(response_text: str):
    """Display the results in a structured format"""
    try:
        # Split response into sections
        sections = response_text.split('\n\n')
        
        cols = st.columns(2)
        
        # Left column for news
        with cols[0]:
            st.subheader("Latest News")
            for section in sections:
                if "latest news" in section.lower():
                    news_items = [item.strip() for item in section.split('.') if item.strip()]
                    for item in news_items:
                        if item and len(item) > 10:  # Avoid empty or very short items
                            st.markdown(f"""
                                <div style="padding: 10px; border-left: 3px solid #0066cc; 
                                margin: 10px 0; background-color: #f8f9fa;">
                                    {item}.
                                </div>
                            """, unsafe_allow_html=True)
        
        # Right column for analysis
        with cols[1]:
            st.subheader("Analysis")
            for section in sections:
                if '|' in section:
                    df = parse_table_from_markdown(section)
                    if df is not None:
                        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Additional information at the bottom
        for section in sections:
            if "note" in section.lower() or "please" in section.lower():
                st.markdown("---")
                st.info(section.strip())

    except Exception as e:
        st.error(f"Error processing results: {str(e)}")

# Initialize Agents
web_search_agent = Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api_key),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,
)

Finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api_key),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
            company_info=True,
        ),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, Finance_agent],
    model=Groq(id="llama-3.1-70b-versatile", api_key=groq_api_key),
    instructions=["Always include sources", "Use tables to show the data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit Interface
st.set_page_config(page_title="Investment Analysis AI Agent", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Investment Analysis AI Agent")

# Sidebar
with st.sidebar:
    st.title("Agent Selection")
    agent_option = st.selectbox(
        "Select an Agent",
        ("Finance Agent", "Web Search Agent", "Multi AI Agent"),
    )
    
    # Add agent descriptions
    st.markdown("### Agent Descriptions")
    if agent_option == "Finance Agent":
        st.info("Specializes in financial analysis and stock market data.")
    elif agent_option == "Web Search Agent":
        st.info("Searches the web for latest news and information.")
    else:
        st.info("Combines both financial analysis and web search capabilities.")

# Main content area
query = st.text_input("Enter your query:", placeholder="e.g., 'Analyze NVDA stock performance and latest news'")

if st.button("Run Query", type="primary"):
    if not query:
        st.warning("Please enter a query!")
    else:
        with st.spinner(f"Running analysis using {agent_option}..."):
            try:
                # Get response from selected agent
                if agent_option == "Finance Agent":
                    response = Finance_agent.run(query)
                elif agent_option == "Web Search Agent":
                    response = web_search_agent.run(query)
                else:
                    response = multi_ai_agent.run(query)
                
                # Display the structured results
                if response and hasattr(response, 'content'):
                    display_results(response.content)
                else:
                    st.error("No response received from the agent.")
                    
            except Exception as e:
                st.error(f"Error running query: {str(e)}")
