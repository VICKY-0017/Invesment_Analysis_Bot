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
import json
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

def format_response(response: str) -> Dict[str, Any]:
    """Parse and structure the agent's response"""
    result = {
        "news": [],
        "table": None,
        "additional_info": ""
    }
    
    # Split response into sections
    sections = response.split('\n\n')
    
    for section in sections:
        if "latest news" in section.lower():
            # Extract news items
            news_items = [item.strip() for item in section.split(',') if item.strip()]
            result["news"] = news_items
        elif '|' in section:
            # Extract table data
            result["table"] = parse_table_from_markdown(section)
        elif "note" in section.lower() or "please" in section.lower():
            # Extract additional information
            result["additional_info"] = section.strip()
    
    return result

# Initialize Agents (same as before)
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
    .news-item {
        padding: 10px;
        border-left: 3px solid #0066cc;
        margin: 10px 0;
        background-color: #f8f9fa;
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
            # Get response from selected agent
            if agent_option == "Finance Agent":
                response = Finance_agent.print_response(query)
            elif agent_option == "Web Search Agent":
                response = web_search_agent.print_response(query)
            else:
                response = multi_ai_agent.print_response(query)
            
            # Parse and structure the response
            structured_result = format_response(response)
            
            # Display results in organized sections
            cols = st.columns(2)
            
            # Display news in left column
            with cols[0]:
                st.subheader("Latest News")
                for news_item in structured_result["news"]:
                    st.markdown(f"""
                        <div class="news-item">
                            {news_item}
                        </div>
                    """, unsafe_allow_html=True)
            
            # Display analysis in right column
            with cols[1]:
                if structured_result["table"] is not None:
                    st.subheader("Analysis")
                    st.dataframe(
                        structured_result["table"],
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Display additional information
            if structured_result["additional_info"]:
                st.markdown("---")
                st.info(structured_result["additional_info"])
