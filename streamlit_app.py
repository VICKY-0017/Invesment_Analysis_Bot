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
import re
from typing import Dict, Any, Tuple

# Load environment variables
load_dotenv()
phi.api = os.getenv("Phidata_API")
groq_api_key = os.getenv("GROQ_API_KEY")

def extract_news_and_table(text: str) -> Tuple[list, pd.DataFrame, str]:
    """
    Extract news items and table data from the text
    Returns: (news_items, dataframe, additional_info)
    """
    # Debug logging
    # st.write("Raw response:", text)  # Debug output
    
    news_items = []
    table_data = None
    additional_info = ""
    
    # Split text into lines
    lines = text.split('\n')
    current_section = None
    table_lines = []
    
    for line in lines:
        # Identify sections
        if "latest news" in line.lower():
            current_section = "news"
            continue
        elif "|" in line and "-|-" not in line:
            current_section = "table"
            table_lines.append(line)
        elif "note:" in line.lower() or "please note" in line.lower():
            current_section = "info"
            additional_info += line + "\n"
        # Process sections
        elif current_section == "news" and line.strip():
            news_items.extend([item.strip() for item in line.split('.') if item.strip()])
        elif current_section == "table":
            table_lines.append(line)
        elif current_section == "info" and line.strip():
            additional_info += line + "\n"
    
    # Process table if exists
    if table_lines:
        try:
            # Remove empty columns
            table_lines = [re.sub(r'\|\s*\|', '|', line) for line in table_lines]
            # Remove leading/trailing |
            table_lines = [line.strip('|') for line in table_lines]
            
            # Extract headers
            headers = [col.strip() for col in table_lines[0].split('|')]
            
            # Extract data
            data = []
            for line in table_lines[2:]:  # Skip separator line
                if line.strip():
                    row = [cell.strip() for cell in line.split('|')]
                    data.append(row)
            
            table_data = pd.DataFrame(data, columns=headers)
        except Exception as e:
            st.error(f"Error processing table: {str(e)}")
    
    return news_items, table_data, additional_info.strip()

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

# Custom CSS
st.markdown("""
    <style>
    .news-item {
        padding: 15px;
        border-left: 4px solid #0066cc;
        margin: 15px 0;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
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
                
                # Process response
                if response:
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Extract and display information
                    news_items, table_data, additional_info = extract_news_and_table(response_text)
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    # Display news
                    with col1:
                        st.subheader("Latest News")
                        if news_items:
                            for item in news_items:
                                if len(item) > 10:  # Avoid empty or very short items
                                    st.markdown(f"""
                                        <div class="news-item">
                                            {item}
                                        </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No news items found in the response.")
                    
                    # Display analysis
                    with col2:
                        st.subheader("Analysis")
                        if table_data is not None and not table_data.empty:
                            st.dataframe(table_data, use_container_width=True, hide_index=True)
                        else:
                            st.info("No analysis table found in the response.")
                    
                    # Display additional information
                    if additional_info:
                        st.markdown("---")
                        st.info(additional_info)
                
                else:
                    st.error("No response received from the agent.")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.write("Full error details:", str(e))  # Debug output
