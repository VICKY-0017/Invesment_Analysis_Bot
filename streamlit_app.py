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
    Extract and filter news items, table data, and additional information
    Returns: (news_items, dataframe, additional_info)
    """
    news_items = []
    table_data = None
    additional_info = ""
    
    # Split text into lines
    lines = text.split('\n')
    current_section = None
    table_lines = []
    
    for line in lines:
        # Skip function calls and irrelevant lines
        if "transfer_task_to_" in line.lower():
            continue  # Skip these lines
        
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

# Initialize Multi-Agent
multi_ai_agent = Agent(
    team=[
        Agent(
            name="Web Search Agent",
            role="Search the web for the information",
            model=Groq(id="qwen-2.5-32b", api_key=groq_api_key),
            tools=[DuckDuckGo()],
            instructions=["Always include the sources"],
            show_tool_calls=True,
            markdown=True,
        ),
        Agent(
            name="Finance AI Agent",
            model=Groq(id="qwen-2.5-32b", api_key=groq_api_key),
            tools=[YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
                company_info=True,
            )],
            instructions=["Use tables to display the data"],
            show_tool_calls=True,
            markdown=True,
        )
    ],
    model=Groq(id="qwen-2.5-32b", api_key=groq_api_key),
    instructions=["Always include sources", "Use tables to show the data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit Interface
st.set_page_config(page_title="Investment Analysis AI Agent", layout="wide")

# Page Title
st.title("Investment Analysis AI Agent")

# Sidebar
with st.sidebar:
    st.title("Agent Information")
    st.markdown("This agent uses both web search and finance tools to provide insights.")
    
# Main content area
query = st.text_input("Enter your query:", placeholder="e.g., 'Analyze NVDA stock performance and latest news'")

if st.button("Run Query", type="primary"):
    if not query:
        st.warning("Please enter a query!")
    else:
        with st.spinner(f"Running analysis using the Multi-Agent..."):
            try:
                # Get response from the multi-agent
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
                                    st.write(f"- {item}")
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
                        st.subheader("Additional Information")
                        st.info(additional_info)
                
                else:
                    st.error("No response received from the agent.")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.write("Full error details:", str(e))  # Debug output
