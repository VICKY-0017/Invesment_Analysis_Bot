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

# Load environment variables from .env file
load_dotenv()

phi.api = os.getenv("Phidata_API")
groq_api_key = os.getenv("GROQ_API_KEY")

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

multi_ai_agent = Agent(
    team=[web_search_agent, Finance_agent],
    model=Groq(id="llama-3.1-70b-versatile", api_key=groq_api_key),
    instructions=["Always include sources", "Use tables to show the data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit Interface
st.title("Investment Analysis AI Agent")

# Sidebar Options
st.sidebar.title("Choose an AI Agent")
agent_option = st.sidebar.selectbox(
    "Select an Agent",
    ("Finance Agent", "Web Search Agent", "Multi AI Agent"),
)

# Input Section
query = st.text_input("Enter your query:")

# Function to format Analyst Recommendation in Markdown
def format_analyst_recommendation(analysis_data):
    # Assuming analysis_data is a dictionary containing the recommendation counts
    table = pd.DataFrame(analysis_data)
    return table.to_markdown(index=False)

# Function to format News in Markdown
def format_news(news_data):
    formatted_news = ""
    for idx, news_item in enumerate(news_data, start=1):
        formatted_news += f"**{idx}. {news_item['title']}**\n - {news_item['description']}\n\n"
    return formatted_news

# Function to process the result and ensure markdown formatting
def process_agent_result(agent_response):
    """
    This function processes the raw agent response and converts it to markdown format
    depending on whether it's a list, text, or table.
    """
    # Check if the result is in a structured format or plain text
    if isinstance(agent_response, dict):
        # If the result is a dictionary, format as a table
        if 'analyst_recommendations' in agent_response:
            return format_analyst_recommendation(agent_response['analyst_recommendations'])
        elif 'news' in agent_response:
            return format_news(agent_response['news'])
    elif isinstance(agent_response, str):
        # If it's a plain string (could be a summary), just return as is
        return agent_response
    else:
        return str(agent_response)  # For other unexpected formats

# Run the selected agent
if st.button("Run Query"):
    if agent_option == "Finance Agent":
        st.header("Finance Agent Results")
        if query:
            # Run the agent and get the result
            result = Finance_agent.run(query)  # Assuming `run` returns structured data
            # Process and display the structured result
            processed_result = process_agent_result(result)
            st.markdown(processed_result)  # Display structured result
        else:
            st.warning("Please enter a query!")

    elif agent_option == "Web Search Agent":
        st.header("Web Search Agent Results")
        if query:
            # Run the agent and get the result
            result = web_search_agent.run(query)  # Assuming `run` returns structured data
            # Process and display the structured result
            processed_result = process_agent_result(result)
            st.markdown(processed_result)  # Display structured result
        else:
            st.warning("Please enter a query!")

    elif agent_option == "Multi AI Agent":
        st.header("Multi AI Agent Results")
        if query:
            # Run the multi-agent system
            response = multi_ai_agent.run(query)
            
            # Process and display the structured result
            processed_result = process_agent_result(response)
            st.markdown(processed_result)  # Display structured result
            
        else:
            st.warning("Please enter a query!")
