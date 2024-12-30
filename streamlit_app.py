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
st.title("Invesment Analysis AI Agent")

# Sidebar Options
st.sidebar.title("Choose an AI Agent")
agent_option = st.sidebar.selectbox(
    "Select an Agent",
    ("Finance Agent", "Web Search Agent", "Multi AI Agent"),
)

# Input Section
query = st.text_input("Enter your query:")

# Run the selected agent
if st.button("Run Query"):
    if agent_option == "Finance Agent":
        st.header("Finance Agent Results")
        if query:
            result = Finance_agent.print_response(query)
            st.markdown(result)
        else:
            st.warning("Please enter a query!")

    elif agent_option == "Web Search Agent":
        st.header("Web Search Agent Results")
        if query:
            result = web_search_agent.print_response(query)
            st.markdown(result)
        else:
            st.warning("Please enter a query!")

    elif agent_option == "Multi AI Agent":
        st.header("Multi AI Agent Results")
        if query:
            result = multi_ai_agent.print_response(query)
            st.markdown(result)
        else:
            st.warning("Please enter a query!")
