from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the Groq model ID to use across agents
GROQ_MODEL_ID = "deepseek-r1-distill-llama-70b"

# Web search agent for general information
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for financial and market information",
    model=Groq(id=GROQ_MODEL_ID, api_key=groq_api_key),
    tools=[DuckDuckGo()],
    instructions=[
        "Always include the sources",
        "Provide detailed and accurate information",
        "Format responses in a clear, readable manner",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Finance-specific agent for market analysis
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id=GROQ_MODEL_ID, api_key=groq_api_key),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
            company_info=True,
            income_statements=True,
            technical_indicators=True,
            historical_prices=True,
        ),
    ],
    instructions=[
        "Use tables to display financial data",
        "Provide comprehensive market analysis",
        "Include relevant metrics and indicators",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Multi-agent system combining both agents
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id=GROQ_MODEL_ID, api_key=groq_api_key),
    instructions=[
        "Always include sources",
        "Use tables to show financial data",
        "Provide comprehensive analysis combining market data and web information",
    ],
    show_tool_calls=True,
    markdown=True,
)


def analyze_stock(ticker_symbol):
    """
    Analyze a stock using the multi-agent system

    Args:
        ticker_symbol (str): The stock ticker symbol to analyze
    """
    query = f"Summarize analyst recommendations and tell me target prices or qualitative comments and which one to invest with the latest news for {ticker_symbol}"
    return multi_ai_agent.print_response(query, stream=True)


# Example usage
if __name__ == "__main__":
    analyze_stock("Apple and Google")
