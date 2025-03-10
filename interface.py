import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app

# Load environment variables from .env file
load_dotenv()

phi.api = os.getenv("Phidata_API")
groq_api_key = os.getenv("GROQ_API_KEY")

web_search_agent = Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="deepseek-r1-distill-llama-70b", api_key=groq_api_key),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,
)

Finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="deepseek-r1-distill-llama-70b", api_key=groq_api_key),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
            company_info=True,
        ),
    ],
    instructions=["use tables to display the Data"],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[Finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("interface:app", reload=True)
