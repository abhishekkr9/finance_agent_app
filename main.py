import streamlit as st
from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.openbb import OpenBBTools
from agno.workflow import Workflow

MODEL_ID = "llama3.2:latest"

class StockAnalysis(Workflow):
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id)

        self.finance_agent2 = Agent(
            model=Ollama(id=MODEL_ID),
            tools=[OpenBBTools(company_news=True, company_profile=True)],
            markdown=True,
            show_tool_calls=True,
            description="You are an investor gathering additional financial information and news about a company.",
            instructions=[
                "First extract the stock ticker from the [STOCK:...] format in the query",
                "Prioritize information with strong analyst recommendations and positive stock fundamentals (e.g., low P/E ratio, high growth potential).",
                "Analyze trends in past, present, and future news to understand their impact on stock prices.",
                "Only use financial data from reliable sources.",
                "Compare fundamentals with industry averages",
                "Format response using markdown tables",
                "If not found any result, move on to next agent"
            ],
        )

        self.web_agent = Agent(
            model=Ollama(id=MODEL_ID),
            tools=[DuckDuckGoTools()],
            show_tool_calls=True,
            markdown=True,
            description="You are an investor gathering general news and analysis on a company.",
            instructions=[
                "Extract company name from [STOCK:...] parameter",
                "Search for recent news and analysis about the specified company",
                "Focus on financial results, partnerships, and regulatory changes",
                "Prioritize sources: Moneycontrol, Economic Times, Business Standard"
            ],
        )

        self.agent_team = Agent(
            team=[self.finance_agent2, self.web_agent],
            model=Ollama(id=MODEL_ID),
            instructions=[
                "Always include sources",
                "Use tables to display data",
                "Gather all information from all agents and provide a comprehensive answer.",
                "Provide a final recommendation whether to invest or not, based on the gathered information.",
                "Conclude with 'Yes' or 'No' based on your evaluation and mention sources",
            ],
            show_tool_calls=True,
            markdown=True,
        )

    def run(self, stock_name: str) -> RunResponse:
        query = f"[STOCK:{stock_name}] Based on current scenario, should I invest in this stock?"
        response = self.agent_team.run(query, stream=False)
        print(f"Agent Team Response: {response}")
        return response

st.set_page_config(page_title="Stock Analysis App", page_icon="📈")
st.title("Stock Analysis App")
st.write("Enter a stock name or ticker symbol and click Analyze to see if you should invest in the stock.")

with st.sidebar:
    st.header("Input")
    stock_name = st.text_input("Stock Name or Ticker Symbol", value="Tesla")
    analyze_button = st.button("Analyze")

status_placeholder = st.empty()
result_placeholder = st.empty()

if analyze_button:
    if stock_name:
        status_placeholder.write("Analyzing, please wait...")
        analysis_workflow = StockAnalysis(session_id=f"stock-analysis-{stock_name}")
        response = analysis_workflow.run(stock_name=stock_name)
        if response and response.content:
            result_placeholder.markdown(response.content)
            status_placeholder.empty()
    else:
        st.error("Please enter a valid stock name or ticker symbol.")