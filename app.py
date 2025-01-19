import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

class StockAnalyzer:
    def __init__(self, stock_symbol):
        """Initialize the stock analyzer with a stock symbol"""
        self.stock_symbol = stock_symbol
        self.yf_symbol = stock_symbol + ".NS"  # Adding .NS for Yahoo Finance
        self.yf_stock = yf.Ticker(self.yf_symbol)

    def get_yahoo_data(self, period="1y"):
        """Fetch data from Yahoo Finance"""
        try:
            historical_data = self.yf_stock.history(period=period)
            historical_data['MA50'] = historical_data['Close'].rolling(window=50).mean()
            historical_data['MA200'] = historical_data['Close'].rolling(window=200).mean()
            financials = self.yf_stock.financials

            return {
                "historical_data": historical_data,
                "financials": financials
            }
        except Exception as e:
            st.error(f"Error fetching Yahoo Finance data: {e}")
            return None

    def get_screener_data(self):
        """Fetch data from Screener.in"""
        url = f"https://www.screener.in/company/{self.stock_symbol}/consolidated/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            company_name = soup.find('h1').text.strip()

            key_metrics = {}
            for row in soup.select('#top-ratios li'):
                key = row.find('span', class_='name').text.strip()
                value = row.find('span', class_='number').text.strip()
                key_metrics[key] = value

            tables = soup.find_all('table')
            table_data = {}

            for i, table in enumerate(tables):
                table_name = f"Table_{i + 1}"
                if table.find_previous('h2'):
                    table_name = table.find_previous('h2').text.strip()

                headers = [th.text.strip() for th in table.find_all('th')]
                rows = []

                for tr in table.find_all('tr'):
                    cells = tr.find_all('td')
                    if cells:
                        row = [cell.text.strip() for cell in cells]
                        if len(row) == len(headers):
                            rows.append(row)

                if headers and rows:
                    table_data[table_name] = pd.DataFrame(rows, columns=headers)

            return {
                "Company Name": company_name,
                "Key Metrics": key_metrics,
                "Tables": table_data
            }

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching Screener.in data: {e}")
            return None

    def analyze(self):
        """Perform complete analysis using both data sources"""
        yahoo_data = self.get_yahoo_data()
        screener_data = self.get_screener_data()

        return {
            "yahoo_finance_data": yahoo_data,
            "screener_data": screener_data
        }

def format_data_for_llm(analysis_results):
    """Format the analysis results for LLM consumption"""
    formatted_data = ""

    # Format Yahoo Finance Data
    yf_data = analysis_results.get("yahoo_finance_data")
    if yf_data:
        formatted_data += "\nYahoo Finance Data:\n"
        formatted_data += "\nLatest Historical Data (last 5 days):\n"
        formatted_data += str(yf_data["historical_data"].tail())
        formatted_data += "\n\nFinancials:\n"
        formatted_data += str(yf_data["financials"])

    # Format Screener.in Data
    screener_data = analysis_results.get("screener_data")
    if screener_data:
        formatted_data += "\n\nScreener.in Data:\n"
        formatted_data += f"\nCompany Name: {screener_data['Company Name']}\n"

        formatted_data += "\nKey Metrics:\n"
        for key, value in screener_data["Key Metrics"].items():
            formatted_data += f"{key}: {value}\n"

        formatted_data += "\nTables:\n"
        for table_name, df in screener_data["Tables"].items():
            formatted_data += f"\n{table_name}:\n"
            formatted_data += str(df)

    return formatted_data

def display_stock_analysis(analysis_results):
    """Display the stock analysis results"""
    st.header("Stock Data")

    # Display Yahoo Finance Data
    if analysis_results["yahoo_finance_data"]:
        st.subheader("Historical Data (Last 5 Days)")
        st.dataframe(analysis_results["yahoo_finance_data"]["historical_data"].tail())

        st.subheader("Moving Averages")
        st.line_chart(analysis_results["yahoo_finance_data"]["historical_data"][["Close", "MA50", "MA200"]])

    # Display Screener.in Data
    if analysis_results["screener_data"]:
        st.subheader("Key Metrics")
        metrics_df = pd.DataFrame.from_dict(analysis_results["screener_data"]["Key Metrics"],
                                          orient='index',
                                          columns=['Value'])
        st.dataframe(metrics_df)

def main():
    st.title("Stock Analysis with AI")

    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'formatted_data' not in st.session_state:
        st.session_state.formatted_data = None
    if 'initial_analysis' not in st.session_state:
        st.session_state.initial_analysis = None

    # Set API key
    api_key = "YOUR_API_KEY"

    # Initialize LangChain with Claude
    llm = ChatAnthropic(
        anthropic_api_key=api_key,
        model="claude-3-sonnet-20240229"
    )

    # Main input area
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE, INFY, TCS)")

    # Initial Analysis Button
    if st.button("Analyze Stock") and stock_symbol:
        try:
            with st.spinner("Fetching and analyzing stock data..."):
                # Initialize stock analyzer
                analyzer = StockAnalyzer(stock_symbol)

                # Get analysis results and store in session state
                st.session_state.analysis_results = analyzer.analyze()
                st.session_state.formatted_data = format_data_for_llm(st.session_state.analysis_results)

                # Create prompt template for initial analysis
                prompt_template = PromptTemplate(
                    input_variables=["stock_data"],
                    template="""You are a financial analyst. Analyze the following stock data and provide:
                    1. A summary of the company's current financial position
                    2. Key strengths and weaknesses
                    3. Technical analysis based on moving averages
                    4. Recommendation for investors (short-term and long-term perspective)
                    5. Is this the right time to buy/sell/hold
                    Important formatting instructions:
                    - Use standard minus signs (-) instead of em dashes (−)
                    - Maintain consistent spacing between numbers and units
                    - Format all numerical values with up to 2 decimal places
                    - Use proper spacing after punctuation marks
                    - Ensure consistent font formatting throughout the response
                    Stock Data:
                    {stock_data}
                    
                    Provide your analysis in a clear, structured format with bullet points and sections."""
                )

                # Get initial AI analysis
                prompt = prompt_template.format(stock_data=st.session_state.formatted_data)
                st.session_state.initial_analysis = llm.invoke(prompt)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Display analysis results if available
    if st.session_state.analysis_results:
        # Display stock data
        display_stock_analysis(st.session_state.analysis_results)

        # Display initial AI analysis
        if st.session_state.initial_analysis:
            st.header("Initial AI Analysis")
            st.write(st.session_state.initial_analysis.content)

        # Follow-up questions section
        st.header("Ask More Questions")
        st.write("Feel free to ask any specific questions about this stock:")
        
        user_question = st.text_input("Your question:")
        
        if st.button("Get Answer") and user_question:
            follow_up_template = PromptTemplate(
                input_variables=["stock_data", "question"],
                template="""You are a financial analyst. Using the following stock data:

                {stock_data}

                Please answer this specific question about the stock:
                {question}

                Provide a clear and concise answer based on the available data.DONT SAY THAT PLS CONDUCT UR OWN STUDY OR CONSULT ANY 
                FINANCIAL ADVISOR, GIVE YOUR VIEWS CORRECTLY BASED ON YOUR ANALYSIS.
                At the end you can mention that These are just advises from our end.              """
            )
            
            follow_up_prompt = follow_up_template.format(
                stock_data=st.session_state.formatted_data,
                question=user_question
            )
            
            with st.spinner("Analyzing your question..."):
                follow_up_response = llm.invoke(follow_up_prompt)
                st.subheader("Answer to Your Question")
                st.write(follow_up_response.content)

    else:
        if not stock_symbol:
            st.info("Enter an stock symbol to begin analysis.")
    st.subheader("Disclaimer")
    st.write("The insights and analyses provided by this application are intended to offer financial advice based on the available data. However, the stock market is inherently unpredictable, and outcomes may vary. While we aim to assist and guide your decisions, the ultimate responsibility for investment choices lies with you. Please consider the risks involved before proceeding.")
if __name__ == "__main__":
    main()
