import streamlit as st
import pandas as pd
from io import BytesIO
import os

# Import the newer google-genai package
import google.generativeai as genai

# Configure API key directly in the app
API_KEY = "AIzaSyCdEYjUnK1fMEDBAYkbXHFV6BEY6reAVm8"

# Configure Google Generative AI
def configure_gemini(api_key):
    try:
        # For the newer google.generativeai package
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Google Generative AI: {e}")
        return False

# Prompt templates for different financial documents
PROMPT_TEMPLATES = {
    "balance_sheet": """
    You are a financial analyst. Analyze the following balance sheet data and provide a clear summary including:
    - Key financial metrics (total assets, total liabilities, equity)
    - Trends over time
    - Financial health indicators
    - Any anomalies or notable changes

    Data:
    {data}
    """,
    "profit_loss": """
    You are a financial analyst. Analyze the following profit and loss statement data and provide a clear summary including:
    - Revenue trends
    - Profit margins
    - Major expense categories
    - Profitability analysis
    - Any anomalies or notable changes

    Data:
    {data}
    """,
    "cash_flow": """
    You are a financial analyst. Analyze the following cash flow statement data and provide a clear summary including:
    - Operating cash flow trends
    - Investing activities
    - Financing activities
    - Liquidity analysis
    - Any anomalies or notable changes

    Data:
    {data}
    """
}

def read_financial_data(uploaded_file):
    """Read financial data from CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def clean_financial_data(df):
    """Clean and normalize financial data"""
    try:
        # Convert numeric columns using recommended approach (fixes deprecation warning)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric, handle errors explicitly
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep original value if conversion fails
                    pass

        # Drop rows with all NA values
        df = df.dropna(how='all')

        return df
    except Exception as e:
        st.error(f"Error cleaning data: {e}")
        return df

def create_enhanced_visualizations(df, document_type):
    """Create enhanced visualizations that work with various data formats"""
    try:
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            st.info("No numeric data available for visualization")
            return

        st.subheader("📊 Available Numeric Columns")
        st.write(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")

        # Always show all numeric data trends
        if len(numeric_cols) > 0:
            st.subheader("📈 All Financial Trends")
            st.line_chart(df[numeric_cols])

        # Document-type specific visualizations
        if document_type == "balance_sheet":
            st.subheader("🏦 Balance Sheet Analysis")

            # Look for asset/liability patterns
            asset_cols = [col for col in numeric_cols if 'asset' in col.lower()]
            liability_cols = [col for col in numeric_cols if 'liabilit' in col.lower()]

            if asset_cols and liability_cols:
                st.subheader("Assets vs Liabilities Comparison")
                comparison_df = pd.DataFrame()
                for asset_col in asset_cols[:2]:  # Limit to 2 for clarity
                    for liability_col in liability_cols[:2]:  # Limit to 2 for clarity
                        if len(comparison_df.columns) < 4:  # Limit total columns
                            comparison_df[asset_col] = df[asset_col]
                            comparison_df[liability_col] = df[liability_col]
                if not comparison_df.empty:
                    st.line_chart(comparison_df)

            # Look for equity/capital patterns
            equity_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['equity', 'capital', 'retained'])]
            if equity_cols:
                st.subheader("Equity/Capital Analysis")
                st.line_chart(df[equity_cols])

        elif document_type == "profit_loss":
            st.subheader("💰 Profit & Loss Analysis")

            # Look for revenue/income patterns
            revenue_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['revenue', 'sales', 'income'])]
            if revenue_cols:
                st.subheader("Revenue/Income Trends")
                st.line_chart(df[revenue_cols])

            # Look for profit patterns
            profit_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['profit', 'net', 'earning'])]
            if profit_cols:
                st.subheader("Profit Trends")
                st.line_chart(df[profit_cols])

            # Calculate profit margins if we have both revenue and profit
            if revenue_cols and profit_cols:
                try:
                    margin = (df[profit_cols[0]] / df[revenue_cols[0]]) * 100
                    st.subheader("Profit Margin (%)")
                    st.line_chart(margin)
                except:
                    st.info("Could not calculate profit margin due to data format")

            # Look for expense patterns
            expense_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['expense', 'cost', 'spend'])]
            if expense_cols:
                st.subheader("Expense Analysis")
                st.line_chart(df[expense_cols])

        elif document_type == "cash_flow":
            st.subheader("💳 Cash Flow Analysis")

            # Look for cash flow patterns
            cash_flow_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['cash', 'flow'])]
            if cash_flow_cols:
                st.subheader("Cash Flow Trends")
                st.line_chart(df[cash_flow_cols])
            else:
                # If no specific cash flow columns, show all data again with different label
                st.subheader("Financial Flow Analysis")
                st.line_chart(df[numeric_cols])

        # Additional analysis: Show correlations if we have enough columns
        if len(numeric_cols) >= 2:
            st.subheader("🔗 Data Correlations")
            correlation_matrix = df[numeric_cols].corr()
            st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm'))

    except Exception as e:
        st.error(f"Error creating visualizations: {e}")

def generate_ai_summary(data, document_type, api_key):
    """Generate AI summary using Google Generative AI"""
    try:
        if not configure_gemini(api_key):
            return None

        # Select appropriate prompt template
        prompt_template = PROMPT_TEMPLATES.get(document_type, PROMPT_TEMPLATES["balance_sheet"])

        # Format the prompt with data
        prompt = prompt_template.format(data=data.head(20).to_string())

        # Generate summary using the newer API
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        st.error(f"Error generating AI summary: {e}")
        return None

def main():
    st.set_page_config(page_title="Gemini Pro Financial Decoder", layout="wide")

    st.title("💰 Gemini Pro Financial Decoder")
    st.markdown("""
    Transform financial documents into actionable insights using Google Generative AI.
    Upload your balance sheets, profit & loss statements, or cash flow statements.
    """)

    # Initialize session state for API key
    if 'api_key' not in st.session_state:
        # Use the configured API key
        st.session_state.api_key = API_KEY

    # Sidebar for API key and document type selection
    with st.sidebar:
        st.header("Configuration")

        # API Key input with better visibility
        api_key_input = st.text_input(
            "Google Generative AI API Key",
            type="password",
            value=st.session_state.api_key,
            key="api_key_input",
            help="Enter your API key from Google AI Studio"
        )

        # Update session state if input changes
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            if api_key_input:
                os.environ["GOOGLE_API_KEY"] = api_key_input
                st.success("✅ API Key saved successfully!")
            else:
                st.info("ℹ️ Enter your Google Generative AI API key to enable AI analysis")

        # Show API key status
        if st.session_state.api_key:
            st.success("🔑 API Key is configured")
            # Button to clear API key
            if st.button("Clear API Key"):
                st.session_state.api_key = ""
                os.environ.pop("GOOGLE_API_KEY", None)
                st.experimental_rerun()
        else:
            st.warning("⚠️ API Key required for AI analysis")

        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Get your Google Generative AI API key from [Google AI Studio](https://aistudio.google.com/)
        2. Enter your API key above
        3. Upload your financial documents (CSV or Excel)
        4. Select the document type
        5. View AI-generated insights and visualizations
        """)

    # File upload
    st.header("📁 Upload Financial Document")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload balance sheet, profit & loss, or cash flow statement"
    )

    if uploaded_file is not None:
        # Document type selection
        document_type = st.selectbox(
            "Select Document Type",
            ["balance_sheet", "profit_loss", "cash_flow"],
            format_func=lambda x: {
                "balance_sheet": "Balance Sheet",
                "profit_loss": "Profit & Loss Statement",
                "cash_flow": "Cash Flow Statement"
            }[x]
        )

        # Process data
        with st.spinner("Processing financial data..."):
            df = read_financial_data(uploaded_file)
            if df is not None:
                df = clean_financial_data(df)

                # Display raw data preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head())

                # Generate AI summary if API key is provided
                if st.session_state.api_key:
                    st.subheader("🤖 AI Financial Analysis")

                    with st.spinner("Generating AI insights..."):
                        summary = generate_ai_summary(df, document_type, st.session_state.api_key)

                        if summary:
                            st.success("🎉 AI analysis completed!")
                            st.markdown(summary)

                            # Download summary as text file
                            st.download_button(
                                label="📥 Download AI Summary",
                                data=summary,
                                file_name="financial_summary.txt",
                                mime="text/plain"
                            )
                else:
                    st.warning("⚠️ Please enter your Google Generative AI API key in the sidebar to enable AI analysis.")

                # Create enhanced visualizations
                st.subheader("📈 Financial Visualizations")
                create_enhanced_visualizations(df, document_type)

if __name__ == "__main__":
    main()
