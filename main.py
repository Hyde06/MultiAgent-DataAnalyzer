import streamlit as st
import os
from dotenv import load_dotenv
from core.data_processor import DataProcessor
from core.llm_utils import generate_text
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from core.advanced_analysis import AdvancedAnalyzer
from core.column_analyzer import ColumnAnalyzer
from core.sales_analyzer import SalesDataAnalyzer
from core.business_intelligence import BusinessIntelligence
from core.advanced_forecasting import AdvancedForecasting
from core.agents import SalesAgent, BIAgent, ForecastAgent, VisualizationAgent, SalesVisualizationAgent, AIAnalysisAgent

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Multi-Agent Data Analysis",
    page_icon="📊",
    layout="wide"
)

def create_sales_visualizations(df, sales_analyzer):
    """Create sales-specific visualizations"""
    visualizations = []
    
    # Sales trends
    sales_trends = sales_analyzer.analyze_sales_trends()
    if sales_trends and 'daily_trend' in sales_trends:
        fig = px.line(sales_trends['daily_trend'], 
                     x=sales_trends['daily_trend'].columns[0],
                     y=sales_trends['daily_trend'].columns[1],
                     title="Daily Sales Trend")
        visualizations.append(("Daily Sales Trend", fig))
    
    # Product performance
    product_perf = sales_analyzer.analyze_product_performance()
    if product_perf and 'top_products' in product_perf:
        fig = px.bar(product_perf['top_products'].reset_index(),
                    x=product_perf['top_products'].index.name,
                    y='sum',
                    title="Top 10 Products by Sales")
        visualizations.append(("Top Products", fig))
    
    # Customer segments
    customer_segments = sales_analyzer.analyze_customer_segments()
    if customer_segments and 'customer_segments' in customer_segments:
        fig = px.box(customer_segments['customer_segments'],
                    x='segment',
                    y='sum',
                    title="Customer Segment Distribution")
        visualizations.append(("Customer Segments", fig))
    
    return visualizations

def main():
    st.title("Multi-Agent Data Analysis Automation")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    # Sidebar
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            if st.session_state.current_file != uploaded_file.name:
                st.session_state.processed_data = None
                st.session_state.current_file = uploaded_file.name
                
                # Save uploaded file
                file_path = os.path.join("data/raw", uploaded_file.name)
                os.makedirs("data/raw", exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                
                # Process the file
                with st.spinner("Processing data..."):
                    data_processor = DataProcessor()
                    result = data_processor.process_file(file_path)
                    if result is not None:
                        st.session_state.processed_data = result
                    else:
                        st.error("Error processing the file. Please check the file format and try again.")
    
    # Main content
    if st.session_state.processed_data is not None:
        result = st.session_state.processed_data
        stats = result['statistics']
        df = result['data']
        
        # Initialize agents
        sales_agent = SalesAgent(df)
        bi_agent = BIAgent(df)
        forecast_agent = ForecastAgent(df)
        viz_agent = VisualizationAgent(df)
        sales_viz_agent = SalesVisualizationAgent(df)

        # Run deterministic analysis for each agent
        sales_results = sales_agent.analyze()
        bi_results = bi_agent.analyze()
        forecast_results = forecast_agent.analyze()
        viz_results = viz_agent.analyze()
        sales_viz_results = sales_viz_agent.analyze()
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Overview", "Sales Analysis", "Business Intelligence", 
            "Forecasting", "Visualizations", "AI Analysis", "Data Chat"
        ])
        
        with tab1:
            st.subheader("Data Overview")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", stats['shape'][0])
            with col2:
                st.metric("Columns", stats['shape'][1])
            with col3:
                st.metric("Missing Values", sum(stats['missing_values'].values()))
            
            # Data types and missing values
            col1, col2 = st.columns(2)
            with col1:
                st.write("Data Types:")
                st.write(pd.DataFrame({
                    'Column': list(stats['dtypes'].keys()),
                    'Type': list(stats['dtypes'].values())
                }))
            
            with col2:
                st.write("Missing Values:")
                st.write(pd.Series(stats['missing_values']))
        
        with tab2:
            st.subheader("Sales Analysis")
            
            # Deterministic outputs (as before)
            key_metrics = sales_results['key_metrics']
            if key_metrics:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Sales", f"${key_metrics['total_sales']:,.2f}")
                with col2:
                    st.metric("Average Order Value", f"${key_metrics['average_order_value']:,.2f}")
                with col3:
                    st.metric("Total Orders", f"{key_metrics['total_orders']:,}")
                with col4:
                    st.metric("Unique Customers", f"{key_metrics['unique_customers']:,}")
            st.write("### Sales Trends")
            sales_trends = sales_results['sales_trends']
            if sales_trends:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Monthly Sales Trend")
                    st.line_chart(sales_trends['monthly_trend'])
                with col2:
                    st.write(f"Year-over-Year Growth: {sales_trends['yoy_growth']:.1f}%")
            st.write("### Product Performance")
            product_perf = sales_results['product_performance']
            if product_perf:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Top Products")
                    st.dataframe(product_perf['top_products'])
                with col2:
                    if product_perf['category_sales'] is not None:
                        st.write("Category Sales")
                        st.bar_chart(product_perf['category_sales'])
            st.write("### Customer Analysis")
            customer_segments = sales_results['customer_segments']
            if customer_segments:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Customer Segments")
                    st.dataframe(customer_segments['customer_segments'])
                with col2:
                    st.write("RFM Analysis")
                    st.dataframe(customer_segments['rfm_analysis'])
            # AI summary at the bottom
            st.markdown("---")
            st.subheader("AI Sales Summary")
            st.markdown(sales_agent.summarize())
        
        with tab3:
            st.subheader("Business Intelligence")
            
            # Deterministic outputs (as before)
            kpis = bi_results['kpis']
            if kpis:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sales", f"${kpis['total_sales']:,.2f}")
                    st.metric("Sales Growth", f"{kpis['sales_growth']:.1f}%")
                with col2:
                    st.metric("Average Order Value", f"${kpis['average_order_value']:,.2f}")
                    st.metric("Customer Retention", f"{kpis['customer_retention']:.1f}%")
                with col3:
                    st.metric("Profit Margin", f"{kpis['profit_margin']:.1f}%")
                    st.metric("Inventory Turnover", f"{kpis['inventory_turnover']:.1f}")
            st.write("### Territory Analysis")
            territory_analysis = bi_results['territory']
            if territory_analysis:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Territory Performance")
                    st.dataframe(territory_analysis['metrics'])
                with col2:
                    st.write("Territory Growth")
                    st.dataframe(territory_analysis['growth'])
            st.write("### Competitor Analysis")
            competitor_analysis = bi_results['competitors']
            if competitor_analysis:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Market Share")
                    st.dataframe(competitor_analysis['market_shares'])
                with col2:
                    st.write("Growth Rates")
                    st.dataframe(competitor_analysis['growth_rates'])
            st.write("### Inventory Optimization")
            inventory_analysis = bi_results['inventory']
            if inventory_analysis:
                st.write("Inventory Metrics")
                st.dataframe(inventory_analysis['metrics'])
                st.write("Recommendations")
                for rec in inventory_analysis['recommendations']:
                    st.write(f"- {rec}")
            st.write("### Customer Lifetime Value")
            clv_analysis = bi_results['clv']
            if clv_analysis:
                st.write("CLV Metrics")
                st.dataframe(clv_analysis['metrics'])
                st.write("Customer Segments")
                st.dataframe(clv_analysis['segments'])
            # AI summary at the bottom
            st.markdown("---")
            st.subheader("AI BI Summary")
            st.markdown(bi_agent.summarize())
        
        with tab4:
            st.subheader("Sales Forecasting")
            
            # Deterministic outputs (as before)
            if forecast_results:
                st.write("### Sales Forecast")
                fig = forecast_agent.analyzer.generate_forecast_visualization('prophet')
                st.plotly_chart(fig, use_container_width=True)
                st.write("### Model Performance")
                performance = forecast_results['model_performance']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"${performance['mae']:,.2f}")
                with col2:
                    st.metric("Root Mean Square Error", f"${performance['rmse']:,.2f}")
                st.write("### Forecast Components")
                if 'components' in forecast_results:
                    components_df = pd.DataFrame({
                        'date': forecast_results['historical']['ds'],
                        'trend': forecast_results['components']['trend'][:len(forecast_results['historical'])],
                        'weekly': forecast_results['components']['weekly'][:len(forecast_results['historical'])]
                    })
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.line(components_df, x='date', y='trend', title="Trend Component")
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.line(components_df, x='date', y='weekly', title="Weekly Seasonality")
                        st.plotly_chart(fig, use_container_width=True)
            # AI summary at the bottom
            st.markdown("---")
            st.subheader("AI Forecast Summary")
            st.markdown(forecast_agent.summarize())
        
        with tab5:
            st.subheader("Data Visualizations")
            # Sales-specific visualizations
            st.write("### Sales Visualizations")
            for title, fig in sales_viz_results:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.subheader("AI Sales Visualization Summary")
            st.markdown(sales_viz_agent.summarize())
            # Advanced visualizations
            st.write("### Advanced Analysis")
            for title, fig in viz_results:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.subheader("AI Advanced Visualization Summary")
            st.markdown(viz_agent.summarize())
        
        # Prepare agent summaries for AIAnalysisAgent
        agent_summaries = {
            'sales': sales_agent.summary,
            'bi': bi_agent.summary,
            'forecast': forecast_agent.summary,
            'visualization': viz_agent.summary
        }

        with tab6:
            st.subheader("AI Analysis")
            ai_analysis_agent = AIAnalysisAgent(agent_summaries)
            st.markdown(ai_analysis_agent.synthesize_report())
        
        with tab7:
            st.subheader("💬 Chat with Your Data")
            # Initialize chat history and loading state in session state
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'chat_loading' not in st.session_state:
                st.session_state.chat_loading = False
            if 'chat_input_value' not in st.session_state:
                st.session_state.chat_input_value = ''
            import datetime
            # --- Chat UI/UX Styles ---
            chat_container_style = """
                <style>
                .chat-history-container {
                    max-height: 400px;
                    overflow-y: auto;
                    padding: 1em;
                    background: #f4f6fa;
                    border-radius: 12px;
                    margin-bottom: 1em;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    display: flex;
                    flex-direction: column;
                }
                .chat-bubble-user {
                    background: #2563eb;
                    color: white;
                    padding: 0.8em 1.2em;
                    border-radius: 20px 20px 6px 20px;
                    margin-bottom: 1.2em;
                    max-width: 80%;
                    margin-left: auto;
                    text-align: right;
                    box-shadow: 0 2px 8px rgba(37,99,235,0.08);
                    position: relative;
                }
                .chat-bubble-ai {
                    background: #fff;
                    color: #23272f;
                    padding: 0.8em 1.2em;
                    border-radius: 20px 20px 20px 6px;
                    margin-bottom: 1.2em;
                    max-width: 80%;
                    margin-right: auto;
                    text-align: left;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                    position: relative;
                }
                .chat-timestamp {
                    font-size: 0.75em;
                    color: #a1a1aa;
                    margin-bottom: 0.2em;
                }
                .chat-avatar {
                    width: 28px;
                    height: 28px;
                    border-radius: 50%;
                    display: inline-block;
                    vertical-align: middle;
                    margin-right: 0.5em;
                    margin-left: 0.5em;
                }
                .chat-row {
                    display: flex;
                    align-items: flex-end;
                }
                @media (max-width: 600px) {
                    .chat-history-container { max-height: 250px; padding: 0.5em; }
                    .chat-bubble-user, .chat-bubble-ai { max-width: 95%; font-size: 0.95em; }
                }
                </style>
            """
            st.markdown(chat_container_style, unsafe_allow_html=True)
            # --- Chat History (auto-scroll) ---
            chat_html = '''<div class="chat-history-container" id="chat-history">'''
            for msg in st.session_state.chat_history:
                # User row
                chat_html += '<div class="chat-row" style="justify-content: flex-end;">'
                chat_html += '<span class="chat-timestamp" style="text-align:right;">{}</span>'.format(msg["timestamp"])
                chat_html += '<span class="chat-bubble-user">'.format(msg["timestamp"])
                chat_html += '<img class="chat-avatar" src="https://img.icons8.com/ios-filled/50/ffffff/user-male-circle.png" style="background:#2563eb;">'
                chat_html += msg["user"] + '</span></div>'
                # AI row
                chat_html += '<div class="chat-row" style="justify-content: flex-start;">'
                chat_html += '<span class="chat-bubble-ai">'
                chat_html += '<img class="chat-avatar" src="https://img.icons8.com/ios-filled/50/23272f/robot-2.png" style="background:#e0e7ef;">'
                chat_html += msg["ai"] + '</span>'
                chat_html += '<span class="chat-timestamp" style="text-align:left;">{}</span></div>'.format(msg["timestamp"])
            chat_html += '</div>'
            # Auto-scroll to bottom using JS
            chat_html += '''<script>
                var chatHistory = document.getElementById('chat-history');
                if(chatHistory){ chatHistory.scrollTop = chatHistory.scrollHeight; }
            </script>'''
            st.markdown(chat_html, unsafe_allow_html=True)
            # --- Loading Indicator ---
            if st.session_state.chat_loading:
                st.info("AI is typing...")
            # --- Chat Input (multi-line, Send button below) ---
            user_input = st.text_area(
                "Ask a question about your data:",
                value=st.session_state.chat_input_value,
                key="data_chat_input",
                height=80,
                max_chars=500,
                placeholder="Type your question and press Send...",
                disabled=st.session_state.chat_loading
            )
            send_clicked = st.button("Send", key="data_chat_send", disabled=st.session_state.chat_loading)
            # Handle sending
            if send_clicked and user_input.strip():
                st.session_state.chat_loading = True
                # Gather context from all previous analysis
                sales_insights = sales_agent.generate_sales_insights()
                bi_kpis = bi_agent.track_kpis()
                territory_analysis = bi_agent.analyze_territories()
                competitor_analysis = bi_agent.analyze_competitors()
                inventory_analysis = bi_agent.optimize_inventory()
                clv_analysis = bi_agent.calculate_customer_lifetime_value()
                forecast_results = forecast_agent.forecast_with_prophet(14)
                # Build context string
                context = f"""
DATA CONTEXT:
Key Metrics: {bi_kpis}
Sales Insights: {sales_insights}
Territory Analysis: {territory_analysis}
Competitor Analysis: {competitor_analysis}
Inventory Analysis: {inventory_analysis}
Customer Lifetime Value: {clv_analysis}
Forecast (next 14 days): {forecast_results['forecast']['yhat'][-14:].tolist() if forecast_results else 'N/A'}
"""
                # Build chat history string
                chat_history_str = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.chat_history])
                # Compose prompt
                prompt = f"""
You are a helpful business data assistant. Use the following data context and chat history to answer the user's question conversationally, referencing the data and analysis where relevant. Be clear, concise, and business-focused.

{context}

CHAT HISTORY:
{chat_history_str}

USER QUESTION:
{user_input}

AI RESPONSE (be conversational, reference data, and provide actionable insights if possible):
"""
                ai_response = generate_text(prompt)
                # Add to chat history with timestamp
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                st.session_state.chat_history.append({
                    "user": user_input,
                    "ai": ai_response,
                    "timestamp": now
                })
                st.session_state.chat_loading = False
                # Clear input by updating session state
                st.session_state.chat_input_value = ''
            else:
                # Keep the input value in session state
                st.session_state.chat_input_value = user_input

if __name__ == "__main__":
    main() 