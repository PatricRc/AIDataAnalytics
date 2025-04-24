import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go  # Import Plotly graph objects
from dotenv import load_dotenv
from utils import (
    load_data, create_vectorstore, get_gemini_model, generate_report,
    get_chat_response, create_kpi_cards, create_department_charts,
    create_job_type_charts, create_time_series, create_data_table,
    generate_and_execute_code, get_direct_chat_response
)
from advanced_analytics import display_advanced_analytics
from datetime import datetime, timedelta
import plotly.express as px

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Job Costing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for a modern, minimalistic look with improved readability
st.markdown("""
<style>
    /* Main styles */
    .main {
        background-color: #121212;
        color: #f0f0f0;
    }
    
    /* Dashboard cards */
    div[data-testid="stMetric"] {
        background-color: #1E3A8A;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    div[data-testid="stMetric"] > div {
        justify-content: center;
        text-align: center;
    }
    
    div[data-testid="stMetric"] label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] data-testid="stMetricValue" {
        color: #ffffff !important;
    }
    
    div[data-testid="stMetric"] data-testid="stMetricDelta" {
        color: #8dd8fc !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Improve chart appearance */
    div[data-testid="stPlotlyChart"] {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    
    /* Button styles */
    .stButton button {
        border-radius: 8px;
        background-color: #2563eb;
        color: white;
        font-weight: 500;
        padding: 10px 20px;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Enhanced chatbot styling */
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        background-color: #1a1f2a;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Chat message container */
    .chat-message {
        padding: 8px 14px;
        border-radius: 10px 10px 0 0;
        margin-bottom: 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
        font-size: 15px;
        line-height: 1.4;
    }
    
    .user-message {
        background-color: #3b82f6;
        border-left: 4px solid #1d4ed8;
        margin-left: 12px;
        margin-right: 3px;
        color: #ffffff;
    }
    
    .bot-message {
        background-color: #374151;
        border-left: 4px solid #2563eb;
        margin-left: 3px;
        margin-right: 12px;
        color: #ffffff;
    }
    
    .user-message b, .bot-message b {
        color: #f0f0f0;
        font-size: 15px;
        margin-bottom: 3px;
        display: block;
    }
    
    /* Style for the message content displayed using st.text */
    .stTextArea, pre {
        background-color: rgba(26, 32, 44, 0.8) !important;
        border-radius: 0 0 10px 10px !important;
        border: none !important;
        margin-top: 0 !important;
        margin-bottom: 15px !important;
        padding: 8px 14px !important;
        color: #f0f0f0 !important;
        font-size: 14px !important;
    }
    
    /* For user content */
    .user-message + pre, .user-message + .stTextArea {
        border-left: 5px solid #1d4ed8 !important;
        margin-left: 20px !important;
        margin-right: 5px !important;
    }
    
    /* For bot content */
    .bot-message + pre, .bot-message + .stTextArea {
        border-left: 5px solid #2563eb !important;
        margin-left: 5px !important;
        margin-right: 20px !important;
    }
    
    /* Chat input styles */
    .chat-input {
        background-color: #374151;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    /* Chat header styles */
    .chat-header {
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 2px solid #4b5563;
    }
    
    .chat-header h1 {
        font-size: 32px;
        margin-bottom: 10px;
        color: #ffffff;
    }
    
    .chat-header p {
        font-size: 18px;
        color: #d1d5db;
        line-height: 1.4;
    }
    
    /* Text input styling - Improved for chatbot */
    div.stTextInput > div > div > input {
        background-color: #111827;
        color: #ffffff;
        border: 1px solid #374151;
        padding: 10px 12px;
        font-size: 14px;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    div.stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }
    
    div.stTextInput > div > div > input::placeholder {
        color: #6b7280;
    }
    
    /* Text input label - Improved for chatbot */
    div.stTextInput label {
        font-size: 14px;
        color: #e5e7eb;
        font-weight: 500;
        margin-bottom: 4px;
        letter-spacing: 0.01em;
    }
    
    /* Chat container specific improvements */
    #chat-container {
        border: none;
        scroll-padding: 10px;
    }
    
    /* Smaller margins for improved spacing */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    
    /* Cleaner button styling */
    .stButton > button {
        margin-top: 0.25rem;
        min-height: 2.25rem;
        font-size: 0.9rem;
        transition: all 0.15s ease;
    }
    
    /* Streamlit standard elements */
    .stMarkdown p {
        color: #d1d5db;
    }
    
    .stMarkdown a {
        color: #60a5fa;
    }
    
    /* Fix text colors inside metric elements */
    div[data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #8dd8fc !important;
    }
    
    /* Horizontal rule styling */
    hr {
        border-color: #4b5563;
    }
    
    /* Footer styling */
    .footer {
        color: #9ca3af;
        font-size: 14px;
        text-align: center;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Check if API key is set
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Please set your Google API key in the .env file")
    st.stop()

# Initialize session state
if 'data' not in st.session_state:
    try:
        st.session_state.data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    # Clean up any duplicates in existing chat history
    cleaned_chat_history = []
    seen_messages = set()
    for message in st.session_state.chat_history:
        # Create a unique message identifier using content and timestamp
        message_id = f"{message['role']}_{message['content']}_{message.get('timestamp', '')}"
        
        # Skip this message if we've already seen it
        if message_id in seen_messages:
            continue
            
        # Add this message ID to our tracking set
        seen_messages.add(message_id)
        cleaned_chat_history.append(message)
    
    # Replace with deduplicated history
    st.session_state.chat_history = cleaned_chat_history

if 'code_chat_history' not in st.session_state:
    st.session_state.code_chat_history = []
else:
    # Clean up any duplicates in existing code chat history
    cleaned_code_chat_history = []
    seen_messages = set()
    for message in st.session_state.code_chat_history:
        # Create a unique message identifier using content and timestamp
        message_id = f"{message['role']}_{message['content']}_{message.get('timestamp', '')}"
        
        # Skip this message if we've already seen it
        if message_id in seen_messages:
            continue
            
        # Add this message ID to our tracking set
        seen_messages.add(message_id)
        cleaned_code_chat_history.append(message)
    
    # Replace with deduplicated history
    st.session_state.code_chat_history = cleaned_code_chat_history

# Create sidebar navigation
st.sidebar.title("AI Data Analytics")
st.sidebar.image("https://img.icons8.com/color/96/null/combo-chart--v1.png", width=100)

# Navigation
page = st.sidebar.radio("Navigation", ["Dashboard", "Advanced Analytics", "AI Report Generator", "AI Chatbot", "Chat Coding"])

# Data filters in sidebar
st.sidebar.subheader("Data Filters")

# Year selection if date column exists
if 'Created Date.1' in st.session_state.data.columns:
    years = sorted(st.session_state.data['Created Date.1'].dt.year.unique().tolist(), reverse=True)
    if years:
        selected_year = st.sidebar.selectbox("Select a year", years)
        year_start = datetime(year=selected_year, month=1, day=1)
        year_end = datetime(year=selected_year, month=12, day=31)
        mask = (st.session_state.data['Created Date.1'] >= year_start) & (st.session_state.data['Created Date.1'] <= year_end)
        filtered_data = st.session_state.data[mask]
    else:
        filtered_data = st.session_state.data
else:
    filtered_data = st.session_state.data

# Color theme selection
color_theme = st.sidebar.selectbox("Select a color theme", ["blues", "greens", "reds", "purples", "oranges"])
theme_colors = {
    "blues": {"primary": "#1d4ed8", "secondary": "#3b82f6", "background": "#1E293B", "text": "#ffffff"},
    "greens": {"primary": "#15803d", "secondary": "#22c55e", "background": "#1E3A29", "text": "#ffffff"},
    "reds": {"primary": "#b91c1c", "secondary": "#ef4444", "background": "#3A1E1E", "text": "#ffffff"},
    "purples": {"primary": "#7e22ce", "secondary": "#a855f7", "background": "#2E1E3A", "text": "#ffffff"},
    "oranges": {"primary": "#c2410c", "secondary": "#f97316", "background": "#3A291E", "text": "#ffffff"}
}

# Apply selected theme
selected_theme = theme_colors[color_theme]

# Apply theme to page
st.markdown(f"""
<style>
    .main {{
        background-color: {selected_theme["background"]};
        color: {selected_theme["text"]};
    }}
    div[data-testid="stMetric"] {{
        background-color: {selected_theme["primary"]};
    }}
</style>
""", unsafe_allow_html=True)

# Department filter if available
if 'Department' in filtered_data.columns:
    departments = ['All'] + sorted(filtered_data['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments, key="sidebar_dept")
    
    if selected_dept != 'All':
        filtered_data = filtered_data[filtered_data['Department'] == selected_dept]

# Main content based on selected page
if page == "Dashboard":
    st.title("Job Costing Dashboard")
    st.markdown("---")
    
    # Calculate key metrics for KPI cards
    total_jobs = len(filtered_data)
    total_revenue_value = filtered_data['Actual Revenue'].sum() if 'Actual Revenue' in filtered_data.columns else 0
    total_profit = filtered_data['Actual Profit'].sum() if 'Actual Profit' in filtered_data.columns else 0
    profit_margin = (total_profit / total_revenue_value) if total_revenue_value > 0 else 0
    
    # Get average completion time if dates are available
    avg_completion_time = None
    if 'Created Date.1' in filtered_data.columns and 'Completed Date.1' in filtered_data.columns:
        filtered_data['Completion Time'] = (filtered_data['Completed Date.1'] - filtered_data['Created Date.1']).dt.days
        avg_completion_time = filtered_data['Completion Time'].mean()
    
    # Calculate metrics for the last 7 days for delta comparisons
    max_date = filtered_data['Created Date.1'].max() if 'Created Date.1' in filtered_data.columns else None
    
    if max_date is not None:
        seven_days_before = max_date - timedelta(days=7)
        
        # Last 7 days data
        last_7_days_data = filtered_data[filtered_data['Created Date.1'] > seven_days_before]
        
        # Earlier data (before last 7 days)
        earlier_data = filtered_data[filtered_data['Created Date.1'] <= seven_days_before]
        
        # Calculate metrics for both periods
        jobs_last_7_days = len(last_7_days_data)
        jobs_delta = jobs_last_7_days - (len(earlier_data) / max(1, (len(filtered_data) - jobs_last_7_days)) * jobs_last_7_days)
        jobs_delta_percent = (jobs_delta / max(1, len(filtered_data) - jobs_last_7_days)) * 100
        
        revenue_last_7_days = last_7_days_data['Actual Revenue'].sum() if 'Actual Revenue' in last_7_days_data.columns else 0
        revenue_earlier = earlier_data['Actual Revenue'].sum() if 'Actual Revenue' in earlier_data.columns else 0
        revenue_delta = revenue_last_7_days - revenue_earlier
        
        profit_last_7_days = last_7_days_data['Actual Profit'].sum() if 'Actual Profit' in last_7_days_data.columns else 0
        
        profit_margin_last_7_days = (profit_last_7_days / revenue_last_7_days) if revenue_last_7_days > 0 else 0
        profit_margin_earlier = (earlier_data['Actual Profit'].sum() / earlier_data['Actual Revenue'].sum()) if 'Actual Profit' in earlier_data.columns and 'Actual Revenue' in earlier_data.columns and earlier_data['Actual Revenue'].sum() > 0 else 0
        profit_margin_delta = profit_margin_last_7_days - profit_margin_earlier
        
        completion_time_last_7_days = last_7_days_data['Completion Time'].mean() if 'Completion Time' in last_7_days_data else None
        completion_time_earlier = earlier_data['Completion Time'].mean() if 'Completion Time' in earlier_data else None
        completion_time_delta = (completion_time_last_7_days - completion_time_earlier) if completion_time_last_7_days is not None and completion_time_earlier is not None else None
    else:
        jobs_delta_percent = None
        revenue_delta = None
        profit_margin_delta = None
        completion_time_delta = None
    
    # Display 4 KPI cards at the top in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Jobs",
            value=f"{total_jobs:,}",
            delta=f"{jobs_delta_percent:.1f}% last 7 days" if jobs_delta_percent is not None else None,
            delta_color="normal" if jobs_delta_percent is None or jobs_delta_percent >= 0 else "inverse"
        )
    
    with col2:
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue_value:,.2f}",
            delta=f"${revenue_delta:,.2f} last 7 days" if revenue_delta is not None else None,
            delta_color="normal" if revenue_delta is None or revenue_delta >= 0 else "inverse"
        )
    
    with col3:
        st.metric(
            label="Profit Margin",
            value=f"{profit_margin:.2%}",
            delta=f"{profit_margin_delta:.2%} last 7 days" if profit_margin_delta is not None else None,
            delta_color="normal" if profit_margin_delta is None or profit_margin_delta >= 0 else "inverse"
        )
    
    with col4:
        if avg_completion_time is not None:
            st.metric(
                label="Avg Completion Time",
                value=f"{avg_completion_time:.1f} days",
                delta=f"{completion_time_delta:.1f} days last 7 days" if completion_time_delta is not None else None,
                delta_color="inverse" if completion_time_delta is None or completion_time_delta <= 0 else "normal"
            )
        else:
            st.metric(
                label="Avg Completion Time",
                value="N/A",
                delta=None
            )
    
    # Add some space after the KPI cards
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create a 2-column layout for the charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Department Revenue Bar Chart with improved colors and visibility
        if 'Department' in filtered_data.columns and 'Actual Revenue' in filtered_data.columns:
            dept_revenue = filtered_data.groupby('Department')['Actual Revenue'].sum().reset_index()
            dept_revenue = dept_revenue.sort_values('Actual Revenue', ascending=False)
            
            # Get the current theme color for gradient
            primary_color = selected_theme["primary"]
            secondary_color = selected_theme["secondary"]
            
            fig1 = px.bar(
                dept_revenue,
                x='Department',
                y='Actual Revenue',
                title='Revenue by Department',
                color='Department',
                color_discrete_sequence=[primary_color, secondary_color],
                text=dept_revenue['Actual Revenue'].apply(lambda x: f"${x:,.0f}")
            )
            fig1.update_layout(
                xaxis_title='Department', 
                yaxis_title='Revenue ($)',
                plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
                font=dict(color='#FFFFFF', size=12),  # White font for better visibility
                title_font=dict(color='#FFFFFF', size=18),  # White title font
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.1,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.2)",
                    font=dict(color='#FFFFFF', size=10)
                ),
                height=450,
                xaxis=dict(showgrid=False),  # Remove gridlines
                yaxis=dict(showgrid=False)   # Remove gridlines
            )
            fig1.update_traces(
                textposition='outside',
                textfont=dict(size=12, color='#FFFFFF'),  # White text on bars
                marker=dict(line=dict(width=1, color='#333333'))
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col_right:
        # Create a donut chart for top 10 jobs by revenue (moved from below)
        if 'Job Number' in filtered_data.columns and 'Actual Revenue' in filtered_data.columns:
            # Get top 10 jobs by revenue
            top_jobs = filtered_data.sort_values('Actual Revenue', ascending=False).head(10)
            
            # Create a more descriptive job label that includes job number and revenue
            top_jobs['Job Label'] = top_jobs.apply(
                lambda row: f"Job #{row['Job Number']} - ${row['Actual Revenue']:,.0f}", 
                axis=1
            )
            
            # Create a color gradient based on the theme
            color_scale = []
            try:
                # Try to create a gradient from the theme colors
                base_color = primary_color.lstrip('#')
                r = int(base_color[0:2], 16)
                g = int(base_color[2:4], 16)
                b = int(base_color[4:6], 16)
                
                # Create a gradient of 10 colors
                for i in range(10):
                    opacity = 0.5 + (0.5 * i/9)  # Gradient from 50% to 100% opacity
                    color_scale.append(f"rgba({r}, {g}, {b}, {opacity})")
            except:
                # Fallback to a predefined color scale if there's an error
                color_scale = px.colors.sequential.Blues
            
            # Create donut chart
            fig3 = px.pie(
                top_jobs,
                values='Actual Revenue',
                names='Job Label',
                title='Top 10 Jobs by Revenue',
                hole=0.4,
                color_discrete_sequence=color_scale
            )
            
            fig3.update_layout(
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='#FFFFFF', size=12),
                title_font=dict(color='#FFFFFF', size=18),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=1.1,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="rgba(255,255,255,0.2)",
                    font=dict(color='#FFFFFF', size=10)
                ),
                height=450
            )
            
            # Add total revenue annotation in the center
            total_revenue = top_jobs['Actual Revenue'].sum()
            fig3.add_annotation(
                text=f"${total_revenue:,.0f}",
                x=0.5, y=0.5,
                font=dict(size=16, color='#FFFFFF'),
                showarrow=False
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Required data for Top 10 Jobs chart is not available.")
    
    # Add a new row for the Daily Profit Margin Line Chart (moved from above)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Daily Profit Margin Line Chart by Department
    if 'Department' in filtered_data.columns and 'Actual Margin' in filtered_data.columns and 'Created Date.1' in filtered_data.columns:
        # Make sure date is in datetime format
        filtered_data['Date'] = pd.to_datetime(filtered_data['Created Date.1']).dt.date
        
        # Group by Department and Date, calculate average margin for each day
        daily_margin_by_dept = filtered_data.groupby(['Department', 'Date'])['Actual Margin'].mean().reset_index()
        
        # Create line chart with theme-based colors
        fig2 = px.line(
            daily_margin_by_dept,
            x='Date',
            y='Actual Margin',
            color='Department',
            title='Daily Profit Margin Trends by Department',
            color_discrete_sequence=[primary_color, secondary_color],
            markers=True
        )
        
        fig2.update_layout(
            xaxis_title='Date',
            yaxis_title='Profit Margin (%)',
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            font=dict(color='#FFFFFF', size=12),  # White font for better visibility
            title_font=dict(color='#FFFFFF', size=18),  # White title font
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.1,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="rgba(255,255,255,0.2)",
                font=dict(color='#FFFFFF', size=10)
            ),
            height=450,
            yaxis=dict(
                tickformat='.0%',
                showgrid=False  # Remove gridlines
            ),
            xaxis=dict(
                showgrid=False  # Remove gridlines
            )
        )
        
        fig2.update_traces(
            line=dict(width=3),
            marker=dict(size=8, line=dict(width=2, color='#333333'))
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Required data for Daily Profit Margin chart is not available.")
    
    # Add a header for the interactive data table
    st.markdown("### Interactive Job Costing Data")
    st.markdown("Use the table below to explore and filter the entire dataset.")
    
    # Import slickgrid
    try:
        from streamlit_slickgrid import slickgrid
        from streamlit_slickgrid import Formatters, Filters, FieldType, OperatorType
        
        # Create a copy of the dataframe to modify for grid display
        df_for_grid = filtered_data.copy()
        
        # Convert all datetime columns to string format for JSON serialization
        date_columns = df_for_grid.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            df_for_grid[col] = df_for_grid[col].astype(str)
        
        # Replace NaN values with None (which will be converted to null in JSON)
        # This fixes the "Unexpected token 'N', ..."d Hours": NaN, "Tota"... is not valid JSON" error
        df_for_grid = df_for_grid.fillna(value=None)
        
        # Convert DataFrame to a list of dictionaries for slickgrid
        grid_data = df_for_grid.to_dict('records')
        
        # Define grid columns
        columns = []
        
        # Add columns from the DataFrame
        for col_name in filtered_data.columns:
            column_def = {
                "id": col_name,
                "name": col_name,
                "field": col_name,
                "sortable": True,
                "filterable": True,
                "minWidth": 100,
            }
            
            # Configure column types based on data
            if col_name in ['Actual Revenue', 'Actual Profit', 'Actual Cost']:
                column_def["type"] = FieldType.number
                column_def["formatter"] = Formatters.currency
                column_def["params"] = {"thousandSeparator": ",", "decimalSeparator": ".", "symbolPrefix": "$"}
            elif col_name == 'Actual Margin':
                column_def["type"] = FieldType.number
                column_def["formatter"] = Formatters.percentCompleteBar
                column_def["params"] = {"multiplier": 100}
            elif 'Date' in col_name:
                column_def["type"] = FieldType.date
                column_def["formatter"] = Formatters.dateIso
                column_def["filter"] = {"model": Filters.compoundDate}
            elif filtered_data[col_name].dtype == 'bool':
                column_def["type"] = FieldType.boolean
                column_def["formatter"] = Formatters.checkmarkMaterial
            else:
                column_def["type"] = FieldType.string
                
            columns.append(column_def)
        
        # Configure grid options
        options = {
            "enableFiltering": True,
            "enableSorting": True,
            "enableTextExport": True,
            "autoHeight": False,
            "autoResize": {
                "minHeight": 500,
            },
            "enablePagination": True,
            "pagination": {
                "pageSize": 25,
            },
            "enableColumnReorder": True,
            "enableCellNavigation": True,
            "frozenColumn": -1,  # -1 means no frozen columns
            "multiColumnSort": True,
        }
        
        # Display the grid
        slickgrid(
            grid_data,
            columns,
            options,
            key='job_costing_grid'
        )
    except ImportError:
        st.warning("streamlit-slickgrid is not installed. Using standard Streamlit data display instead.")
        st.dataframe(filtered_data, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating grid: {e}")
        # Fallback to standard dataframe display
        st.dataframe(filtered_data, use_container_width=True)
    
    # Add explanation section
    with st.expander("About This Dashboard"):
        st.markdown("""
        - **KPI Cards**: Shows key performance indicators including total jobs, revenue, profit margin, and average job completion time.
        - **Revenue by Department**: Visual distribution of revenue across departments.
        - **Profit Margin by Department**: Average profit margins for each department.
        - **Interactive Table**: Full dataset with sorting, filtering, and pagination capabilities.
        
        Use the filters in the sidebar to analyze data for specific time periods, departments, or other criteria.
        """)

elif page == "Advanced Analytics":
    display_advanced_analytics(filtered_data)

elif page == "AI Report Generator":
    st.title("AI Report Generator")
    st.markdown("Generate insights and reports from your job costing data using AI.")
    st.markdown("---")
    
    # Report configuration
    report_type = st.selectbox(
        "Report Type",
        ["Performance Analysis", "Profitability Analysis", "Job Type Analysis", "Department Analysis", "Overall Department Analysis", "Custom Query"]
    )
    
    # If Department Analysis is selected, show department selector
    if report_type == "Department Analysis" and 'Department' in filtered_data.columns:
        departments = sorted(filtered_data['Department'].unique().tolist())
        selected_department = st.selectbox("Select Department", departments)
    else:
        selected_department = None
    
    # Custom query input
    if report_type == "Custom Query":
        custom_query = st.text_area("Enter your specific query or analysis request:", height=100)
    else:
        # Predefined queries based on report type
        query_map = {
            "Performance Analysis": "Analyze the overall performance metrics, focusing on revenue, profit margins, and job completion rates.",
            "Profitability Analysis": "Analyze the profitability of jobs, identifying factors that contribute to higher profit margins.",
            "Job Type Analysis": "Compare the performance and profitability of different job types.",
            "Department Analysis": f"Analyze the performance of the {selected_department} department." if selected_department else "Analyze performance across departments.",
            "Overall Department Analysis": "Overall Department Analysis: Provide a comprehensive analysis comparing all departments, highlighting top performers, areas for improvement, and key trends across departments."
        }
        custom_query = query_map[report_type]
    
    # Display the query to the user
    st.markdown(f"**Query:** {custom_query}")
    
    # Generate report button
    if st.button("Generate Report", key="gen_report_btn"):
        with st.spinner("Generating AI Report..."):
            try:
                report = generate_report(
                    filtered_data,
                    custom_query,
                    sector=selected_department if report_type == "Department Analysis" else None
                )
                
                # Display the report
                st.markdown("## AI-Generated Report")
                st.markdown(report)
                
                # Format report title based on report type
                if report_type == "Custom Query":
                    report_title = "Custom Analysis Report"
                else:
                    report_title = f"{report_type}"
                
                # Add download button for the report as text
                st.download_button(
                    label="📄 Download as Text",
                    data=report,
                    file_name=f"arroyo_{report_type.lower().replace(' ', '_')}_report.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating report: {e}")

elif page == "AI Chatbot":
    # Chat header with improved spacing and alignment
    st.markdown("""
    <div style="margin: 0 0 8px 0; padding: 0;">
        <h1 style="margin: 0 0 5px 0; padding: 0; font-size: 28px;">✨ Job Costing AI Assistant 🤖</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Add clear chat button in a more compact layout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <div style="margin: 0; padding: 0;">
            💬 Ask questions about your job costing data and get AI-powered answers. Try questions about revenue, margins, department performance, or job types.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Compact separator
    st.markdown("<hr style='margin: 8px 0; border-color: #4b5563; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Remove the fixed height container styling completely
    st.markdown("""
    <style>
    /* Make sure no backgrounds or borders appear */
    .stMarkdown, 
    .stMarkdown > div,
    .element-container {
        background: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat history with improved spacing - no container wrapper
    if st.session_state.chat_history:
        seen_messages = set()  # Track messages we've already displayed
        for message in st.session_state.chat_history:
            # Create a unique message identifier using content and timestamp
            message_id = f"{message['role']}_{message['content']}_{message.get('timestamp', '')}"
            
            # Skip this message if we've already displayed it
            if message_id in seen_messages:
                continue
                
            # Add this message ID to our tracking set
            seen_messages.add(message_id)
            
            if message["role"] == "user":
                # Get user message content and escape any HTML
                content = message["content"]
                content = content.replace("<", "&lt;").replace(">", "&gt;")
                
                # Get timestamp (defaulting to empty if not available)
                timestamp = message.get("timestamp", "")
                
                # Display user message in a chat bubble with improved spacing
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
                        <div style="background-color: #3b82f6; color: white; border-radius: 16px 16px 0 16px; 
                                    padding: 10px 16px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                                <span style="font-weight: 600; font-size: 13px;">👤 You</span>
                                <span style="font-size: 11px; opacity: 0.7; margin-left: 8px;">{timestamp}</span>
                            </div>
                            <div style="line-height: 1.4;">{content}</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                # Get assistant message content and escape any HTML
                content = message["content"]
                content = content.replace("<", "&lt;").replace(">", "&gt;")
                
                # Replace newlines with HTML breaks for proper formatting
                content = content.replace("\n", "<br>")
                
                # Get timestamp (defaulting to empty if not available)
                timestamp = message.get("timestamp", "")
                
                # Display assistant message in a chat bubble with improved spacing
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                        <div style="background-color: #374151; color: white; border-radius: 16px 16px 16px 0; 
                                    padding: 10px 16px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                                <span style="font-weight: 600; font-size: 13px;">🤖 Assistant</span>
                                <span style="font-size: 11px; opacity: 0.7; margin-left: 8px;">{timestamp}</span>
                            </div>
                            <div style="line-height: 1.4;">{content}</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    else:
        # Show welcome message with minimal height
        st.markdown(
            """
            <div style="text-align: center; color: #9ca3af; margin: 0; padding: 10px 0;">
                <div style="font-size: 24px; margin-bottom: 2px;">👋</div>
                <h3 style="margin: 0 0 2px 0; padding: 0; font-size: 18px;">Welcome to the Job Costing AI Assistant!</h3>
                <p style="margin: 0 0 2px 0;">Ask me anything about your job costing data. For example:</p>
                <ul style="text-align: left; display: inline-block; margin: 0; padding-left: 20px; line-height: 1.3; max-width: 500px;">
                    <li>What's our highest performing department?</li>
                    <li>Show me jobs with profit margins above 30%</li>
                    <li>Compare revenue between different job types</li>
                    <li>Analyze trends in our completion rates</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Add a compact separator between chat history and input area
    st.markdown("<hr style='margin: 8px 0; border-color: #4b5563; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Chat input with improved styling and compact design
    st.markdown("<div style='background-color: #1e293b; padding: 12px; border-radius: 10px; margin-top: 8px;'>", unsafe_allow_html=True)
    
    user_question = st.text_input("🔍 Ask a question about your job costing data:", key="user_question", 
                                placeholder="e.g., 📊 What's our highest performing department?")
    
    # Center the button with improved styling
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        ask_button = st.button("🚀 Ask Question", key="ask_btn")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add JS to scroll to bottom of chat container after new messages
    st.markdown("""
    <script>
        // Scroll the chat container to the bottom
        function scrollToBottom() {
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        
        // Execute scroll on page load
        window.onload = function() {
            scrollToBottom();
        };
    </script>
    """, unsafe_allow_html=True)
    
    if ask_button and user_question:
        # Check if this exact question was just asked (avoid duplicates)
        is_duplicate = False
        if st.session_state.chat_history:
            # Check the last message in the history
            last_message = st.session_state.chat_history[-1]
            if last_message["role"] == "user" and last_message["content"] == user_question:
                is_duplicate = True
        
        # Only process if not a duplicate
        if not is_duplicate:
            # Add user message to chat history
            timestamp = datetime.now().strftime("%I:%M %p")  # Format: 10:30 AM
            st.session_state.chat_history.append({"role": "user", "content": user_question, "timestamp": timestamp})
        
        with st.spinner("Generating response..."):
            try:
                # Get AI response using the direct approach without embeddings
                response = get_direct_chat_response(user_question)
                
                # Add AI response to chat history with timestamp
                timestamp = datetime.now().strftime("%I:%M %p")
                st.session_state.chat_history.append({"role": "assistant", "content": response, "timestamp": timestamp})
                
                # Rerun to update the UI with the new messages
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")
                error_message = f"I'm sorry, I encountered an error: {str(e)}"
                
                # Add error message to chat history with timestamp
                timestamp = datetime.now().strftime("%I:%M %p")
                st.session_state.chat_history.append({"role": "assistant", "content": error_message, "timestamp": timestamp})
                st.rerun()

elif page == "Chat Coding":
    # Chat header with improved spacing and alignment
    st.markdown("""
    <div style="margin: 0 0 8px 0; padding: 0;">
        <h1 style="margin: 0 0 5px 0; padding: 0; font-size: 28px;">🧪 Code Execution Assistant 💻</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Add clear chat button in a more compact layout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
        <div style="margin: 0; padding: 0;">
            💬 Ask questions about your job costing data and get Python code that analyzes the data. See both the code and its results.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_code_chat"):
            st.session_state.code_chat_history = []
            st.rerun()
    
    # Compact separator
    st.markdown("<hr style='margin: 8px 0; border-color: #4b5563; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Remove the fixed height container styling completely
    st.markdown("""
    <style>
    /* Make sure no backgrounds or borders appear */
    .stMarkdown, 
    .stMarkdown > div,
    .element-container {
        background: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat history with improved spacing - no container wrapper
    if st.session_state.code_chat_history:
        seen_messages = set()  # Track messages we've already displayed
        for message in st.session_state.code_chat_history:
            # Create a unique message identifier using content and role
            message_id = f"{message['role']}_{message['content']}_{message.get('timestamp', '')}"
            
            # Skip this message if we've already displayed it
            if message_id in seen_messages:
                continue
                
            # Add this message ID to our tracking set
            seen_messages.add(message_id)
            
            if message["role"] == "user":
                # Get user message content and escape any HTML
                content = message["content"]
                content = content.replace("<", "&lt;").replace(">", "&gt;")
                
                # Get timestamp (defaulting to empty if not available)
                timestamp = message.get("timestamp", "")
                
                # Display user message in a chat bubble with improved spacing
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
                        <div style="background-color: #3b82f6; color: white; border-radius: 16px 16px 0 16px; 
                                    padding: 10px 16px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                                <span style="font-weight: 600; font-size: 13px;">👤 You</span>
                                <span style="font-size: 11px; opacity: 0.7; margin-left: 8px;">{timestamp}</span>
                            </div>
                            <div style="line-height: 1.4;">{content}</div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif message["role"] == "assistant_code":
                # Get generated code
                code = message["content"]
                
                # Get timestamp
                timestamp = message.get("timestamp", "")
                
                # Display code block with syntax highlighting
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 0px;">
                        <div style="background-color: #374151; color: white; border-radius: 16px 16px 16px 0; 
                                    padding: 10px 16px; width: 90%; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                                <span style="font-weight: 600; font-size: 13px;">💻 Generated Code</span>
                                <span style="font-size: 11px; opacity: 0.7; margin-left: 8px;">{timestamp}</span>
                            </div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Display the code with syntax highlighting
                st.code(code, language="python")
                
            elif message["role"] == "assistant_result":
                # Get result content
                result = message["content"]
                
                # Check if the result is a plotly figure
                if isinstance(result, go.Figure):
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                            <div style="background-color: #374151; color: white; border-radius: 16px 16px 16px 0; 
                                        padding: 10px 16px; width: 90%; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                                    <span style="font-weight: 600; font-size: 13px;">📊 Result</span>
                                </div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Display the plotly figure
                    st.plotly_chart(result, use_container_width=True)
                else:
                    # For text results or other types
                    text_result = str(result)
                    
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                            <div style="background-color: #374151; color: white; border-radius: 16px 16px 16px 0; 
                                        padding: 10px 16px; width: 90%; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                                    <span style="font-weight: 600; font-size: 13px;">📊 Result</span>
                                </div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Format the result text
                    formatted_result = text_result
                    
                    # Check if the result contains HTML with image data (from the base64 encoded plot)
                    if "<img src=\"data:image/png;base64," in formatted_result:
                        # This is a plot result with an embedded image, render it directly
                        st.markdown(formatted_result, unsafe_allow_html=True)
                    # Check for image data in the result
                    elif "<matplotlib.figure.Figure" in formatted_result or "Figure()" in formatted_result:
                        st.info("A plot was generated! The analysis of the plot is included in the text results.")
                        # Look for keywords in text that indicate the plot contains useful information
                        if any(keyword in formatted_result.lower() for keyword in ['trend', 'increase', 'decrease', 'fluctuation', 'pattern', 'spike', 'dip', 'variation']):
                            st.success("Plot analysis shows patterns or trends in your data. Check the detailed explanation below.")
                    
                    # Handle case where matplotlib failed to display
                    elif "ModuleNotFoundError: No module named 'matplotlib'" in formatted_result:
                        st.error("Matplotlib library is not available. Please try again.")
                        formatted_result = formatted_result.replace("ModuleNotFoundError: No module named 'matplotlib'", 
                                                                  "**Error**: Matplotlib library was not available. The system has been updated to fix this issue. Please try your query again.")
                    
                    # Handle case where plot appears to be working but image not shown
                    elif "plt.show()" in formatted_result and not ("<matplotlib.figure.Figure" in formatted_result) and not ("<img src=\"data:image/png;base64," in formatted_result):
                        st.warning("The plot code was executed but the visualization couldn't be displayed directly. The textual analysis should still be helpful.")
                    
                    # Check if it contains code blocks and render them properly
                    if "<img src=\"data:image/png;base64," not in formatted_result:
                        import re
                        code_blocks = re.findall(r'```python\s*(.*?)\s*```', formatted_result, re.DOTALL)
                        if code_blocks:
                            for code_block in code_blocks:
                                # Replace each code block with a properly formatted one
                                st.code(code_block, language="python")
                                # Remove the code block from formatted_result to avoid duplication
                                formatted_result = formatted_result.replace(f"```python\n{code_block}\n```", "")
                        
                        # Display any remaining text
                        if formatted_result.strip() and "<img src=\"data:image/png;base64," not in formatted_result:
                            st.markdown(formatted_result)
    else:
        # Show welcome message with minimal height
        st.markdown(
            """
            <div style="text-align: center; color: #9ca3af; margin: 0; padding: 10px 0;">
                <div style="font-size: 24px; margin-bottom: 2px;">👋</div>
                <h3 style="margin: 0 0 2px 0; padding: 0; font-size: 18px;">Welcome to the Code Execution Assistant!</h3>
                <p style="margin: 0 0 2px 0;">Ask questions about your data and get Python code that analyzes it. For example:</p>
                <ul style="text-align: left; display: inline-block; margin: 0; padding-left: 20px; line-height: 1.3; max-width: 500px;">
                    <li>Create a bar chart of revenue by department</li>
                    <li>Calculate the average profit margin by job type</li>
                    <li>Find the top 5 most profitable jobs</li>
                    <li>Show a time series of monthly job counts</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Add a compact separator between chat history and input area
    st.markdown("<hr style='margin: 8px 0; border-color: #4b5563; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Chat input with improved styling and compact design
    st.markdown("<div style='background-color: #1e293b; padding: 12px; border-radius: 10px; margin-top: 8px;'>", unsafe_allow_html=True)
    
    code_question = st.text_input("🔍 Ask a question and get Python code:", key="code_question", 
                                placeholder="e.g., Create a bar chart showing revenue by department")
    
    # Center the button with improved styling
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_button = st.button("🚀 Generate Code", key="generate_btn")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add JS to scroll to bottom of chat container after new messages
    st.markdown("""
    <script>
        // Scroll the chat container to the bottom
        function scrollToBottom() {
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        
        // Execute scroll on page load
        window.onload = function() {
            scrollToBottom();
        };
    </script>
    """, unsafe_allow_html=True)
    
    if generate_button and code_question:
        # Check if this exact question was just asked (avoid duplicates)
        is_duplicate = False
        if st.session_state.code_chat_history:
            # Check the last message in the history
            last_message = st.session_state.code_chat_history[-1]
            if last_message["role"] == "user" and last_message["content"] == code_question:
                is_duplicate = True
        
        # Only process if not a duplicate
        if not is_duplicate:
            # Add user message to chat history
            timestamp = datetime.now().strftime("%I:%M %p")  # Format: 10:30 AM
            st.session_state.code_chat_history.append({"role": "user", "content": code_question, "timestamp": timestamp})
        
        with st.spinner("Generating and executing code..."):
            try:
                # Generate and execute code using the updated function that uses Gemini's built-in code execution
                generated_code, execution_result, raw_response = generate_and_execute_code(code_question)
                
                # Check if there's an error in the execution result
                error_markers = [
                    "Error:", "An error occurred:", "cannot access local variable", 
                    "UnboundLocalError", "TypeError", "ValueError", "NameError",
                    "ModuleNotFoundError", "ImportError"
                ]
                
                # If the query contains plot/chart/visualization keywords and there's an error
                plot_related = any(word in code_question.lower() for word in ["plot", "chart", "graph", "visualiz", "visual"])
                contains_error = any(marker in execution_result for marker in error_markers)
                
                if plot_related and contains_error:
                    # Try to fix the common matplotlib error
                    fixed_code = generated_code.replace(
                        "except Exception as error:", 
                        "except Exception as e:"
                    ).replace(
                        "print(f\"An error occurred: {error}\")", 
                        "print(f\"An error occurred: {e}\")"
                    )
                    
                    if fixed_code != generated_code:
                        # Use the fixed code instead
                        generated_code = fixed_code
                        execution_result += "\n\nNote: The code has been automatically fixed to handle errors properly."
                
                # Add code to chat history
                timestamp = datetime.now().strftime("%I:%M %p")
                st.session_state.code_chat_history.append({"role": "assistant_code", "content": generated_code, "timestamp": timestamp})
                
                # Add result to chat history
                st.session_state.code_chat_history.append({"role": "assistant_result", "content": execution_result})
                
                # Rerun to update the UI with the new messages
                st.rerun()
            except Exception as e:
                st.error(f"Error generating or executing code: {e}")
                error_message = f"I'm sorry, I encountered an error: {str(e)}"
                
                # Add error message to chat history
                timestamp = datetime.now().strftime("%I:%M %p")
                st.session_state.code_chat_history.append({"role": "assistant_code", "content": "# Error occurred", "timestamp": timestamp})
                st.session_state.code_chat_history.append({"role": "assistant_result", "content": error_message})
                st.rerun()

# Footer
st.markdown("---")
st.markdown('<div class="footer">🏢 AI Data Analytics | 🚀 Powered by Gemini 2.0 Flash ✨</div>', unsafe_allow_html=True) 