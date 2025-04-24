import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from utils import get_gemini_model

def analyze_profit_drivers(data):
    """
    Use Gemini AI to analyze the key profit drivers in the dataset.
    """
    model = get_gemini_model()
    
    # Calculate some statistical summaries
    if 'Actual Profit' in data.columns and 'Job Type' in data.columns:
        profit_by_job_type = data.groupby('Job Type')['Actual Profit'].mean().to_dict()
    else:
        profit_by_job_type = {}
    
    if 'Actual Profit' in data.columns and 'Department' in data.columns:
        profit_by_dept = data.groupby('Department')['Actual Profit'].mean().to_dict()
    else:
        profit_by_dept = {}
    
    # Prepare the prompt
    prompt = f"""
    You are a financial analyst expert in job costing. Analyze the following data about profit metrics and identify key profit drivers:
    
    Average Profit by Job Type:
    {profit_by_job_type}
    
    Average Profit by Department:
    {profit_by_dept}
    
    Provide three key insights about what drives profit in this business based on this data. 
    Format your response with bullet points. Be specific and focus on actionable insights.
    """
    
    # Get response from Gemini
    response = model.generate_content(prompt)
    
    return response.text

def forecast_future_performance(data, months_ahead=3):
    """
    Use Gemini AI to forecast future performance based on historical data.
    """
    model = get_gemini_model()
    
    # Extract time series data if available
    if 'Created Date.1' in data.columns and 'Actual Revenue' in data.columns:
        # Convert to datetime if not already
        data['Created Date'] = pd.to_datetime(data['Created Date.1'])
        data['Month'] = data['Created Date'].dt.to_period('M').astype(str)
        
        # Calculate monthly metrics
        monthly_revenue = data.groupby('Month')['Actual Revenue'].sum().to_dict()
        
        if 'Actual Profit' in data.columns:
            monthly_profit = data.groupby('Month')['Actual Profit'].sum().to_dict()
        else:
            monthly_profit = {}
            
        # Count jobs per month
        monthly_jobs = data.groupby('Month').size().to_dict()
    else:
        monthly_revenue = {}
        monthly_profit = {}
        monthly_jobs = {}
    
    # Create the prompt for Gemini
    prompt = f"""
    You are a financial forecasting expert. Based on the following historical monthly data, forecast the expected performance for the next {months_ahead} months:
    
    Monthly Revenue:
    {monthly_revenue}
    
    Monthly Profit:
    {monthly_profit}
    
    Jobs per Month:
    {monthly_jobs}
    
    Provide a forecast for the next {months_ahead} months with estimated values for:
    1. Monthly Revenue
    2. Monthly Profit
    3. Number of Jobs
    
    Format your response as a forecast table with these three metrics for each of the next {months_ahead} months. 
    After the table, provide a brief explanation of your forecast and any trends or seasonality you've identified.
    """
    
    # Get response from Gemini
    response = model.generate_content(prompt)
    
    return response.text

def generate_recommendations(data, focus_area="profitability"):
    """
    Generate business recommendations based on the data using Gemini AI.
    """
    model = get_gemini_model()
    
    # Prepare different prompts based on the focus area
    if focus_area == "profitability":
        if 'Actual Margin' in data.columns and 'Job Type' in data.columns:
            margin_by_job_type = data.groupby('Job Type')['Actual Margin'].mean().to_dict()
            lowest_margin_jobs = sorted(margin_by_job_type.items(), key=lambda x: x[1])[:3]
            highest_margin_jobs = sorted(margin_by_job_type.items(), key=lambda x: x[1], reverse=True)[:3]
        else:
            lowest_margin_jobs = []
            highest_margin_jobs = []
        
        prompt = f"""
        You are a business consultant specializing in profitability improvement. 
        Based on the following job costing data, provide specific recommendations to improve profitability:
        
        Lowest Margin Job Types:
        {lowest_margin_jobs}
        
        Highest Margin Job Types:
        {highest_margin_jobs}
        
        Provide 3-5 actionable recommendations that would help improve overall profitability. 
        For each recommendation, explain the rationale and the expected impact. 
        Format your response with clear headings and bullet points.
        """
    
    elif focus_area == "efficiency":
        if 'Created Date.1' in data.columns and 'Completed Date.1' in data.columns:
            data['Completion Days'] = (data['Completed Date.1'] - data['Created Date.1']).dt.days
            avg_completion_time = data['Completion Days'].mean()
            completion_by_job_type = data.groupby('Job Type')['Completion Days'].mean().to_dict() if 'Job Type' in data.columns else {}
        else:
            avg_completion_time = "N/A"
            completion_by_job_type = {}
        
        prompt = f"""
        You are a business efficiency consultant. Based on the following job completion data, 
        provide specific recommendations to improve operational efficiency:
        
        Average Job Completion Time: {avg_completion_time} days
        
        Completion Time by Job Type:
        {completion_by_job_type}
        
        Provide 3-5 actionable recommendations that would help reduce job completion time and improve operational efficiency.
        For each recommendation, explain the rationale and the expected impact.
        Format your response with clear headings and bullet points.
        """
    
    else:  # General recommendations
        prompt = f"""
        You are a business consultant for the construction and services industry. 
        Based on your expertise in job costing and financial analysis, provide general 
        recommendations for improving business performance.
        
        Provide 5 actionable recommendations that cover different aspects of the business:
        1. Financial performance
        2. Operational efficiency
        3. Resource allocation
        4. Customer satisfaction
        5. Growth opportunities
        
        For each recommendation, explain the rationale and the expected impact.
        Format your response with clear headings and bullet points.
        """
    
    # Get response from Gemini
    response = model.generate_content(prompt)
    
    return response.text

def analyze_anomalies(data):
    """
    Use Gemini AI to identify anomalies or outliers in the job costing data.
    """
    model = get_gemini_model()
    
    # Check for jobs with unusually high or low margins if the column exists
    if 'Actual Margin' in data.columns:
        avg_margin = data['Actual Margin'].mean()
        std_margin = data['Actual Margin'].std()
        
        # Identify outliers (using 2 standard deviations)
        high_margin_threshold = avg_margin + 2 * std_margin
        low_margin_threshold = avg_margin - 2 * std_margin
        
        high_margin_jobs = data[data['Actual Margin'] > high_margin_threshold]
        low_margin_jobs = data[data['Actual Margin'] < low_margin_threshold]
        
        # Convert to dictionaries for the prompt
        high_margin_dict = high_margin_jobs[['Job Number', 'Actual Margin']].set_index('Job Number').to_dict()['Actual Margin'] if not high_margin_jobs.empty else {}
        low_margin_dict = low_margin_jobs[['Job Number', 'Actual Margin']].set_index('Job Number').to_dict()['Actual Margin'] if not low_margin_jobs.empty else {}
    else:
        high_margin_dict = {}
        low_margin_dict = {}
        avg_margin = "N/A"
    
    # Check for unusually high revenue jobs
    if 'Actual Revenue' in data.columns:
        avg_revenue = data['Actual Revenue'].mean()
        std_revenue = data['Actual Revenue'].std()
        
        # Identify outliers (using 2 standard deviations)
        high_revenue_threshold = avg_revenue + 2 * std_revenue
        
        high_revenue_jobs = data[data['Actual Revenue'] > high_revenue_threshold]
        high_revenue_dict = high_revenue_jobs[['Job Number', 'Actual Revenue']].set_index('Job Number').to_dict()['Actual Revenue'] if not high_revenue_jobs.empty else {}
    else:
        high_revenue_dict = {}
        avg_revenue = "N/A"
    
    # Format values properly
    avg_margin_formatted = f"{avg_margin:.2%}" if isinstance(avg_margin, (int, float)) else avg_margin
    avg_revenue_formatted = f"${avg_revenue:,.2f}" if isinstance(avg_revenue, (int, float)) else avg_revenue
    
    # Prepare the prompt
    prompt = f"""
    You are a data analyst specializing in anomaly detection. Analyze the following outliers in job costing data and identify possible reasons for these anomalies:
    
    Average Profit Margin: {avg_margin_formatted}
    
    Jobs with Unusually High Margins:
    {high_margin_dict}
    
    Jobs with Unusually Low Margins:
    {low_margin_dict}
    
    Average Revenue: {avg_revenue_formatted}
    
    Jobs with Unusually High Revenue:
    {high_revenue_dict}
    
    Analyze these anomalies and provide:
    1. Possible reasons why these jobs deviate significantly from the average
    2. What can be learned from the high-performing outliers
    3. What might be causing the low-performing outliers
    4. Recommendations for handling similar jobs in the future
    
    Format your response with clear sections and bullet points.
    """
    
    # Get response from Gemini
    response = model.generate_content(prompt)
    
    return response.text

def display_advanced_analytics(data):
    """
    Display the advanced analytics page with Gemini AI insights.
    """
    st.title("Advanced Analytics with AI")
    st.markdown("Leverage AI to gain deeper insights into your job costing data.")
    st.markdown("---")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Profit Drivers", 
        "📈 Performance Forecast", 
        "💡 Business Recommendations",
        "⚠️ Anomaly Detection"
    ])
    
    with tab1:
        st.subheader("Key Profit Drivers Analysis")
        st.markdown("AI analysis of what factors are driving profitability in your business.")
        
        # Add a button to generate the analysis
        if st.button("Analyze Profit Drivers", key="profit_drivers_btn"):
            with st.spinner("Analyzing profit drivers..."):
                try:
                    analysis = analyze_profit_drivers(data)
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Error analyzing profit drivers: {e}")
    
    with tab2:
        st.subheader("Performance Forecast")
        st.markdown("AI-powered forecast of future performance based on historical trends.")
        
        # Add input for forecast horizon
        months = st.slider("Forecast Horizon (Months)", min_value=1, max_value=12, value=3)
        
        # Add a button to generate the forecast
        if st.button("Generate Forecast", key="forecast_btn"):
            with st.spinner("Generating performance forecast..."):
                try:
                    forecast = forecast_future_performance(data, months_ahead=months)
                    st.markdown(forecast)
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
    
    with tab3:
        st.subheader("Business Recommendations")
        st.markdown("AI-generated recommendations to improve your business performance.")
        
        # Add selection for recommendation focus
        focus = st.selectbox(
            "Focus Area", 
            ["Profitability", "Efficiency", "General Business Improvement"],
            key="recommendation_focus"
        )
        
        # Map the selection to the function parameter
        focus_map = {
            "Profitability": "profitability",
            "Efficiency": "efficiency",
            "General Business Improvement": "general"
        }
        
        # Add a button to generate recommendations
        if st.button("Generate Recommendations", key="recommendations_btn"):
            with st.spinner("Generating business recommendations..."):
                try:
                    recommendations = generate_recommendations(data, focus_area=focus_map[focus])
                    st.markdown(recommendations)
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
    
    with tab4:
        st.subheader("Anomaly Detection")
        st.markdown("AI-powered analysis of outliers and unusual patterns in your job costing data.")
        
        # Add a button to detect anomalies
        if st.button("Detect Anomalies", key="anomalies_btn"):
            with st.spinner("Analyzing anomalies..."):
                try:
                    anomalies = analyze_anomalies(data)
                    st.markdown(anomalies)
                except Exception as e:
                    st.error(f"Error detecting anomalies: {e}")

def fix_formatting_in_anomaly_analysis():
    """
    Helper function to fix string formatting in the analyze_anomalies function.
    """
    # This function would actually modify the code, but we'll handle this with a direct edit 