# Arroyo Job Costing Analytics Documentation

## Introduction

Arroyo Job Costing Analytics is a Streamlit-based web application that provides advanced analytics for job costing data using Google's Gemini AI. The application features an interactive dashboard, AI-powered reporting, and a chatbot for querying the job costing data.

## Table of Contents

1. [Application Overview](#application-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Application Pages](#application-pages)
   - [Dashboard](#dashboard)
   - [Advanced Analytics](#advanced-analytics)
   - [AI Report Generator](#ai-report-generator)
   - [AI Chatbot](#ai-chatbot)
4. [Technical Architecture](#technical-architecture)
5. [Data Processing](#data-processing)
6. [Gemini AI Integration](#gemini-ai-integration)
7. [RAG Implementation](#rag-implementation)
8. [Troubleshooting](#troubleshooting)
9. [Further Development](#further-development)

## Application Overview

The application analyzes job costing data from an Excel file and provides visualizations, insights, and AI-powered analysis to help understand business performance. It uses Google's Gemini 2.0 Flash model for advanced AI capabilities, including natural language understanding, report generation, and answering questions about the data.

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- Google API key for Gemini (can be obtained from [Google AI Studio](https://makersuite.google.com/app/apikey))
- JobCosting.xlsx file with job costing data

### Installation Steps

1. Clone the repository or download the application files
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
4. Ensure the JobCosting.xlsx file is in the ArroyoADO directory
5. Run the application using the provided script:
   ```
   python run.py
   ```
   Alternatively, run the Streamlit app directly:
   ```
   streamlit run app.py
   ```

## Application Pages

### Dashboard

The dashboard page provides a comprehensive overview of the job costing data, including:

- KPI cards showing key metrics (total jobs, revenue, profit margin)
- Performance analysis charts by department and job type
- Time series analysis of jobs and revenue
- Interactive data table with filtering capabilities

### Advanced Analytics

The advanced analytics page leverages Gemini AI to provide deeper insights into the job costing data:

- **Profit Drivers Analysis**: AI-powered analysis of what factors are driving profitability
- **Performance Forecast**: AI-generated forecast of future performance based on historical data
- **Business Recommendations**: AI-generated recommendations for improving business performance
- **Anomaly Detection**: Identification of outliers in the data and possible explanations

### AI Report Generator

The AI report generator allows users to create comprehensive reports with insights derived from the job costing data:

- Select from predefined report types or create a custom query
- Generate reports focused on specific departments or aspects of the business
- Download reports for sharing with stakeholders

### AI Chatbot

The AI chatbot provides a conversational interface for asking questions about the job costing data:

- Uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on the data
- Maintains conversation history for context
- Answers questions about profitability, job performance, trends, and more

## Technical Architecture

The application is built using the following technologies:

- **Streamlit**: Web application framework for creating the user interface
- **Pandas**: Data processing and analysis
- **Plotly**: Interactive data visualizations
- **Google Generative AI (Gemini)**: AI model for natural language understanding and generation
- **LangChain**: Framework for building applications with large language models
- **FAISS**: Vector store for efficient similarity search for RAG implementation

The application follows a modular architecture with the following components:

- **app.py**: Main application entry point and UI definition
- **utils.py**: Utility functions for data processing and visualization
- **advanced_analytics.py**: Functions for advanced AI-powered analytics
- **run.py**: Script for starting the application
- **test_app.py**: Unit tests for the application components

## Data Processing

The application processes job costing data from an Excel file with the following steps:

1. **Loading**: Reads the JobCosting.xlsx file into a pandas DataFrame
2. **Preprocessing**: Converts date columns to datetime format, creates clean Department and Location columns
3. **Filtering**: Allows filtering by date range, department, job type, and job status
4. **Aggregation**: Calculates aggregated metrics for visualization and analysis

## Gemini AI Integration

The application integrates with Google's Gemini 2.0 Flash model for AI capabilities:

- Uses the `google-generativeai` library for API communication
- Implements prompt engineering for specific tasks like report generation and recommendations
- Handles API responses and formats them for display in the application

## RAG Implementation

The application uses Retrieval-Augmented Generation (RAG) to improve the AI's responses:

1. **Vector Store Creation**: Converts job costing data into embeddings and stores them in a FAISS vector store
2. **Query Processing**: When a user asks a question, the application searches the vector store for relevant data
3. **Context-Enhanced Generation**: The retrieved data is used as context for the Gemini model, enabling it to provide more accurate and data-specific responses

## Troubleshooting

### Common Issues

- **API Key Errors**: Ensure your Google API key is correctly set in the .env file
- **Data Loading Errors**: Verify that the JobCosting.xlsx file is in the correct location and has the expected sheet names
- **Memory Issues**: If processing large datasets, consider running the application on a machine with more memory

### Testing

The application includes a test script (`test_app.py`) that can be used to verify that all components are working correctly:

```
python test_app.py
```

## Further Development

Potential areas for further development include:

- **User Authentication**: Add user login and role-based access control
- **Data Import/Export**: Allow users to upload their own job costing data
- **Custom Visualizations**: Enable users to create and save custom charts
- **Scheduled Reports**: Implement automatic report generation and email delivery
- **Integration with Other Systems**: Connect to ERP or accounting systems for real-time data 