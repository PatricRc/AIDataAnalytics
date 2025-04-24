# Arroyo Job Costing Analytics

This Streamlit application provides advanced analytics for Arroyo's job costing data using Google's Gemini AI.

## Features

- **Interactive Dashboard**: Visualize key performance indicators and trends with interactive charts and tables
- **AI Report Generator**: Generate comprehensive reports with insights derived from your data
- **AI Chatbot**: Ask questions about your job costing data and get AI-powered responses

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   You can get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
4. Run the application:
   ```
   streamlit run app.py
   ```

## Data Requirements

The application uses the `JobCosting.xlsx` file in the root directory. This file should contain job costing data with the following structure:
- Job Number
- Department Names
- Job Type
- Job Status
- Created Date
- Completed Date
- Various cost and revenue metrics

## Technology Stack

- Streamlit for the web application framework
- Google Gemini 2.0 Flash for AI capabilities
- Pandas for data processing
- Plotly and Altair for data visualization

## Contact

For questions or support, please contact the Arroyo team. 