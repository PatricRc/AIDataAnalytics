# 🚀 AI Job Costing Analytics

<div align="center">

![AI Analytics Banner](https://raw.githubusercontent.com/patric-richard/readme-assets/main/analytics-banner.png)

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Gemini_AI-886FBF?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/generative-ai-studio)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

## 📊 Overview

AI Costing Analytics transforms your construction and service industry data into actionable intelligence through an intuitive, AI-powered dashboard. Leveraging Google's Gemini AI, this application delivers advanced data visualization, automated report generation, and intelligent conversational analytics.

<div align="center">
  <img src="https://raw.githubusercontent.com/patric-richard/readme-assets/main/dashboard-preview.png" alt="Dashboard Preview" width="80%">
</div>

## ✨ Key Features

- **🔍 Multi-view Dashboard** - Interactive visualizations of KPIs, revenue metrics, and profit margins
- **🤖 AI Chatbot Assistant** - Ask questions about your job costing data in natural language
- **📝 Automated Report Generation** - Create comprehensive analysis reports with a single click
- **💻 Code Execution Assistant** - Generate and run custom Python analyses on your data
- **📈 Advanced Analytics** - AI-driven insights on profit drivers, performance forecasting, and anomaly detection
- **🎨 Customizable Themes** - Choose from multiple color themes to match your company branding

## 🛠️ Tech Stack

<div align="center">

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit with custom CSS |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib |
| **AI Engine** | Google Gemini 2.0 Flash |
| **Embeddings** | LangChain, Google AI Embeddings |
| **Data Tables** | Streamlit-SlickGrid |
| **Export** | FPDF, WeasyPrint |

</div>

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- Google API key for Gemini AI ([Get one here](https://makersuite.google.com/app/apikey))
- Job costing data in Excel format

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-analytics.git
   cd ai-analytics
   ```

2. **Set up your environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure your API key**
   ```bash
   # Create a .env file in the project root
   echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
   ```

4. **Add your data**
   - Place your `JobCosting.xlsx` file in the project root directory
   - Ensure it contains columns like Job Number, Department, Job Type, Job Status, Created Date, etc.

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

## 📱 User Guide

### Dashboard

The main dashboard provides a comprehensive overview of your job costing data with:

- **KPI Cards** - Track total jobs, revenue, profit margins, and completion times
- **Department Charts** - Visualize performance across different departments
- **Job Type Analysis** - Compare metrics across different types of jobs
- **Time Series Analysis** - Track trends and patterns over time
- **Interactive Data Table** - Sort, filter, and explore your complete dataset

<div align="center">
  <img src="https://raw.githubusercontent.com/patric-richard/readme-assets/main/features-collage.png" alt="Features Collage" width="90%">
</div>

### AI Assistant

The AI Chatbot lets you interact with your data through natural language:

- Ask about department performance
- Query profit margins on specific job types
- Identify trends in completion times
- Get recommendations for improving profitability

### Report Generator

Generate comprehensive reports with:

- Executive summaries
- Key performance indicators
- Trend analysis
- Strategic recommendations

## 🧠 How It Works

AI Analytics combines traditional data visualization with advanced AI:

1. **Data Processing** - Your job costing Excel file is loaded and preprocessed
2. **Visualization Layer** - Interactive charts and tables are generated using Plotly
3. **AI Layer** - Google's Gemini AI model analyzes the data and responds to queries
4. **RAG Implementation** - Retrieval-Augmented Generation enhances AI responses with relevant context

## 🔮 Future Roadmap

- [ ] **User Authentication** - Role-based access control
- [ ] **Data Integration** - Connect directly to accounting systems
- [ ] **Mobile Application** - Native mobile experience
- [ ] **Predictive Analytics** - AI-powered forecasting models
- [ ] **Multi-language Support** - Internationalization for global teams

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please file an issue in the GitHub repository or contact the AI Analytics team.

---

<div align="center">
  <p>Built with ❤️ by <a href="https://your-website.com">Your Name</a></p>
  <p>
    <a href="https://twitter.com/your-twitter">Twitter</a> •
    <a href="https://linkedin.com/in/your-linkedin">LinkedIn</a> •
    <a href="https://github.com/your-github">GitHub</a>
  </p>
</div> 