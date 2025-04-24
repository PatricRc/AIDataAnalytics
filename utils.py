import os
import re
import pandas as pd
import numpy as np
from google import generativeai as genai
from google.generativeai.types import GenerationConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from dotenv import load_dotenv
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics.pairwise import cosine_similarity
import time
import random

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
def get_gemini_model():
    """
    Get the Gemini model for text generation.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model

def get_embeddings_model():
    """
    Get the embeddings model for RAG.
    """
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_data():
    """
    Load and preprocess the JobCosting data.
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create path to the Excel file
    file_path = os.path.join(current_dir, 'JobCosting.xlsx')
    
    df = pd.read_excel(file_path, sheet_name='JobCosting')
    
    # Clean up date columns
    date_columns = [col for col in df.columns if 'Date' in col]
    for col in date_columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # Create a clean Department column
    if 'Department Names - Copy.1' in df.columns:
        df['Department'] = df['Department Names - Copy.1']
    elif 'Department Names' in df.columns:
        df['Department'] = df['Department Names'].str.split(' - ').str[0]
    
    # Create a clean Location column
    if 'Department Names - Copy.2' in df.columns:
        df['Location'] = df['Department Names - Copy.2']
    elif 'Department Names' in df.columns:
        df['Location'] = df['Department Names'].str.split(' - ').str[-1]
    
    return df

# Simple vector store class with rate limiting
class SimpleVectorStore:
    def __init__(self, documents, embeddings_model):
        self.documents = documents
        self.embeddings_model = embeddings_model
        self._doc_embeddings = None
        
    def _get_doc_embeddings(self):
        """Get embeddings for all documents with rate limiting"""
        if self._doc_embeddings is None:
            all_embeddings = []
            # Process documents in smaller batches to avoid rate limits
            batch_size = 20  # Process 20 documents at a time
            
            for i in range(0, len(self.documents), batch_size):
                batch = self.documents[i:i+batch_size]
                retry_count = 0
                max_retries = 5
                
                while retry_count < max_retries:
                    try:
                        # Create embeddings for this batch
                        batch_embeddings = []
                        for doc in batch:
                            # Add jitter to avoid synchronized requests
                            jitter = random.uniform(0.1, 0.5)
                            time.sleep(jitter)  # Add random delay between requests
                            embedding = self.embeddings_model.embed_query(doc.page_content)
                            batch_embeddings.append(embedding)
                        
                        all_embeddings.extend(batch_embeddings)
                        
                        # Add delay between batches to avoid rate limits
                        if i + batch_size < len(self.documents):
                            time.sleep(2)  # 2 seconds between batches
                        
                        break  # Successfully processed batch, exit retry loop
                        
                    except Exception as e:
                        retry_count += 1
                        if "RATE_LIMIT_EXCEEDED" in str(e) or "429" in str(e):
                            # Exponential backoff with jitter for rate limit errors
                            wait_time = (2 ** retry_count) + random.uniform(0, 1)
                            print(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}")
                            time.sleep(wait_time)
                        else:
                            # For other errors, wait a bit but not as long
                            wait_time = retry_count + random.uniform(0, 1)
                            print(f"Error: {e}. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}")
                            time.sleep(wait_time)
                            
                        if retry_count == max_retries:
                            # If we've reached max retries for this batch, log the error and continue
                            print(f"Failed to process batch after {max_retries} retries. Skipping batch.")
                            # Use zeros as placeholder embeddings for failed documents
                            # Use the dimension from the embeddings model or a default
                            embedding_dim = 768  # Default dimension
                            if all_embeddings:
                                embedding_dim = len(all_embeddings[0])
                            for _ in batch:
                                all_embeddings.append([0.0] * embedding_dim)
            
            self._doc_embeddings = all_embeddings
        
        return self._doc_embeddings
    
    def similarity_search(self, query, k=5):
        """Find most similar documents to query with rate limiting"""
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                # Get query embedding
                query_embedding = self.embeddings_model.embed_query(query)
                
                # Get document embeddings
                doc_embeddings = self._get_doc_embeddings()
                
                # Convert to numpy arrays
                query_embedding_np = np.array(query_embedding).reshape(1, -1)
                doc_embeddings_np = np.array(doc_embeddings)
                
                # Calculate similarities
                similarities = cosine_similarity(query_embedding_np, doc_embeddings_np)[0]
                
                # Get top-k document indices
                top_indices = np.argsort(similarities)[-k:][::-1]
                
                # Return the documents
                return [self.documents[i] for i in top_indices]
                
            except Exception as e:
                retry_count += 1
                if "RATE_LIMIT_EXCEEDED" in str(e) or "429" in str(e):
                    # Exponential backoff with jitter for rate limit errors
                    wait_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Rate limit exceeded in search. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    # For other errors, wait a bit but not as long
                    wait_time = retry_count + random.uniform(0, 1)
                    print(f"Error in search: {e}. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                
                if retry_count == max_retries:
                    # If we've reached max retries, raise the error
                    raise Exception(f"Failed to perform similarity search after {max_retries} retries: {e}")
                    
        # Shouldn't reach here, but just in case
        return []

def create_vectorstore(df):
    """
    Create a vector store from the dataframe for RAG.
    """
    documents = []
    
    # Convert each row to a document
    for i, row in df.iterrows():
        # Convert row to string representation
        content = " ".join([f"{col}: {row[col]}" for col in df.columns])
        metadata = {"job_number": row["Job Number"] if "Job Number" in row else i}
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Create embeddings
    embeddings = get_embeddings_model()
    
    # Create vector store using SimpleVectorStore
    vectorstore = SimpleVectorStore(documents, embeddings)
    
    return vectorstore

def generate_report(data, query, sector=None):
    """
    Generate a report using Gemini model based on the data and query.
    """
    model = get_gemini_model()
    
    if sector:
        filtered_data = data[data['Department'] == sector]
    else:
        filtered_data = data
    
    # Prepare summary statistics
    summary = {
        "total_jobs": len(filtered_data),
        "avg_profit_margin": filtered_data['Actual Margin'].mean() if 'Actual Margin' in filtered_data else "N/A",
        "total_revenue": filtered_data['Actual Revenue'].sum() if 'Actual Revenue' in filtered_data else "N/A",
        "avg_completion_time": "N/A"  # Will calculate if we have the right date columns
    }
    
    # Try to calculate average completion time
    if 'Created Date.1' in filtered_data and 'Completed Date.1' in filtered_data:
        filtered_data['Completion Time'] = (filtered_data['Completed Date.1'] - filtered_data['Created Date.1']).dt.days
        summary["avg_completion_time"] = filtered_data['Completion Time'].mean()
    
    # Format values properly
    avg_margin_formatted = f"{summary['avg_profit_margin']:.2%}" if isinstance(summary['avg_profit_margin'], (int, float)) else summary['avg_profit_margin']
    revenue_formatted = f"${summary['total_revenue']:,.2f}" if isinstance(summary['total_revenue'], (int, float)) else summary['total_revenue']
    completion_time_formatted = f"{summary['avg_completion_time']:.1f} days" if isinstance(summary['avg_completion_time'], (int, float)) else summary['avg_completion_time']
    
    # Check if this is a department comparison query (for Overall Department Analysis)
    is_department_comparison = "comparing all departments" in query.lower() or "overall department analysis" in query.lower()
    
    # Generate department-specific data if needed
    department_data = ""
    if is_department_comparison and 'Department' in data.columns:
        # Get list of departments
        departments = data['Department'].unique()
        
        # Create a summary table of department metrics
        department_data = "Department-Specific Metrics:\n\n"
        department_data += "| Department | # of Jobs | Total Revenue | Avg Revenue/Job | Avg Profit Margin | Avg Completion Time (Days) | Total Cost |\n"
        department_data += "|------------|-----------|---------------|----------------|-------------------|----------------------------|------------|\n"
        
        for dept in departments:
            dept_data = data[data['Department'] == dept]
            
            # Calculate metrics for this department
            dept_jobs = len(dept_data)
            dept_revenue = dept_data['Actual Revenue'].sum() if 'Actual Revenue' in dept_data else 0
            dept_revenue_per_job = dept_revenue / dept_jobs if dept_jobs > 0 else 0
            dept_margin = dept_data['Actual Margin'].mean() if 'Actual Margin' in dept_data else 0
            
            # Calculate completion time if possible
            dept_completion_time = "N/A"
            if 'Created Date.1' in dept_data and 'Completed Date.1' in dept_data:
                dept_data['Completion Time'] = (dept_data['Completed Date.1'] - dept_data['Created Date.1']).dt.days
                dept_completion_time = dept_data['Completion Time'].mean()
            
            # Calculate total cost if possible
            dept_cost = dept_data['Actual Cost'].sum() if 'Actual Cost' in dept_data else 0
            
            # Format values
            dept_revenue_formatted = f"${dept_revenue:,.2f}"
            dept_revenue_per_job_formatted = f"${dept_revenue_per_job:,.2f}"
            dept_margin_formatted = f"{dept_margin:.2%}" if isinstance(dept_margin, (int, float)) else dept_margin
            dept_completion_time_formatted = f"{dept_completion_time:.1f}" if isinstance(dept_completion_time, (int, float)) else dept_completion_time
            dept_cost_formatted = f"${dept_cost:,.2f}"
            
            # Add row to table
            department_data += f"| {dept} | {dept_jobs} | {dept_revenue_formatted} | {dept_revenue_per_job_formatted} | {dept_margin_formatted} | {dept_completion_time_formatted} | {dept_cost_formatted} |\n"
    
    # Prepare the prompt for the Gemini model
    prompt = f"""
    You are a financial analyst expert in job costing. Analyze the following job costing data summary and provide insights:
    
    Data Summary:
    - Total Jobs: {summary['total_jobs']}
    - Average Profit Margin: {avg_margin_formatted}
    - Total Revenue: {revenue_formatted}
    - Average Completion Time: {completion_time_formatted}
    
    Job Types: {', '.join(filtered_data['Job Type'].unique()) if 'Job Type' in filtered_data else 'N/A'}
    Departments: {', '.join(filtered_data['Department'].unique()) if 'Department' in filtered_data else 'N/A'}
    
    {department_data}
    
    Specific Query: {query}
    
    Provide a professional report with the following sections:
    1. Executive Summary
    2. Key Performance Indicators
    3. Trend Analysis
    4. Recommendations
    
    Focus on profitability, efficiency, and potential areas for improvement.
    """
    
    response = model.generate_content(prompt)
    
    # Extract text from response based on new API structure
    if hasattr(response, 'text'):
        response_text = response.text
    elif hasattr(response, 'parts'):
        response_text = response.parts[0].text
    else:
        response_text = "No response generated."
    
    # More aggressive cleaning of HTML tags and artifacts
    # First, remove all complete HTML tags
    response_text = re.sub(r'<[^>]*>', '', response_text)
    
    # Then remove any stray closing tags that might remain (like </div>)
    response_text = re.sub(r'</?\w+\s*/?>', '', response_text)
    
    # Remove any other HTML-like artifacts that might remain
    response_text = response_text.replace('</div>', '')
    response_text = response_text.replace('<div>', '')
    response_text = response_text.replace('</p>', '')
    response_text = response_text.replace('<p>', '')
    
    return response_text

def get_chat_response(vectorstore, query):
    """
    Get a response from the Gemini model with RAG enhancement.
    """
    # Set up retry parameters
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Search for relevant documents - increase from 5 to 15 for more comprehensive context
            results = vectorstore.similarity_search(query, k=15)
            context = "\n\n".join([doc.page_content for doc in results])
            
            # Get access to the global data for summary statistics
            try:
                df = st.session_state.data
                
                # Calculate dataset summary statistics
                dataset_summary = {
                    "total_records": len(df),
                    "departments": df['Department'].unique().tolist() if 'Department' in df.columns else [],
                    "job_types": df['Job Type'].unique().tolist() if 'Job Type' in df.columns else [],
                    "date_range": f"{df['Created Date.1'].min().date()} to {df['Created Date.1'].max().date()}" if 'Created Date.1' in df.columns else "Unknown",
                    "total_revenue": f"${df['Actual Revenue'].sum():,.2f}" if 'Actual Revenue' in df.columns else "Unknown",
                    "avg_margin": f"{df['Actual Margin'].mean():.2%}" if 'Actual Margin' in df.columns else "Unknown",
                }
                
                # Count jobs by department
                if 'Department' in df.columns:
                    dept_counts = df.groupby('Department').size().to_dict()
                    dept_summary = "\n".join([f"- {dept}: {count} jobs" for dept, count in dept_counts.items()])
                else:
                    dept_summary = "Department information not available"
                
                # Enhanced context with dataset summary
                enhanced_context = f"""
DATASET SUMMARY:
Total Records: {dataset_summary['total_records']}
Date Range: {dataset_summary['date_range']}
Total Revenue: {dataset_summary['total_revenue']}
Average Profit Margin: {dataset_summary['avg_margin']}

Available Departments:
{dept_summary}

Available Job Types:
{', '.join(dataset_summary['job_types']) if dataset_summary['job_types'] else 'Job type information not available'}

SPECIFIC DOCUMENT CONTEXT:
{context}
"""
            except:
                # If accessing session state data fails, fall back to just using the retrieved documents
                enhanced_context = context
            
            model = get_gemini_model()
            
            # Prepare the prompt for the Gemini model with enhanced context
            prompt = f"""
            You are a helpful assistant specializing in job costing and construction financial analysis.
            Use the following context to answer the question at the end. The context includes both a general dataset summary and specific relevant document snippets.
            
            Context:
            {enhanced_context}
            
            Question: {query}
            
            Provide a clear, concise, and precise answer based on the context. Include specific numbers and statistics when relevant.
            If the information is not in the context, say that you don't have enough information to answer accurately.
            
            CRITICALLY IMPORTANT: Write your response in PLAIN TEXT ONLY.
            DO NOT include HTML tags of any kind, including </div>, <div>, <p>, </p>, or any other markup.
            Your response should contain only text with no HTML formatting whatsoever.
            """
            
            # Add retry logic for the model.generate_content call
            generation_retry_count = 0
            while generation_retry_count < 3:  # Max 3 retries for generation
                try:
                    response = model.generate_content(prompt)
                    break  # Break out of the inner retry loop if successful
                except Exception as e:
                    generation_retry_count += 1
                    if "RATE_LIMIT_EXCEEDED" in str(e) or "429" in str(e):
                        # Exponential backoff with jitter for rate limit errors
                        wait_time = (2 ** generation_retry_count) + random.uniform(0, 1)
                        print(f"Rate limit exceeded in generate_content. Waiting {wait_time:.2f} seconds before retry {generation_retry_count}/3")
                        time.sleep(wait_time)
                    else:
                        # For other errors, shorter wait
                        time.sleep(1)
                    
                    # If we've reached max retries, let it fail and be caught by the outer try-except
                    if generation_retry_count == 3:
                        response = model.generate_content(prompt)  # This will raise if it fails
            
            # Extract text from response based on new API structure
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'parts'):
                response_text = response.parts[0].text
            else:
                response_text = "No response generated."
            
            # More aggressive cleaning of HTML tags and artifacts
            # First, remove all complete HTML tags
            response_text = re.sub(r'<[^>]*>', '', response_text)
            
            # Then remove any stray closing tags that might remain (like </div>)
            response_text = re.sub(r'</?\w+\s*/?>', '', response_text)
            
            # Remove any other HTML-like artifacts that might remain
            response_text = response_text.replace('</div>', '')
            response_text = response_text.replace('<div>', '')
            response_text = response_text.replace('</p>', '')
            response_text = response_text.replace('<p>', '')
            
            return response_text
            
        except Exception as e:
            retry_count += 1
            if "RATE_LIMIT_EXCEEDED" in str(e) or "429" in str(e):
                # Exponential backoff with jitter for rate limit errors
                wait_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"Rate limit exceeded in chat response. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            else:
                # For other errors, wait a bit but not as long
                wait_time = retry_count + random.uniform(0, 1)
                print(f"Error in chat response: {e}. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            
            if retry_count == max_retries:
                # If we've reached max retries, return an error message
                return f"I apologize, but I encountered an error while processing your request. The system is currently experiencing high demand. Please try again in a few moments. Error details: {str(e)}"
                
    # Shouldn't reach here due to the return in the exception handler, but just in case
    return "I apologize, but I encountered an error while processing your request. Please try again later."

def generate_and_execute_code(query):
    """
    Generate and execute Python code based on the user's query using the Gemini model's
    built-in code execution functionality.
    
    Args:
        query (str): The user's query about the data
        
    Returns:
        tuple: (generated_code, execution_result, raw_response)
    """
    try:
        # Make sure re is imported and available inside the function
        import re
        
        # Access the data
        df = st.session_state.data
        
        # Access the vectorstore for RAG instead of using raw CSV data
        vectorstore = st.session_state.vectorstore
        
        # Search for relevant documents to provide context
        rag_results = vectorstore.similarity_search(query, k=20)
        rag_context = "\n\n".join([doc.page_content for doc in rag_results])
        
        # Calculate dataset summary statistics for context
        dataset_summary = {
            "total_records": len(df),
            "departments": df['Department'].unique().tolist() if 'Department' in df.columns else [],
            "job_types": df['Job Type'].unique().tolist() if 'Job Type' in df.columns else [],
            "date_range": f"{df['Created Date.1'].min().date()} to {df['Created Date.1'].max().date()}" if 'Created Date.1' in df.columns else "Unknown",
            "total_revenue": f"${df['Actual Revenue'].sum():,.2f}" if 'Actual Revenue' in df.columns else "Unknown",
            "avg_margin": f"{df['Actual Margin'].mean():.2%}" if 'Actual Margin' in df.columns else "Unknown",
        }
        
        # Count jobs by department
        if 'Department' in df.columns:
            dept_counts = df.groupby('Department').size().to_dict()
            dept_summary = "\n".join([f"- {dept}: {count} jobs" for dept, count in dept_counts.items()])
        else:
            dept_summary = "Department information not available"
        
        # Set up the model with code execution capability
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
            ),
        )
        
        # Prepare the context message with dataset information from RAG
        context_message = f"""
        I have a pandas DataFrame with job costing data. Here's a summary of the dataset:
        
        DATASET SUMMARY:
        Total Records: {dataset_summary['total_records']}
        Date Range: {dataset_summary['date_range']}
        Total Revenue: {dataset_summary['total_revenue']}
        Average Profit Margin: {dataset_summary['avg_margin']}
        
        Available Departments:
        {dept_summary}
        
        Available Job Types:
        {', '.join(dataset_summary['job_types']) if dataset_summary['job_types'] else 'Job type information not available'}
        
        The DataFrame is available as the variable 'df' in your Python environment.
        Please analyze this data to answer my question: "{query}"
        
        Important guidelines:
        1. Use pandas and numpy for data analysis
        2. For matplotlib plots, you MUST START with these EXACT imports:
           ```python
           import pandas as pd
           import numpy as np
           import matplotlib.pyplot as plt
           import matplotlib
           matplotlib.use('Agg')  # Use non-interactive backend
           ```
        3. Then, when creating plots, follow this EXACT structure:
           ```python
           # Data preparation code here
           # YOUR CUSTOM DATA PREPARATION CODE GOES HERE - DON'T COPY THIS COMMENT
           
           # Creating the figure - this is REQUIRED
           plt.figure(figsize=(10, 6))
           
           # Create your plot here
           # YOUR CUSTOM PLOT CODE GOES HERE - DON'T COPY THIS COMMENT
           
           # Add titles, labels, etc.
           plt.title('Your Title', fontsize=16)
           plt.xlabel('X Label', fontsize=12)
           plt.ylabel('Y Label', fontsize=12)
           plt.grid(True)
           
           # ESSENTIAL for display:
           plt.tight_layout()
           plt.savefig('plot.png')  # CRITICAL: This line is required to make the plot visible to the user
           plt.show()
           ```
        4. CRITICAL: Never omit plt.figure(), plt.tight_layout(), plt.savefig('plot.png'), and plt.show()
           - plt.savefig('plot.png') is ABSOLUTELY REQUIRED for the visualization to be visible to the user
        5. Use vibrant colors (like 'blue', 'green', 'red', not light pastels)
        6. Always include markers in line plots with marker='o' parameter
        7. Always add a grid with plt.grid(True)
        8. Use descriptive titles and labels with appropriate fontsize
        9. Include proper error handling with try/except blocks
        10. Format percentages properly (use actual percentage values, not decimals)
        11. ONLY provide code in the code implementation - do not include code in your explanation
        12. After executing the code, provide a summary of the results labeled as "Results Analysis:"
        """
        
        # Add example code template for plotting
        context_message += """
        Here's an example template for creating a line chart for profit margin by date:
        
        ```python
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Set backend
        
        try:
            # First check if the necessary columns exist and handle missing values
            if 'Created Date.1' not in df.columns:
                raise ValueError("Date column 'Created Date.1' not found in the DataFrame")
            if 'Actual Margin' not in df.columns:
                raise ValueError("Profit Margin column 'Actual Margin' not found in the DataFrame")
                
            # Make a copy of the DataFrame to avoid modifying the original
            plot_df = df.copy()
            
            # Ensure date column is datetime
            plot_df['Date'] = pd.to_datetime(plot_df['Created Date.1'])
            
            # Group by date and calculate average profit margin
            daily_avg_profit = plot_df.groupby(plot_df['Date'].dt.date)['Actual Margin'].mean()
            
            # Create the figure
            plt.figure(figsize=(12, 6))
            
            # Create the line plot
            plt.plot(
                daily_avg_profit.index, 
                daily_avg_profit.values, 
                marker='o',
                color='blue',
                linewidth=2,
                markersize=6
            )
            
            # Add titles and labels
            plt.title('Average Daily Profit Margin', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Profit Margin (%)', fontsize=12)
            
            # Customize the plot
            plt.grid(True)
            plt.tight_layout()
            
            # Format y-axis as percentage
            from matplotlib.ticker import PercentFormatter
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            
            # CRITICAL: Save the figure as a PNG file for the user to see
            plt.savefig('plot.png')
            
            # Show the plot
            plt.show()
            
        except Exception as e:
            print(f"An error occurred: {e}")
        ```
        
        REMINDER: 
        - Adapt the code to the actual DataFrame column names, don't just copy and paste this template. 
        - The example uses 'Created Date.1' and 'Actual Margin', but your code should use the actual column names in the dataset.
        - ALWAYS include the plt.savefig('plot.png') line in your code - this is REQUIRED for the visualization to be visible to the user.
        
        Please generate and execute Python code to answer my question.
        """
        
        # Generate code with the model
        response = model.generate_content(context_message + "\n" + query)
        
        # Extract code and results
        generated_code = ""
        execution_result = ""
        explanation_text = ""
        
        # Process the response parts
        if hasattr(response, 'text'):
            explanation_text = response.text
        elif hasattr(response, 'parts'):
            explanation_text = response.parts[0].text
        
        # If no code was extracted but there's text content, try to extract code blocks from the text
        if explanation_text:
            # Try to find Python code blocks in markdown format
            code_blocks = re.findall(r'```python\s*(.*?)\s*```', explanation_text, re.DOTALL)
            if code_blocks:
                generated_code = code_blocks[0]
                # Remove the code blocks from explanation to avoid duplication
                for code_block in code_blocks:
                    explanation_text = explanation_text.replace(f"```python\n{code_block}\n```", "")
                    explanation_text = explanation_text.replace(f"```python\n{code_block}```", "")
            # If no python blocks, try generic code blocks
            elif not code_blocks:
                code_blocks = re.findall(r'```\s*(.*?)\s*```', explanation_text, re.DOTALL)
                if code_blocks:
                    generated_code = code_blocks[0]
                    # Remove the code blocks from explanation to avoid duplication
                    for code_block in code_blocks:
                        explanation_text = explanation_text.replace(f"```\n{code_block}\n```", "")
                        explanation_text = explanation_text.replace(f"```\n{code_block}```", "")
        
        # Clean the explanation text to remove any code implementation sections
        explanation_text = re.sub(r'Code Implementation:.*?(?=Results Analysis:|$)', '', explanation_text, flags=re.DOTALL | re.IGNORECASE)
        explanation_text = re.sub(r'Here\'s the code implementation:.*?(?=Results Analysis:|$)', '', explanation_text, flags=re.DOTALL | re.IGNORECASE)
        explanation_text = re.sub(r'Here\'s the code:.*?(?=Results Analysis:|$)', '', explanation_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Execute the generated code if it exists
        if generated_code:
            try:
                # Create a local namespace for code execution
                local_namespace = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'matplotlib': matplotlib}
                
                # Execute the generated code
                exec(generated_code, globals(), local_namespace)
                
                # Capture any output
                execution_result = "Code executed successfully."
                
                # If there's a Results Analysis section in the explanation, extract it
                result_section_match = re.search(r'Results Analysis:(.*?)(?=\n\n|$)', explanation_text, re.DOTALL | re.IGNORECASE)
                if result_section_match:
                    execution_result = result_section_match.group(1).strip()
            except Exception as e:
                execution_result = f"An error occurred during code execution: {str(e)}"
        
        # If there's no execution result but there is explanation text,
        # try to extract the results section if it exists
        if not execution_result and explanation_text:
            result_section_match = re.search(r'Results Analysis:(.*?)(?=\n\n|$)', explanation_text, re.DOTALL | re.IGNORECASE)
            if result_section_match:
                execution_result = result_section_match.group(1).strip()
            else:
                # If no explicit Results Analysis section, use the explanation as the result
                execution_result = explanation_text.strip()
        
        # Check if the query is related to plotting/visualization
        is_plot_request = any(term in query.lower() for term in [
            'plot', 'chart', 'graph', 'visualiz', 'visual', 'figure', 'histogram', 
            'bar chart', 'line chart', 'scatter plot', 'pie chart', 'heatmap', 'show me'
        ])
        
        # If a plot was requested, check if a plot.png file was created
        if is_plot_request and os.path.exists('plot.png'):
            import base64
            
            # If it's a plot request and we successfully created a plot file
            # Read the saved plot image
            with open('plot.png', 'rb') as f:
                plot_data = f.read()
            
            # Encode the image data
            encoded_plot = base64.b64encode(plot_data).decode('utf-8')
            
            # Create HTML to display the image
            image_html = f'<img src="data:image/png;base64,{encoded_plot}" alt="Plot" style="width:100%;max-width:800px;margin:20px auto;display:block;">'
            
            # Extract or create a description of the plot
            if 'Results Analysis:' in execution_result:
                plot_description = re.search(r'Results Analysis:(.*?)(?=\n\n|$)', execution_result, re.DOTALL | re.IGNORECASE)
                if plot_description:
                    plot_description = plot_description.group(1).strip()
                else:
                    plot_description = "Analysis of the visualization."
            else:
                plot_description = execution_result if execution_result else "Analysis of the visualization."
            
            # Combine the image and description
            html_result = f"""
            <div style="text-align:center;margin:20px 0;">
                {image_html}
                <div style="margin-top:10px;text-align:left;">
                    <strong>Analysis:</strong> {plot_description}
                </div>
            </div>
            """
            
            # Return the generated code and the HTML result that includes the image
            return generated_code, html_result, response
        
        # Ensure we return the full explanation text as part of the execution result
        # but without code blocks to avoid duplication
        if explanation_text and not generated_code:
            return "# No executable code found", explanation_text, response
        elif explanation_text and not execution_result:
            return generated_code, explanation_text, response
        else:
            # Include both the explanation and actual execution result, ensuring no code duplication
            if execution_result not in explanation_text:
                clean_explanation = explanation_text.strip()
                if clean_explanation:
                    combined_result = f"{clean_explanation}\n\n{execution_result}"
                else:
                    combined_result = execution_result
            else:
                combined_result = explanation_text
            
            return generated_code, combined_result, response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"# Error: {str(e)}", f"An error occurred: {str(e)}\n\nDetails:\n{error_details}", None

# Helper function to display code execution results (for debugging)
def display_code_execution_result(response):
    """
    Display the code execution results from a Gemini API response.
    
    Args:
        response: The response from the Gemini API
    
    Returns:
        dict: A dictionary with 'code' and 'output' keys
    """
    result = {"code": "", "output": ""}
    
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                # If there's no code or output yet, store text as output
                if not result["output"]:
                    result["output"] += part.text + "\n"
            if hasattr(part, 'executable_code') and part.executable_code:
                result["code"] = part.executable_code.code
            if hasattr(part, 'code_execution_result') and part.code_execution_result:
                result["output"] = part.code_execution_result.output
    
    return result

def create_kpi_cards(df):
    """
    Create KPI metrics for the dashboard.
    """
    # Calculate KPIs
    total_jobs = len(df)
    closed_jobs = len(df[df['Job Status'] == 'Closed']) if 'Job Status' in df else 0
    completion_rate = closed_jobs / total_jobs if total_jobs > 0 else 0
    
    avg_profit_margin = df['Actual Margin'].mean() if 'Actual Margin' in df.columns else 0
    total_revenue = df['Actual Revenue'].sum() if 'Actual Revenue' in df.columns else 0
    total_profit = df['Actual Profit'].sum() if 'Actual Profit' in df.columns else 0
    
    # Calculate last 7 days metrics for delta
    if 'Created Date.1' in df.columns:
        # Find the max date in the dataset instead of using today's date
        max_date = df['Created Date.1'].max().date()
        seven_days_before_max = max_date - timedelta(days=7)
        
        # Filter data for last 7 days from max date
        last_7_days_data = df[(df['Created Date.1'].dt.date <= max_date) & 
                              (df['Created Date.1'].dt.date > seven_days_before_max)]
        
        # Calculate 7-day profit margin average
        last_7_days_profit_margin = last_7_days_data['Actual Margin'].mean() if len(last_7_days_data) > 0 and 'Actual Margin' in last_7_days_data.columns else 0
        
        # Calculate the delta compared to overall average
        profit_margin_delta = last_7_days_profit_margin - avg_profit_margin
    else:
        profit_margin_delta = None
    
    # Create a container for consistent styling
    metrics_container = st.container()
    
    # Create a 3-column layout for KPI cards
    with metrics_container:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Jobs",
                value=f"{total_jobs:,}",
                delta=f"{completion_rate:.1%} Completed",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                label="Total Revenue",
                value=f"${total_revenue:,.2f}",
                delta=f"${total_profit:,.2f} Profit",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                label="Avg Profit Margin",
                value=f"{avg_profit_margin:.2%}",
                delta=f"{profit_margin_delta:.2%} Last 7 Days" if profit_margin_delta is not None else None,
                delta_color="normal" if profit_margin_delta is None else ("off" if abs(profit_margin_delta) < 0.0001 else ("normal" if profit_margin_delta > 0 else "inverse"))
            )

def create_department_charts(df):
    """
    Create department-based charts for the dashboard.
    """
    if 'Department' not in df.columns or 'Actual Revenue' not in df.columns:
        st.warning("Required columns not found in the dataset.")
        return
    
    # Department Revenue Bar Chart
    dept_revenue = df.groupby('Department')['Actual Revenue'].sum().reset_index()
    dept_revenue = dept_revenue.sort_values('Actual Revenue', ascending=False)
    
    fig1 = px.bar(
        dept_revenue,
        x='Department',
        y='Actual Revenue',
        title='Revenue by Department',
        color='Actual Revenue',
        color_continuous_scale='Blues',
    )
    fig1.update_layout(
        xaxis_title='Department', 
        yaxis_title='Revenue ($)',
        plot_bgcolor='rgba(240, 242, 246, 0.8)',
        paper_bgcolor='rgba(240, 242, 246, 0.8)',
        font=dict(color='#333333'),
        title_font=dict(color='#1E3A8A', size=18)
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Department Profit Margin
    if 'Actual Margin' in df.columns:
        dept_margin = df.groupby('Department')['Actual Margin'].mean().reset_index()
        dept_margin = dept_margin.sort_values('Actual Margin', ascending=False)
        
        fig2 = px.bar(
            dept_margin,
            x='Department',
            y='Actual Margin',
            title='Average Profit Margin by Department',
            color='Actual Margin',
            color_continuous_scale='RdYlGn',
        )
        fig2.update_layout(
            xaxis_title='Department', 
            yaxis_title='Profit Margin (%)',
            plot_bgcolor='rgba(240, 242, 246, 0.8)',
            paper_bgcolor='rgba(240, 242, 246, 0.8)',
            font=dict(color='#333333'),
            title_font=dict(color='#1E3A8A', size=18)
        )
        fig2.update_traces(text=[f"{x:.1%}" for x in dept_margin['Actual Margin']], textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

def create_job_type_charts(df):
    """
    Create job type-based charts for the dashboard.
    """
    if 'Job Type' not in df.columns:
        st.warning("Job Type column not found in the dataset.")
        return
    
    # Job Type Distribution
    job_type_counts = df['Job Type'].value_counts().reset_index()
    job_type_counts.columns = ['Job Type', 'Count']
    
    fig = px.pie(
        job_type_counts,
        names='Job Type',
        values='Count',
        title='Job Type Distribution',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_layout(
        plot_bgcolor='rgba(240, 242, 246, 0.8)',
        paper_bgcolor='rgba(240, 242, 246, 0.8)',
        font=dict(color='#333333'),
        title_font=dict(color='#1E3A8A', size=18)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Job Type by Revenue
    if 'Actual Revenue' in df.columns:
        job_type_revenue = df.groupby('Job Type')['Actual Revenue'].sum().reset_index()
        
        fig2 = px.bar(
            job_type_revenue,
            x='Job Type',
            y='Actual Revenue',
            title='Revenue by Job Type',
            color='Job Type',
            color_discrete_sequence=px.colors.sequential.Blues,
        )
        fig2.update_layout(
            plot_bgcolor='rgba(240, 242, 246, 0.8)',
            paper_bgcolor='rgba(240, 242, 246, 0.8)',
            font=dict(color='#333333'),
            title_font=dict(color='#1E3A8A', size=18)
        )
        st.plotly_chart(fig2, use_container_width=True)

def create_time_series(df):
    """
    Create time series analysis charts.
    """
    if 'Created Date.1' not in df.columns:
        st.warning("Date column not found for time series analysis.")
        return
    
    # Convert to datetime if not already
    df['Created Date'] = pd.to_datetime(df['Created Date.1'])
    df['Month'] = df['Created Date'].dt.to_period('M').astype(str)
    
    if 'Actual Revenue' in df.columns:
        # Monthly Revenue
        monthly_revenue = df.groupby('Month')['Actual Revenue'].sum().reset_index()
        
        fig = px.line(
            monthly_revenue,
            x='Month',
            y='Actual Revenue',
            title='Monthly Revenue',
            markers=True,
            line_shape='linear',
            color_discrete_sequence=['#1E3A8A'],
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=2, color='#1E3A8A')))
        fig.update_layout(
            xaxis_title='Month', 
            yaxis_title='Revenue ($)',
            plot_bgcolor='rgba(240, 242, 246, 0.8)',
            paper_bgcolor='rgba(240, 242, 246, 0.8)',
            font=dict(color='#333333'),
            title_font=dict(color='#1E3A8A', size=18)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Job Count by Month
    monthly_jobs = df.groupby('Month').size().reset_index(name='Count')
    
    fig2 = px.bar(
        monthly_jobs,
        x='Month',
        y='Count',
        title='Number of Jobs by Month',
        color='Count',
        color_continuous_scale='Blues',
    )
    fig2.update_layout(
        plot_bgcolor='rgba(240, 242, 246, 0.8)',
        paper_bgcolor='rgba(240, 242, 246, 0.8)',
        font=dict(color='#333333'),
        title_font=dict(color='#1E3A8A', size=18)
    )
    st.plotly_chart(fig2, use_container_width=True)

def create_data_table(df):
    """
    Create an interactive data table with filters.
    """
    st.subheader("Job Costing Data")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Department' in df.columns:
            departments = ['All'] + sorted(df['Department'].unique().tolist())
            selected_dept = st.selectbox("Department", departments)
    
    with col2:
        if 'Job Type' in df.columns:
            job_types = ['All'] + sorted(df['Job Type'].unique().tolist())
            selected_type = st.selectbox("Job Type", job_types)
    
    with col3:
        if 'Job Status' in df.columns:
            job_statuses = ['All'] + sorted(df['Job Status'].unique().tolist())
            selected_status = st.selectbox("Job Status", job_statuses)
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'Department' in df.columns and selected_dept != 'All':
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    
    if 'Job Type' in df.columns and selected_type != 'All':
        filtered_df = filtered_df[filtered_df['Job Type'] == selected_type]
    
    if 'Job Status' in df.columns and selected_status != 'All':
        filtered_df = filtered_df[filtered_df['Job Status'] == selected_status]
    
    # Display the filtered dataframe
    st.dataframe(filtered_df, use_container_width=True) 

def format_report_as_html(report_text, title="AI-Generated Report"):
    """
    Format a text report as HTML with a clean, modern design for PDF export.
    
    Args:
        report_text (str): The report text content
        title (str): The title of the report
        
    Returns:
        str: HTML formatted report
    """
    # Get current date for the report
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Process the report text to maintain formatting
    # Split by sections (assuming sections start with numbers followed by a dot)
    import re
    
    # Replace markdown headers with HTML headers
    report_text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', report_text, flags=re.MULTILINE)
    report_text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', report_text, flags=re.MULTILINE)
    report_text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', report_text, flags=re.MULTILINE)
    
    # Convert markdown lists to HTML lists
    report_text = re.sub(r'^\* (.*?)$', r'<li>\1</li>', report_text, flags=re.MULTILINE)
    report_text = re.sub(r'^\- (.*?)$', r'<li>\1</li>', report_text, flags=re.MULTILINE)
    report_text = re.sub(r'^\d+\. (.*?)$', r'<li>\1</li>', report_text, flags=re.MULTILINE)
    
    # Wrap consecutive list items in ul tags
    report_text = re.sub(r'(<li>.*?</li>)\n(<li>.*?</li>)', r'\1\2', report_text, flags=re.DOTALL)
    report_text = re.sub(r'(<li>.*?</li>)+', r'<ul>\g<0></ul>', report_text, flags=re.DOTALL)
    
    # Convert markdown tables to HTML tables (if any)
    # This is a simplified approach - complex tables might need more processing
    if '|' in report_text:
        lines = report_text.split('\n')
        table_lines = []
        in_table = False
        html_table = "<table class='report-table'><thead>"
        
        for line in lines:
            if '|' in line and not in_table:
                in_table = True
                # This is a header row
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                html_table += "<tr>"
                for cell in cells:
                    html_table += f"<th>{cell}</th>"
                html_table += "</tr></thead><tbody>"
            elif '|' in line and '-|-' in line.replace(' ', ''):
                # This is a separator row, skip it
                continue
            elif '|' in line and in_table:
                # This is a data row
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                html_table += "<tr>"
                for cell in cells:
                    html_table += f"<td>{cell}</td>"
                html_table += "</tr>"
            elif in_table:
                # End of table
                in_table = False
                html_table += "</tbody></table>"
                table_lines.append(html_table)
                table_lines.append(line)
                html_table = "<table class='report-table'><thead>"
            else:
                table_lines.append(line)
        
        if in_table:
            html_table += "</tbody></table>"
            table_lines.append(html_table)
        
        report_text = '\n'.join(table_lines)
    
    # Convert newlines to <p> tags for better formatting
    paragraphs = report_text.split('\n\n')
    formatted_paragraphs = []
    
    for p in paragraphs:
        if not p.strip():
            continue
        if p.startswith('<h') or p.startswith('<ul') or p.startswith('<table'):
            formatted_paragraphs.append(p)
        else:
            formatted_paragraphs.append(f"<p>{p}</p>")
    
    report_content = '\n'.join(formatted_paragraphs)
    
    # Create the HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                border-bottom: 1px solid #1E3A8A;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .logo {{
                color: #1E3A8A;
                font-size: 24px;
                font-weight: bold;
            }}
            .date {{
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }}
            h1 {{
                color: #1E3A8A;
                font-size: 24px;
                margin-top: 30px;
                margin-bottom: 15px;
            }}
            h2 {{
                color: #2563EB;
                font-size: 20px;
                margin-top: 25px;
                margin-bottom: 10px;
            }}
            h3 {{
                color: #3B82F6;
                font-size: 18px;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            p {{
                margin-bottom: 15px;
            }}
            ul {{
                margin-bottom: 15px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            .report-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .report-table th {{
                background-color: #1E3A8A;
                color: white;
                text-align: left;
                padding: 10px;
            }}
            .report-table td {{
                border: 1px solid #ddd;
                padding: 10px;
            }}
            .report-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .footer {{
                margin-top: 30px;
                padding-top: 10px;
                border-top: 1px solid #ddd;
                font-size: 12px;
                color: #666;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">BGSF Analytics</div>
            <div class="date">Report Generated: {current_date}</div>
        </div>
        
        <h1>{title}</h1>
        
        <div class="report-content">
            {report_content}
        </div>
        
        <div class="footer">
            Generated by BGSF Analytics AI Report Generator
        </div>
    </body>
    </html>
    """
    
    return html 

def get_direct_chat_response(query):
    """
    Get a response from the Gemini model without using embeddings.
    Instead, provide rich context directly from the dataset with enhanced prompt engineering.
    
    Args:
        query (str): User's question about job costing data
        
    Returns:
        str: Response from the Gemini model
    """
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Access the data directly
            df = st.session_state.data
            
            # Calculate dataset summary statistics
            dataset_summary = {
                "total_records": len(df),
                "date_range": f"{df['Created Date.1'].min().date() if 'Created Date.1' in df.columns else 'N/A'} to {df['Created Date.1'].max().date() if 'Created Date.1' in df.columns else 'N/A'}",
                "total_revenue": f"${df['Actual Revenue'].sum():,.2f}" if 'Actual Revenue' in df.columns else "Unknown",
                "avg_margin": f"{df['Actual Margin'].mean():.2%}" if 'Actual Margin' in df.columns else "Unknown",
                "avg_completion_time": f"{(df['Completed Date.1'] - df['Created Date.1']).dt.days.mean():.1f} days" if 'Created Date.1' in df.columns and 'Completed Date.1' in df.columns else "Unknown",
            }
            
            # Extract key information based on the query
            query_keywords = query.lower()
            
            # Prepare specific context based on query keywords
            specific_context = ""
            
            # Department analysis
            if any(keyword in query_keywords for keyword in ['department', 'departments']):
                if 'Department' in df.columns:
                    # Get department metrics
                    dept_stats = []
                    for dept in df['Department'].unique():
                        dept_data = df[df['Department'] == dept]
                        dept_revenue = dept_data['Actual Revenue'].sum() if 'Actual Revenue' in dept_data else 0
                        dept_margin = dept_data['Actual Margin'].mean() if 'Actual Margin' in dept_data else 0
                        dept_jobs = len(dept_data)
                        dept_stats.append({
                            'Department': dept,
                            'Jobs': dept_jobs,
                            'Revenue': dept_revenue,
                            'Margin': dept_margin
                        })
                    
                    # Sort departments by revenue
                    dept_stats.sort(key=lambda x: x['Revenue'], reverse=True)
                    
                    # Format department data - limit to top 10 departments for large datasets
                    specific_context += "DEPARTMENT METRICS:\n"
                    for dept in dept_stats[:10]:  # Limit to top 10
                        specific_context += f"- {dept['Department']}: {dept['Jobs']} jobs, ${dept['Revenue']:,.2f} revenue, {dept['Margin']:.2%} profit margin\n"
            
            # Revenue analysis
            if any(keyword in query_keywords for keyword in ['revenue', 'income', 'earnings', 'money']):
                if 'Actual Revenue' in df.columns:
                    # Get top revenue jobs - limit to top 5
                    top_revenue_jobs = df.nlargest(5, 'Actual Revenue')
                    specific_context += "\nTOP REVENUE JOBS:\n"
                    for _, job in top_revenue_jobs.iterrows():
                        job_number = job['Job Number'] if 'Job Number' in job else 'Unknown'
                        job_revenue = job['Actual Revenue'] if 'Actual Revenue' in job else 0
                        job_dept = job['Department'] if 'Department' in job else 'Unknown'
                        specific_context += f"- Job #{job_number}: ${job_revenue:,.2f}, Department: {job_dept}\n"
                    
                    # Revenue by month if date column exists - limit to most recent 5 months
                    if 'Created Date.1' in df.columns:
                        df['Month'] = df['Created Date.1'].dt.strftime('%Y-%m')
                        monthly_revenue = df.groupby('Month')['Actual Revenue'].sum().reset_index()
                        monthly_revenue = monthly_revenue.sort_values('Month', ascending=False).head(5)
                        
                        specific_context += "\nRECENT MONTHLY REVENUE:\n"
                        for _, month_data in monthly_revenue.iterrows():
                            specific_context += f"- {month_data['Month']}: ${month_data['Actual Revenue']:,.2f}\n"
            
            # Profit margin analysis
            if any(keyword in query_keywords for keyword in ['profit', 'margin', 'profitability']):
                if 'Actual Margin' in df.columns:
                    # Get top margin jobs - limit to top 5
                    top_margin_jobs = df.nlargest(5, 'Actual Margin')
                    specific_context += "\nTOP PROFIT MARGIN JOBS:\n"
                    for _, job in top_margin_jobs.iterrows():
                        job_number = job['Job Number'] if 'Job Number' in job else 'Unknown'
                        job_margin = job['Actual Margin'] if 'Actual Margin' in job else 0
                        job_dept = job['Department'] if 'Department' in job else 'Unknown'
                        specific_context += f"- Job #{job_number}: {job_margin:.2%}, Department: {job_dept}\n"
                    
                    # Average margin by department - limit to top 10
                    if 'Department' in df.columns:
                        dept_margins = df.groupby('Department')['Actual Margin'].mean().reset_index()
                        dept_margins = dept_margins.sort_values('Actual Margin', ascending=False).head(10)
                        
                        specific_context += "\nDEPARTMENT PROFIT MARGINS:\n"
                        for _, dept_data in dept_margins.iterrows():
                            specific_context += f"- {dept_data['Department']}: {dept_data['Actual Margin']:.2%}\n"
            
            # Job type analysis
            if any(keyword in query_keywords for keyword in ['job type', 'job types', 'type of job']):
                if 'Job Type' in df.columns:
                    # Count by job type - limit to top 10
                    job_type_counts = df['Job Type'].value_counts().head(10)
                    specific_context += "\nJOB TYPE DISTRIBUTION:\n"
                    for job_type, count in job_type_counts.items():
                        specific_context += f"- {job_type}: {count} jobs\n"
                    
                    # Revenue by job type - limit to top 10
                    if 'Actual Revenue' in df.columns:
                        job_type_revenue = df.groupby('Job Type')['Actual Revenue'].sum().reset_index()
                        job_type_revenue = job_type_revenue.sort_values('Actual Revenue', ascending=False).head(10)
                        
                        specific_context += "\nJOB TYPE REVENUE:\n"
                        for _, jt_data in job_type_revenue.iterrows():
                            specific_context += f"- {jt_data['Job Type']}: ${jt_data['Actual Revenue']:,.2f}\n"
            
            # Status analysis
            if any(keyword in query_keywords for keyword in ['status', 'progress', 'complete', 'completed']):
                if 'Job Status' in df.columns:
                    # Jobs by status
                    status_counts = df['Job Status'].value_counts()
                    specific_context += "\nJOB STATUS DISTRIBUTION:\n"
                    for status, count in status_counts.items():
                        specific_context += f"- {status}: {count} jobs\n"
            
            # Time/completion analysis
            if any(keyword in query_keywords for keyword in ['time', 'days', 'duration', 'completion time']):
                if 'Created Date.1' in df.columns and 'Completed Date.1' in df.columns:
                    # Calculate completion times
                    df['Completion Time'] = (df['Completed Date.1'] - df['Created Date.1']).dt.days
                    avg_completion = df['Completion Time'].mean()
                    max_completion = df['Completion Time'].max()
                    min_completion = df['Completion Time'].min()
                    
                    specific_context += "\nCOMPLETION TIME STATISTICS:\n"
                    specific_context += f"- Average: {avg_completion:.1f} days\n"
                    specific_context += f"- Minimum: {min_completion:.1f} days\n"
                    specific_context += f"- Maximum: {max_completion:.1f} days\n"
                    
                    # Completion time by department - limit to top 10
                    if 'Department' in df.columns:
                        dept_completion = df.groupby('Department')['Completion Time'].mean().reset_index()
                        dept_completion = dept_completion.sort_values('Completion Time').head(10)
                        
                        specific_context += "\nDEPARTMENT COMPLETION TIMES:\n"
                        for _, dept_data in dept_completion.iterrows():
                            specific_context += f"- {dept_data['Department']}: {dept_data['Completion Time']:.1f} days\n"
            
            # If no specific context was added, provide a general dataset overview
            if not specific_context:
                # General dataset statistics
                specific_context = "GENERAL DATASET OVERVIEW:\n"
                
                # Department stats if available - limit to top 10
                if 'Department' in df.columns:
                    dept_counts = df['Department'].value_counts().head(10)
                    specific_context += "\nTop Departments:\n"
                    for dept, count in dept_counts.items():
                        specific_context += f"- {dept}: {count} jobs\n"
                
                # Job type stats if available - limit to top 10
                if 'Job Type' in df.columns:
                    job_type_counts = df['Job Type'].value_counts().head(10)
                    specific_context += "\nTop Job Types:\n"
                    for job_type, count in job_type_counts.items():
                        specific_context += f"- {job_type}: {count} jobs\n"
                
                # Status stats if available
                if 'Job Status' in df.columns:
                    status_counts = df['Job Status'].value_counts()
                    specific_context += "\nJob Statuses:\n"
                    for status, count in status_counts.items():
                        specific_context += f"- {status}: {count} jobs\n"
            
            # Prepare the full context incorporating the dataset summary and specific context
            full_context = f"""
DATASET SUMMARY:
Total Records: {dataset_summary['total_records']}
Date Range: {dataset_summary['date_range']}
Total Revenue: {dataset_summary['total_revenue']}
Average Profit Margin: {dataset_summary['avg_margin']}
Average Completion Time: {dataset_summary['avg_completion_time']}

{specific_context}
"""
            
            model = get_gemini_model()
            
            # Craft a detailed prompt with enhanced instructions
            prompt = f"""
You are a helpful assistant specializing in job costing and construction financial analysis.
Use the following context to answer the question at the end. The context includes a dataset summary and specific relevant data extracts.

Context:
{full_context}

Question: {query}

Guidelines for your response:
1. Provide a clear, concise answer based on the context
2. Include specific numbers and statistics when relevant
3. If the exact information is not in the context, use the most relevant data available
4. If you cannot answer with available data, say so politely
5. Format numbers professionally (e.g., use commas for thousands, dollar signs for currency)
6. Format percentages properly (e.g., 15.2%)
7. Do not make up data or assumptions
8. Respond in plain text without any HTML or special formatting

CRITICALLY IMPORTANT: Write your response in PLAIN TEXT ONLY.
DO NOT include HTML tags of any kind, including </div>, <div>, <p>, </p>, or any other markup.
Your response should contain only text with no HTML formatting whatsoever.
"""
            
            # Add retry logic for the model.generate_content call
            generation_retry_count = 0
            while generation_retry_count < 3:  # Max 3 retries for generation
                try:
                    # Add safety params to the model generation
                    response = model.generate_content(
                        prompt,
                        generation_config=GenerationConfig(
                            temperature=0.2,
                            top_p=0.95,
                            top_k=40,
                            max_output_tokens=1024  # Ensure reasonable response length
                        )
                    )
                    break  # Break out of the inner retry loop if successful
                except Exception as e:
                    generation_retry_count += 1
                    if "RATE_LIMIT_EXCEEDED" in str(e) or "429" in str(e):
                        # Exponential backoff with jitter for rate limit errors
                        wait_time = (2 ** generation_retry_count) + random.uniform(0, 1)
                        print(f"Rate limit exceeded in generate_content. Waiting {wait_time:.2f} seconds before retry {generation_retry_count}/3")
                        time.sleep(wait_time)
                    else:
                        # For other errors, shorter wait
                        time.sleep(1)
                    
                    # If we've reached max retries, let it fail and be caught by the outer try-except
                    if generation_retry_count == 3:
                        response = model.generate_content(prompt)  # This will raise if it fails
            
            # Extract text from response based on new API structure
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'parts'):
                response_text = response.parts[0].text
            else:
                response_text = "No response generated."
            
            # Clean any HTML that might have been generated despite instructions
            response_text = re.sub(r'<[^>]*>', '', response_text)
            
            return response_text
            
        except Exception as e:
            retry_count += 1
            if "RATE_LIMIT_EXCEEDED" in str(e) or "429" in str(e):
                # Exponential backoff with jitter for rate limit errors
                wait_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"Rate limit exceeded in chat response. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            else:
                # For other errors, wait a bit but not as long
                wait_time = retry_count + random.uniform(0, 1)
                print(f"Error in chat response: {e}. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            
            if retry_count == max_retries:
                # If we've reached max retries, return an error message
                return f"I apologize, but I encountered an error while processing your request. The system is currently experiencing high demand. Please try again in a few moments. Error details: {str(e)}"
                
    # Shouldn't reach here due to the return in the exception handler, but just in case
    return "I apologize, but I encountered an error while processing your request. Please try again later." 