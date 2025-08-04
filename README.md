# Eco_agent_Edunet_IBM_Skillsbuild_Internship
AI-powered eco-friendly chatbot built using IBM Watsonx Granite &amp; Gradio — provides interactive sustainability tips &amp; answers in real-time.
# Eco-Friendly Chatbot using IBM Granite + Gradio

This project is a simple AI-powered chatbot built using IBM's Granite model and Gradio.  
It takes user queries and responds with advice, tips, or answers based on the prompt.

## Features
- Uses IBM Watsonx.ai Granite 3-2B Instruct model.
- Interactive Chatbot UI with Gradio.
- Real-time responses to user questions.
- Easy to run locally or deploy on cloud.

## Project Structure
.
├── app.py              # Main chatbot code (exported from Jupyter Notebook)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation

## Installation & Setup

### 1. Clone this repository

git clone https://github.com/yourusername/eco-friendly-chatbot.git
cd eco-friendly-chatbot

2. Install dependencies

pip install -r requirements.txt

3. Add your IBM Credentials

Edit app.py and replace with your Watsonx.ai credentials:

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "YOUR_API_KEY"
}

4. Run the chatbot

python app.py

The chatbot will be available at:

http://127.0.0.1:7860

Tech Stack

IBM Watsonx.ai Granite Model

Gradio

Python 3.9+

Jupyter Notebook (for development)
