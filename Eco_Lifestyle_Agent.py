#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install ibm-watson-machine-learning --quiet')


# In[4]:


# Store API credentials
API_KEY = "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko"
WML_URL = "https://us-south.ml.cloud.ibm.com"

print("Credentials added successfully ✅")


# In[7]:


project_id = "894f0698-2336-4d9d-b3c9-62fcef9ef65a"


# In[21]:


get_ipython().system('pip install ibm-watson-machine-learning --quiet')

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

project_id = "894f0698-2336-4d9d-b3c9-62fcef9ef65a"

model_id = "ibm/granite-3-2b-instruct"

model = Model(
    model_id=model_id,
    credentials={"apikey": API_KEY, "url": WML_URL},
    params={
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.TEMPERATURE: 0.5
    },
    project_id=project_id
)

print("✅ Granite model connected successfully")


# In[22]:


test_prompt = "Write a one-line welcome message for my AI project."

# Ask for raw response to check structure
response = model.generate_text(prompt=test_prompt, raw_response=True)

print(response)  # Check what Granite is actually returning


# In[23]:


test_prompt = "Write a one-line welcome message for my AI project."

response = model.generate_text(prompt=test_prompt)

# If response is just a string, print it directly
print("✅ Model Output:", response)


# In[25]:


# Download files from bucket
cos.download_file(BUCKET_NAME, "Q&A.txt", "Q&A.txt")
cos.download_file(BUCKET_NAME, "Info.txt", "Info.txt")

# Read Q&A
with open("Q&A.txt", "r", encoding="utf-8") as f:
 qna_text = f.read()

# Read Info
with open("Info.txt", "r", encoding="utf-8") as f:
 info_text = f.read()

print("✅ Q&A and Info loaded successfully!")


# In[33]:


from ibm_boto3 import client
from ibm_botocore.client import Config

# Paste your credentials here
API_KEY = "QS2rDLeiUPC7xAYyKyvMKbjygfBSz2zTrD6GopfSQUz7"
RESOURCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/a62c1f6969474417b3c19425bd6f72fa:7e5d1953-1a30-49b6-91f1-e411cdc7492d::"
COS_ENDPOINT = "https://s3.us-south.cloud-object-storage.appdomain.cloud"  # Change if region different
BUCKET_NAME = "ecolifestyleagen-donotdelete-pr-ugex0qkoku0wcs"

# Create COS client
cos = client(
    "s3",
    ibm_api_key_id=API_KEY,
    ibm_service_instance_id=RESOURCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

print("✅ COS connected successfully")


# In[39]:


with open("Q&A.txt", "r", encoding="utf-8") as f:
    qa_data = f.read()

with open("Info.txt", "r", encoding="utf-8") as f:
    info_data = f.read()


# In[40]:


from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split Q&A data
qa_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
qa_docs = qa_splitter.create_documents([qa_data])

# Split Info data
info_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
info_docs = info_splitter.create_documents([info_data])

len(qa_docs), len(info_docs)


# In[48]:


from ibm_watsonx_ai.foundation_models import Model

llm = Model(
    model_id="ibm/granite-3-2b-instruct",
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a"
)

query = "Suggest 5 ways to reduce plastic usage at home"

response = llm.generate_text(
    prompt=f"""
You are an Eco Lifestyle Agent.  
Use the following knowledge base to answer:
{qa_data[:1000]}  # first part of Q&A
{info_data[:1000]} # first part of Info
Question: {query}
Answer:"""
)

print(response)


# In[50]:


from ibm_watsonx_ai.foundation_models import Model

# Your Watsonx credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",  # Or your region endpoint
    "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko"  # From IBM Cloud > API keys
}

PROJECT_ID = "894f0698-2336-4d9d-b3c9-62fcef9ef65a"

# Initialize Granite
llm = Model(
    model_id="ibm/granite-3-2b-instruct",  # or granite-13b-chat-v2
    credentials=credentials,
    project_id=PROJECT_ID
)

# Query
query = "Suggest 5 ways to reduce plastic usage at home"

response = llm.generate_text(
    prompt=f"""
You are an Eco Lifestyle Agent.
Use the following knowledge base to answer:
{qa_data[:1000]}
{info_data[:1000]}
Question: {query}
Answer:"""
)

print(response)


# In[55]:


get_ipython().system('pip install gradio --quiet')

import gradio as gr

def chat_with_ai(message, history):
    response = llm.generate_text(
        prompt=f"You are an Eco AI. Respond clearly.\nUser: {message}\nAnswer:",
        max_new_tokens=400,
        temperature=0.7
    )
    history.append((message, response))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Ask anything about environment...")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_with_ai, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)


# In[57]:


from ibm_watsonx_ai.foundation_models import Model
import gradio as gr

# ---- 1️⃣ Connect to Granite Model ----
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko"
}

llm = Model(
    model_id="ibm/granite-3-2b-instruct", 
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a",
    credentials=credentials
)

# ---- 2️⃣ Chatbot Function ----
def respond(message, history):
    try:
        # Send prompt to Granite
        response = llm.generate_text(
            prompt=message,
            max_new_tokens=300,   # Ensures it gives longer answers
            temperature=0.7       # Creativity level
        )

        # Extract model output
        model_reply = response["results"][0]["generated_text"]

        # Return updated chat history
        return history + [(message, model_reply)]
    except Exception as e:
        return history + [(message, f"⚠️ Error: {e}")]

# ---- 3️⃣ Gradio Chat UI ----
chatbot = gr.Chatbot()
msg = gr.Textbox(label="Ask me anything about Environment / Eco Awareness")

with gr.Blocks() as demo:
    gr.Markdown("## 🌱 Eco-Friendly AI Chatbot (Granite Powered)")
    chatbot.render()
    msg.render()
    
    msg.submit(respond, [msg, chatbot], chatbot)

demo.launch(share=True)


# In[59]:


from ibm_watsonx_ai.foundation_models import Model
import gradio as gr

# ---- 1️⃣ Connect to Granite Model ----
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko"
}

llm = Model(
    model_id="ibm/granite-3-2b-instruct", 
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a",
    credentials=credentials
)

# ---- 2️⃣ Chatbot Function ----
def respond(message, history):
    try:
        # Send prompt to Granite with proper parameters
        response = llm.generate_text(
            prompt=message,
            parameters={
                "max_new_tokens": 300,   # Longer responses
                "temperature": 0.7       # Creativity
            }
        )

        # Extract model output
        model_reply = response["results"][0]["generated_text"]

        # Return updated chat history
        return history + [(message, model_reply)]
    except Exception as e:
        return history + [(message, f"⚠️ Error: {e}")]

# ---- 3️⃣ Gradio Chat UI ----
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 Eco-Friendly AI Chatbot (Granite Powered)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask me anything about Environment / Eco Awareness")
    
    msg.submit(respond, [msg, chatbot], chatbot)

demo.launch(share=True)


# In[61]:


from ibm_watsonx_ai.foundation_models import Model
import gradio as gr

# ---- 1️⃣ Granite Model Setup ----
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko"
}

llm = Model(
    model_id="ibm/granite-3-2b-instruct", 
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a",
    credentials=credentials
)

# ---- 2️⃣ Chatbot Function ----
def respond(message, history):
    try:
        # Supported parameters (no max_new_tokens)
        params = {
            "decoding_method": "greedy",   # Or "sample"
            "temperature": 0.7,            # Creativity
            "max_tokens": 300              # Controls length
        }

        # Generate response
        response = llm.generate_text(
            prompt=message,
            parameters=params
        )

        reply = response["results"][0]["generated_text"]

        # Update chat history
        return history + [(message, reply)]
    except Exception as e:
        return history + [(message, f"⚠️ Error: {e}")]

# ---- 3️⃣ Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 Eco-Friendly AI Chatbot (Granite Powered)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask me anything about Environment / Eco Awareness")
    
    msg.submit(respond, [msg, chatbot], chatbot)

demo.launch(share=True)


# In[63]:


from ibm_watsonx_ai.foundation_models import Model
import gradio as gr

# 1️⃣ Credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko"
}

# 2️⃣ Model Initialization
llm = Model(
    model_id="ibm/granite-3-2b-instruct",
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a",
    credentials=credentials
)

# 3️⃣ Response Function
def respond(message, history):
    try:
        raw_response = llm.generate_text(prompt=message)
        print(raw_response)  # Debug output

        # Handle list return
        if isinstance(raw_response, list):
            reply = raw_response[0].get("generated_text", str(raw_response[0]))
        # Handle dict return
        elif isinstance(raw_response, dict):
            reply = raw_response.get("results", [{}])[0].get("generated_text", str(raw_response))
        # Handle plain string
        elif isinstance(raw_response, str):
            reply = raw_response
        else:
            reply = str(raw_response)

        return history + [(message, reply)]
    except Exception as e:
        return history + [(message, f"⚠️ Error: {str(e)}")]
# 4️⃣ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 Eco-Friendly AI Chatbot (Granite Minimal Version)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask your eco-friendly question here")

    msg.submit(respond, [msg, chatbot], chatbot)

demo.launch(share=True)


# In[64]:


import gradio as gr
from ibm_watsonx_ai.foundation_models import Model

# Connect to Granite model
llm = Model(
    model_id="ibm/granite-3-2b-instruct",
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a",
    credentials={
        "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko",
        "url": "https://us-south.ml.cloud.ibm.com"
    }
)

def chat_with_model(message, history):
    # Force long, structured, 5-point answers
    prompt = f"""
You are an Eco-Friendly AI Advisor.
ALWAYS respond with exactly 5 numbered points.
Each point should have **at least 30 words** with practical tips and explanations.
Never give short answers. Expand as much as possible.
User Question: {message}
"""
    try:
        response = llm.generate_text(
            prompt=prompt,
            max_tokens=400,   # Allow ~300 words
            temperature=0.7
        )
        return response["results"][0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Create chatbot UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Eco-Friendly Chatbot 🌱")
    msg = gr.Textbox(label="Ask anything about Environment 🌍")
    clear = gr.Button("Clear Chat")

    def respond(message, history):
        bot_message = chat_with_model(message, history)
        history.append((message, bot_message))
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)


# In[66]:


import gradio as gr
from ibm_watsonx_ai.foundation_models import Model

# Connect to Granite model
llm = Model(
    model_id="ibm/granite-3-2b-instruct",
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a",
    credentials={
        "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko",
        "url": "https://us-south.ml.cloud.ibm.com"
    }
)

def chat_with_model(message, history):
    prompt = f"""
You are an Eco-Friendly AI Advisor.
Always respond with **exactly 5 numbered points**.
Each point must have at least 30 words.
Use clear explanations and be helpful.

User Question: {message}
"""

    try:
        response = llm.generate_text(
            prompt=prompt,
            params={
                "max_tokens": 400,   # Long response
                "temperature": 0.7,
                "top_p": 0.9
            }
        )
        return response["results"][0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Create chatbot UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Eco-Friendly Chatbot 🌱", type="messages")
    msg = gr.Textbox(label="Ask anything about Environment 🌍")
    clear = gr.Button("Clear Chat")

    def respond(message, history):
        bot_message = chat_with_model(message, history)
        history.append((message, bot_message))
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)


# In[69]:


import gradio as gr
from ibm_watsonx_ai.foundation_models import Model

# IBM watsonx model connection
llm = Model(
    model_id="ibm/granite-3-2b-instruct",
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a",
    credentials={
        "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko",
        "url": "https://us-south.ml.cloud.ibm.com"
    }
)

def chat_with_model(message, history):
    prompt = f"""
You are an Eco-Friendly AI Advisor.
Always respond with **exactly 5 numbered points**.
Each point must have at least 30 words.
Use clear explanations and be helpful.

User Question: {message}
"""

    try:
        response = llm.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": 400,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )
        return response["results"][0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio chatbot UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Eco-Friendly Chatbot 🌱", type="messages")
    msg = gr.Textbox(label="Ask anything about Environment 🌍")
    clear = gr.Button("Clear Chat")

    def respond(message, history):
        bot_message = chat_with_model(message, history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_message})
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(share=True)


# In[70]:


import gradio as gr
from ibm_watsonx_ai.foundation_models import Model

# IBM watsonx model connection
llm = Model(
    model_id="ibm/granite-3-2b-instruct",
    project_id="894f0698-2336-4d9d-b3c9-62fcef9ef65a",
    credentials={
        "apikey": "z84CoVFFU3f5373GneBXmDOMSUctnb27ERlXbDFCCxko",
        "url": "https://us-south.ml.cloud.ibm.com"
    }
)

def chat_with_model(message, history):
    prompt = f"""
You are an Eco-Friendly AI Advisor.
Always respond with **exactly 5 numbered points**.
Each point must have at least 30 words.
Use clear explanations and be helpful.

User Question: {message}
"""

    try:
        response = llm.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": 400,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )

        # Print raw response to debug format
        print("DEBUG Raw Response:", response)

        # Try extracting text safely
        if isinstance(response, dict) and "results" in response:
            return response["results"][0].get("generated_text", "No text generated.")
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio chatbot UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Eco-Friendly Chatbot 🌱", type="messages")
    msg = gr.Textbox(label="Ask anything about Environment 🌍")
    clear = gr.Button("Clear Chat")

    def respond(message, history):
        bot_message = chat_with_model(message, history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_message})
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(share=True)


# In[ ]:




