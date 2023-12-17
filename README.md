Project Overview:

The Data Gathering Agent is a Python-based chatbot designed to interact with users, collect  information, and store it efficiently. This project utilizes Streamlit and OpenAI to gather  information  through an interactive conversation with an AI assistant. The extracted information is then analyzed and formatted into a JSON string and saved as a CSV file.This project is ideal for researchers, marketers, and organizations seeking a friendly and effective way to gather data from participants.

Key Features:

AI-powered data gathering: Streamlit interface facilitates a natural conversation with an AI assistant that collects user information.
OpenAI analysis: Extracted information is analyzed using OpenAI API for accurate extraction and processing.
Automatic extraction: Extracted data is converted to a JSON format for easy retrieval and saved as CSV file.

Requirements:

Python
Streamlit
OpenAI API key

Getting Started:

Clone the repository.
Install the required Python packages:

pip install -r requirements.txt

Run the application:

streamlit run Data_Gathering_Agent.py

Obtain an OpenAI API key. Paste in the sidebar of the application in API key field.

Usage
Initiates a conversation using data_gatherer and retrieves user and AI assistant messages.
Appends messages to the chats list in session state.
Checks for user exit requests.
Calls value_extractor asynchronously to analyze chats and extract personal information.
Writes extracted information to "output.csv".
Exits upon completion or user request.

Further Development:

This project can be extended in various ways, such as:

Giving roles to the agent of the users desires.
Integrating with additional APIs or databases for data storage and analysis.
Building advanced conversational logic for the AI assistant.
