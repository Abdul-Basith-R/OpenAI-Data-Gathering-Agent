import datetime
import logging
import pandas as pd
import asyncio
from openai import AsyncOpenAI
import os
import openai
import time
import streamlit as st


if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=f'logs/log_{current_time}.txt',
                    level=logging.INFO,
                    format='%(asctime)s: %(levelname)s: %(message)s')


async def value_extractor(chat_history, api_key):
    """
    This function uses OpenAI API to extract key personal information from a chat history.

    Args:
        chat_history: A list of dictionaries representing the chat conversation.
        api_key: Your OpenAI API key.

    Returns:
        A string containing the extracted personal information in JSON format.
        None if any error occurs.
    """

    try:
        # Initialize the OpenAI client with your API key.
        client = AsyncOpenAI(api_key=api_key)

        # Create a context message for the GPT-3 model.
        context = {
            "role": "system",
            "content": """
            Hello AI,

            A conversation between a user and an assistant will be given in a dictionary format.

            Your task is to identify and extract only the key personal information provided by the user. Look for the user's full name, email address, highest level of education, phone number, residential location, and date of birth.

            Once you have identified these details, please format them into a JSON object. Make sure to handle cases where some information might not be provided. Your output should be structured and clear, providing the only the extracted information in a JSON format.

            Remember, the focus is only on accurately extracting the relevant personal details mentioned in the conversation. If certain information is not provided, leave it blank or indicate it as 'Not Provided' in the JSON object.

            Your primary goal is to ensure that the extracted information is accurate and well-structured in the output.
            """
        }

        # Prepend the context message to the chat history.
        chat_history.insert(0, context)

        # Use the GPT-3 model to analyze the chat history and complete the prompt.
        completion = await client.chat.completions.create(model="gpt-3.5-turbo", messages=chat_history)

        # Return the extracted information from the first completion choice.
        return completion.choices[0].message.content

    except Exception as e:
        # Log any exceptions that occur during extraction.
        logging.exception(f"Exception occurred in value_extractor function : {e}")
        return None


def write_dict_to_csv(dict_data, filename):
    """
    This function writes a dictionary containing extracted information to a CSV file.

    Args:
        dict_data: A dictionary containing the extracted personal information.
        filename: The name of the CSV file to write to.

    Returns:
        None

    Raises:
        Exception: If any error occurs during writing.
    """

    try:
        # Convert each scalar value in the dictionary to a list.
        # This is necessary for Pandas to properly write the data.
        dict_data = {key: [value] for key, value in dict_data.items()}

        # Convert the dictionary to a Pandas DataFrame.
        df = pd.DataFrame.from_dict(dict_data)

        # Check if the CSV file already exists.
        if os.path.exists(filename):
            # If the file exists, append the new data to the end (append mode).
            df.to_csv(filename, mode="a", index=False, header=False)
        else:
            # If the file doesn't exist, create it and write the data (no append mode).
            df.to_csv(filename, index=False)

    except Exception as e:
        # Log any exceptions that occur during writing.
        logging.exception("Exception occurred in write_dict_to_csv function")


# Sidebar Widgets for getting API key and Selecting the desired model
with st.sidebar:
    st.header("OpenAI Configuration")

    selected_model = st.selectbox("Model", ['gpt-3.5-turbo', 'gpt-4'], index=1)
    selected_key = st.text_input("API Key", type="password")

st.write("# Data Collection Chat Agent")


def data_gatherer(selected_model, selected_key):
    try:
        with st.container():
            if selected_key:
                if 'client' not in st.session_state:
                    # Initialize the client
                    st.session_state.client = openai.OpenAI( api_key=selected_key)
                    
                    # Step 1: Create an Assistant
                    st.session_state.assistant = st.session_state.client.beta.assistants.create( name="Alice",
                        instructions="""You are a data-gathering agent working for a consultancy company You need to collect basic mandatory information that is used for creating the candidate's profile. We assure that their data will be safe with us and be processed with utmost care. No information leak is possible. Their data will be safe with us.
                        You should initiate the conversation. Ask for one piece of information at a time.  Address the user by name to create a personal connection. Conversations should be Formal Should Keep the conversation natural and avoid repetition or irrelevant questions. When the user responds, acknowledge the information they've shared, and then proceed to the next question. Review the response from the user and If the user hesitates, engage in small talk related to their interests or concerns about data privacy to reassure them before asking the information again. Should continue the conversation until all the information is extracted. If the user asks relevant questions about their data try to answer it 
                        You are allowed to ask only one question
                        Initiating Conversation: Welcome, I'm your AI assistant, here to guide you through creating your profile for exciting career opportunities. Let's get acquainted so we can tailor the experience to your unique background. Can I begin by asking your full name, please?
                        Small Talk Prompt (General Interest):
                        "By the way, [user name], we're always keen to know our candidates better. Do you have any hobbies or interests you'd like to share? It's always great to learn more about the people we're assisting."
                        Small Talk Prompt (Data Privacy):
                        "I noticed you might have some reservations about data privacy. It's a topic I find quite important too. Did you know we use advanced encryption methods to protect all client data? How do you usually approach data privacy in your day-to-day online activities, [user name]?"
                        Continuation After Small Talk (Back to Data Collection):
                        "Thanks for sharing that, [user name]. Now, if you don't mind, shall we continue with setting up your profile? Could you please provide your email address? It's important for sending you notifications about potential job opportunities."
                        Reassuring on Data Privacy: "You seem concerned about data privacy, [User's Name]. Let me assure you that your personal information is encrypted and stored securely. We value your privacy and handle your data with utmost care."
                        Hesitation Response (General): "I understand your concern, [user name]. We take data privacy very seriously, and all your information is securely stored and used only for job-matching purposes. We never share it with third parties without your consent. I understand if some questions feel personal. Your data privacy is our top priority, and we only collect information essential for a smooth profile creation process"
                        Conclusion: "Wonderful, that completes your profile for now. Thank you for your time and valuable information, [user name]. You can start receiving personalized job recommendations. We appreciate your trust in [interview company name]."
                        Response to User's Data-Related Questions:
                        "I'm glad you asked, [user name]. It's important to stay informed. When it comes to [specific user question], here's how we handle it..."
                        Information to be collected and their reasons: 
                        Name: For Account creation and addressing
                        , Email for sending notifications 
                        , Education: essential for job matching
                        , Phone number: for contact on any recruiter actions
                        , Date of Birth: For verification purposes
                        and Location: for tailored opportunities """,
                        model=selected_model)
                    logging.info(f"Assistant Creation Successfull. Assistant id :  {st.session_state.assistant.id}")

                    # Step 2: Create a Thread
                    st.session_state.thread = st.session_state.client.beta.threads.create()
                    logging.info( f"Thread Creation Successfull. Thread ID :  {st.session_state.thread}")

                # Get user input.
                user_query = st.chat_input("Type something...")

                if user_query:
                    # Check if user wants to exit.
                    if user_query.lower() == "exit":
                        return "exit"
                    
                    # Step 3: Add a Message to a Thread
                    message = st.session_state.client.beta.threads.messages.create(
                        thread_id=st.session_state.thread.id,
                        role="user",
                        content=user_query
                    )
                    logging.info(f"message Creation Successfull {message}")

                    # Step 4: Run the Assistant
                    run = st.session_state.client.beta.threads.runs.create(
                        thread_id=st.session_state.thread.id,
                        assistant_id=st.session_state.assistant.id,
                    )
                    logging.info(f"Run Creation Successfull {run}")

                    try:
                        while True:
                            # Wait for 5 seconds for the assistant to process.
                            time.sleep(5)

                            # Retrieve the run status
                            run_status = st.session_state.client.beta.threads.runs.retrieve(
                                thread_id=st.session_state.thread.id,
                                run_id=run.id
                            )

                            # If run is completed, get message
                            if run_status.status == 'completed':
                                messages = st.session_state.client.beta.threads.messages.list(
                                    thread_id=st.session_state.thread.id,
                                    order="asc"
                                )

                                # Loop through messages and print content based on role and add them to conversations
                                for msg in messages.data:
                                    role = msg.role
                                    content = msg.content[0].text.value

                                    st.chat_message(role)
                                    st.markdown(content)

                                    # Store user and assistant messages for return.
                                    if role == "user":
                                        user_chat = {"role": role,"content": content}
                                    if role == "assistant":
                                        assisstant_chat = {"role": role, "content": content}

                                    logging.info(f"{role} : {content}")
                                break
                            # If still processing, show waiting message and try again.
                            else:
                                st.write("Waiting for the Assistant to process...")
                                time.sleep(5)

                    except Exception as e:
                        logging.exception("Exception occurred in write_dict_to_csv function")

            else:
                st.warning('You must provide valid OpenAI API key and choose preferred model', icon="⚠️")
                st.stop()

        if user_chat or assisstant_chat:
            return user_chat, assisstant_chat

    except Exception as e:
        logging.exception("Exception occurred in write_dict_to_csv function")


if 'chats' not in st.session_state:
    st.session_state["chats"] = []


def main():
    """
    This function manages the main application loop for gathering and extracting personal information.

    It loops through the following steps until completion or user exits:
    1. Calls `data_gatherer` to initiate a conversation with the AI assistant and retrieves user and assistant messages.
    2. Checks if both messages are dictionaries (meaning the conversation has progressed).
    3. Logs the user and assistant chat history.
    4. Appends both messages to the `chats` list in session state.
    6. Checks if the user requested to exit.
    7. Logs the final list of chats.
    8. Calls `value_extractor` asynchronously to analyze the chats and extract personal information.
    9. Writes the extracted information to a CSV file named "output.csv".
    10. Exits the loop if completed or user requested to exit.

    **Functions used:**

    * `data_gatherer`: Initiates a conversation with the AI assistant and retrieves user and assistant messages.
    * `logging.info`: Logs information messages to the console.
    * `st.session_state.chats`: Accesses the list of chat messages stored in session state.
    * `st.session_state.chats.append`: Adds a new message dictionary to the `chats` list.
    * `asyncio.run`: Runs the `value_extractor` function asynchronously and collects the result.
    * `value_extractor`: Analyzes the chat history list and extracts personal information as a JSON string.
    * `write_dict_to_csv`: Writes the extracted information dictionary to a CSV file.

    """

    while True:
        result = data_gatherer(selected_key, selected_model)
        if result is not None and result != "exit":

            # 1. Get user and assistant messages from the conversation.
            user_chat, assisstant_chat = result

            # 2. Check if both messages are valid dictionaries.
            if isinstance(assisstant_chat, dict) and isinstance(user_chat, dict):
                # 3. Log the chat history.
                logging.info(f"user Chat History : {user_chat}")
                logging.info(f"user Chat History : {assisstant_chat}")

                # 4. Add both messages to the chats list in session state.
                st.session_state.chats.append(user_chat)
                st.session_state.chats.append(assisstant_chat)

                # 5. Check if conversation is complete.
                if "completes your profile for now" in assisstant_chat["content"]:
                    break

        # 6. Check if user wants to exit.
        elif result == "exit":
            break

        # 7. Log the final chats list.
        logging.info(f"Chats: {st.session_state.chats}")

    # 8. Extract personal information from the chats asynchronously.
    response_message = asyncio.run(value_extractor(st.session_state.chats, selected_key))

    # 9. Write the extracted information to a CSV file.
    write_dict_to_csv(response_message, "output.csv")


if __name__ == "__main__":
    main()
