import streamlit as st
from openai import OpenAI
import os
import io
import tiktoken
import PyPDF2
import pandas as pd
import concurrent.futures
from pydub import AudioSegment
from pydub.playback import play
import logging
from typing import Optional, List
from streamlit.logger import get_logger
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import tempfile
from langchain.schema import Document
import speech_recognition as sr
import threading
import time
import queue
import numpy as np
from streamlit.runtime.scriptrunner import add_script_run_ctx
import uuid
import random
import pyaudio

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit configuration
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Initialize logger
logger = get_logger(__name__)

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'voice_ai_active' not in st.session_state:
    st.session_state.voice_ai_active = False
if 'last_user_input_time' not in st.session_state:
    st.session_state.last_user_input_time = time.time()
if 'document_content' not in st.session_state:
    st.session_state.document_content = ""
if 'queries_left' not in st.session_state:
    st.session_state.queries_left = 10

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize audio queue
audio_queue = queue.Queue()

# Exit keyword
EXIT_KEYWORD = "goodbye assistant"

# Conversation fillers
FILLERS = ["Um", "Uh", "Let's see", "Well", "Hmm"]

def add_filler():
    return random.choice(FILLERS) + ", " if random.random() < 0.3 else ""

def speak(text):
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=text
        )
        audio = AudioSegment.from_mp3(io.BytesIO(response.content))
        play(audio)
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")

def read_file_content(uploaded_file):
    file_type = uploaded_file.type
    file_content = io.BytesIO(uploaded_file.read())
    
    try:
        if file_type == "text/plain":
            return file_content.getvalue().decode("utf-8", errors="replace")
        elif file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file_content)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
        elif file_type == "text/csv":
            df = pd.read_csv(file_content)
            return df.to_string()
        else:
            return "Error: Unsupported file type. Please upload a TXT, PDF, or CSV file."
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return f"Error: Unable to read file. Please try again or contact support."

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith('.csv'):
            loader = CSVLoader(temp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file_path)
        else:
            st.error("Unsupported file format")
            return None, None

        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
        
        conversation = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0),
            retriever=vectorstore.as_retriever()
        )
        
        content = "\n".join([doc.page_content for doc in documents])
        return conversation, content
    finally:
        os.unlink(temp_file_path)

def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.conversation = None
    st.session_state.document_content = ""
    st.session_state.queries_left = 10
    st.success("Chat history cleared!")

def get_ai_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Respond in a natural, conversational manner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=None  # Remove token limit to avoid truncation
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request. Please try again later."

def continuous_listening():
    with sr.Microphone() as source:
        while st.session_state.voice_ai_active:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                audio_queue.put(audio)
            except sr.WaitTimeoutError:
                pass

def transcribe_audio(audio):
    try:
        audio_data = audio.get_wav_data()
        result = client.audio.transcriptions.create(model="whisper-1", file=("audio.wav", audio_data))
        return result.text.strip().lower()
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return ""

def handle_conversation(text, is_voice=False):
    if text.lower() == EXIT_KEYWORD:
        st.session_state.voice_ai_active = False
        response = "Goodbye! Voice AI is now deactivated."
        if is_voice:
            speak(response)
        return response

    if st.session_state.queries_left <= 0:
        response = "You have reached the maximum number of queries. Please clear the chat history to continue."
        if is_voice:
            speak(response)
        return response

    st.session_state.last_user_input_time = time.time()
    
    if st.session_state.conversation:
        chat_history = []
        for msg in st.session_state.chat_history:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            else:
                chat_history.append(HumanMessage(content=msg[0]))
                chat_history.append(AIMessage(content=msg[1]))
        
        response = st.session_state.conversation.invoke({"question": text, "chat_history": chat_history})['answer']
    else:
        context = "\n".join([f"{msg['role'] if isinstance(msg, dict) else msg[0]}: {msg['content'] if isinstance(msg, dict) else msg[1]}" for msg in st.session_state.chat_history[-5:]])
        prompt = f"""
        Previous conversation:
        {context}

        Document content:
        {st.session_state.document_content}

        Human: {text}
        AI: Let's respond in a natural, conversational manner, taking into account the document content if relevant.
        """
        response = get_ai_response(prompt)
    
    if is_voice:
        speak(response)
    
    st.session_state.chat_history.append({"role": "user", "content": text})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.queries_left -= 1
    
    # Trigger UI update
    st.rerun()
    
    return response

def process_audio_queue():
    while st.session_state.voice_ai_active:
        try:
            audio = audio_queue.get(timeout=1)
            text = transcribe_audio(audio)
            if text:
                response = handle_conversation(text, is_voice=True)
        except queue.Empty:
            pass

def generate_podcast_script(file_contents: str) -> str:
    try:
        prompt = f"""
        Create a podcast script for two people, Alex and Jamie, discussing the following content contextually. 
        The discussion should be engaging, informative, and conversational.

        Content:
        {file_contents}

        Script:
        """
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a professional podcast scriptwriter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=None  # Remove token limit to avoid truncation
        )
        raw_script = response.choices[0].message.content
        return process_script(raw_script)
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        return f"Error: Unable to generate script. Please try again or contact support."

def process_script(script: str) -> str:
    processed_lines = []
    for line in script.split('\n'):
        line = line.strip()
        line = line.replace('*', '')
        if line.startswith("Alex:") or line.startswith("Jamie:"):
            processed_lines.append(line)
    return '\n'.join(processed_lines)

def text_to_speech_stream(text: str, voice: str):
    try:
        logger.info(f"Starting text-to-speech conversion for voice: {voice}")
        logger.info(f"Text to convert: {text[:50]}...")  # Log first 50 characters
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text
        )
        logger.info(f"Text-to-speech conversion successful for voice: {voice}")
        return response.content
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion for voice {voice}: {str(e)}")
        return None

def create_podcast(script: str) -> Optional[AudioSegment]:
    lines = script.split('\n')
    audio_segments: List[AudioSegment] = []
    
    def process_line(line: str):
        if line.startswith("Alex:"):
            return text_to_speech_stream(line[5:].strip(), voice="onyx")
        elif line.startswith("Jamie:"):
            return text_to_speech_stream(line[6:].strip(), voice="nova")
        return None
    
    logger.info(f"Starting audio generation for {len(lines)} lines")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        audio_streams = list(executor.map(process_line, lines))
    
    for i, (line, audio_content) in enumerate(zip(lines, audio_streams)):
        if audio_content:
            try:
                voice = "onyx" if line.startswith("Alex:") else "nova"
                logger.info(f"Processing line {i+1}/{len(lines)} for voice: {voice}")
                logger.info(f"Generating new audio for line {i+1}")
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
                audio_segments.append(audio_segment)
                logger.info(f"Audio generated for line {i+1}")
            except Exception as e:
                logger.error(f"Error processing audio stream for line {i+1}: {line}")
                logger.error(str(e))
        else:
            logger.warning(f"No audio stream generated for line {i+1}: {line}")
    
    if audio_segments:
        try:
            logger.info(f"Combining {len(audio_segments)} audio segments")
            final_audio = sum(audio_segments)
            logger.info(f"Final audio duration: {len(final_audio)/1000:.2f} seconds")
            return final_audio
        except Exception as e:
            logger.error(f"Error combining audio segments: {str(e)}")
    else:
        logger.error("No audio segments were generated")
    
    return None

def main():
    st.title("AI Assistant")

    tab1, tab2 = st.tabs(["Conversational AI", "Podcast Generator"])

    with tab1:
        st.header("Chat with AI")
        st.write(f"Queries left: {st.session_state.queries_left}")
        uploaded_file = st.file_uploader("Upload a file (PDF, TXT, CSV)", type=['pdf', 'txt', 'csv'])

        if uploaded_file is not None:
            st.session_state.conversation, file_contents = process_uploaded_file(uploaded_file)
            if st.session_state.conversation:
                st.success("File processed and added to knowledge base successfully!")
                st.text_area("File Contents:", value=file_contents, height=300)
                st.session_state.document_content = file_contents

        if st.button("Clear Chat History"):
            clear_chat_history()

        for message in st.session_state.chat_history:
            role = message["role"] if isinstance(message, dict) else message[0]
            content = message["content"] if isinstance(message, dict) else message[1]
            with st.chat_message(role):
                st.markdown(content)

        prompt = st.chat_input("What is your question?")
        
        voice_ai_active = st.toggle("Activate Voice AI", key="voice_ai_toggle")

        if voice_ai_active != st.session_state.voice_ai_active:
            st.session_state.voice_ai_active = voice_ai_active
            if voice_ai_active:
                st.session_state.listening_thread = threading.Thread(target=continuous_listening, daemon=True)
                st.session_state.processing_thread = threading.Thread(target=process_audio_queue, daemon=True)
                add_script_run_ctx(st.session_state.listening_thread)
                add_script_run_ctx(st.session_state.processing_thread)
                st.session_state.listening_thread.start()
                st.session_state.processing_thread.start()
                welcome_message = "Voice AI is now active. How can I assist you today?"
                speak(welcome_message)
                st.session_state.chat_history.append({"role": "assistant", "content": welcome_message})
            else:
                if hasattr(st.session_state, 'listening_thread'):
                    st.session_state.listening_thread.join()
                if hasattr(st.session_state, 'processing_thread'):
                    st.session_state.processing_thread.join()

        if voice_ai_active:
            st.write("Voice AI is active. Speak to interact or say 'Goodbye Assistant' to exit.")
        else:
            st.write("Voice AI is inactive. Toggle the switch above to activate.")

        if prompt:
            if st.session_state.queries_left > 0:
                response = handle_conversation(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.queries_left -= 1
            else:
                st.error("You have reached the maximum number of queries. Please clear the chat history to continue.")

    with tab2:
        st.header("Podcast Generator")
        podcast_file = st.file_uploader("Upload a file for podcast generation", type=["txt", "pdf", "csv"])

        if podcast_file is not None:
            file_contents = read_file_content(podcast_file)
            
            if file_contents.startswith("Error:"):
                st.error(file_contents)
            else:
                if st.button("Generate Podcast"):
                    with st.spinner("Generating podcast script..."):
                        script = generate_podcast_script(file_contents)
                    
                    if script.startswith("Error:"):
                        st.error(script)
                    else:
                        st.subheader("Generated Script")
                        st.text_area("Script", script, height=300)
                        
                        with st.spinner("Creating podcast audio..."):
                            try:
                                podcast_audio = create_podcast(script)
                            except Exception as e:
                                st.error("An error occurred while creating the podcast. Please try again or contact support.")
                                logger.error(f"Podcast creation error: {str(e)}")
                                podcast_audio = None
                        
                        if podcast_audio:
                            st.subheader("Generated Podcast")
                            with io.BytesIO() as audio_buffer:
                                try:
                                    podcast_audio.export(audio_buffer, format="mp3")
                                    audio_buffer.seek(0)
                                    st.audio(audio_buffer, format="audio/mp3")
                                
                                    st.download_button(
                                        label="Download Podcast",
                                        data=audio_buffer.getvalue(),
                                        file_name="ai_generated_podcast.mp3",
                                        mime="audio/mpeg"
                                    )
                                except Exception as e:
                                    st.error("Error exporting audio. Please try again or contact support.")
                                    logger.error(f"Audio export error: {str(e)}")
                        else:
                            st.error("Failed to generate podcast audio. Please try again or contact support.")

    st.markdown("---")
    st.markdown("Created by NextQAI")

if __name__ == "__main__":
    main()
