import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from googletrans import Translator

# Page config
st.set_page_config(page_title="SmolLM2 Korean Chat", page_icon="🤖")

# Title
st.title("🤖 SmolLM2-360M Korean Chat")
st.markdown("HuggingFaceTB/SmolLM2-360M-Instruct 모델과 Google Translate를 이용한 한국어 채팅 데모입니다.")

# Initialize translator
if 'translator' not in st.session_state:
    st.session_state.translator = Translator()

# Load model and tokenizer (cached)
@st.cache_resource
def load_model():
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

with st.spinner("모델을 불러오는 중입니다..."):
    tokenizer, model = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize model chat history (English) to maintain context
if "english_messages" not in st.session_state:
    st.session_state.english_messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("메시지를 입력하세요..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Translate user input to English
    try:
        translated_prompt = st.session_state.translator.translate(prompt, src='ko', dest='en').text
    except Exception as e:
        st.error(f"Translation error: {e}")
        translated_prompt = prompt # Fallback

    # Add translated user message to English history for context
    st.session_state.english_messages.append({"role": "user", "content": translated_prompt})
    
    # Generate response using the full English conversation history
    input_text = tokenizer.apply_chat_template(st.session_state.english_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    with st.spinner("생각 중..."):
        outputs = model.generate(
            inputs, 
            max_new_tokens=500, 
            temperature=0.2, 
            top_p=0.9, 
            do_sample=True
        )
        
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the assistant's response (remove the input part if present, though decode usually handles it, 
    # but apply_chat_template might leave the prompt in the generation depending on how it's used. 
    # Actually, generate returns the whole sequence usually. Let's slice it.)
    # A better way with chat models is to just decode the new tokens.
    
    new_tokens = outputs[0][inputs.shape[-1]:]
    english_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Add assistant response to English history
    st.session_state.english_messages.append({"role": "assistant", "content": english_response})

    # Translate response back to Korean
    try:
        korean_response = st.session_state.translator.translate(english_response, src='en', dest='ko').text
    except Exception as e:
        st.error(f"Translation error: {e}")
        korean_response = english_response # Fallback

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(korean_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": korean_response})