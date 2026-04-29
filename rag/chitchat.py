import os
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
import asyncio
CHITCHAT_KEYWORDS = [
    "hi", "hello", "hey", "good morning", "good evening",
    "how are you", "what's up", "whats up", "good afternoon",
    "who are you", "your name", "nice to meet you",
    "how is it going", "how are things",
    "thank you", "thanks", "bye", "goodbye", "good night" , "see you", "talk to you later",
    "congratulations", "congrats", "well done", "happy birthday", "happy anniversary",
    "how do you do", "pleased to meet you", "greetings", "salutations", "welcome",
    "how can you help me", "what can you do", "do you have any hobbies", "what are your interests",
    "what can you help me with", "can you tell me a joke", "can you tell me something interesting", 
    "what's your favorite color", "how can you help", "what do you do", "what is your purpose", "how can you assist me"
]

def is_chitchat(message: str) -> bool:
    msg = message.lower().strip()
    return any(k in msg for k in CHITCHAT_KEYWORDS)


def classify_intent(question: str) -> str:
    """
    Returns one of: CHITCHAT, KNOWLEDGE
    """

    llm = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),)
            

    prompt = f"""
    Classify the user's intent into one of the following categories:

    - CHITCHAT: greetings, small talk, casual conversation, social messages and anything which is not KNOWLEDGE related category.
    - KNOWLEDGE: questions seeking factual or document-based information

    Respond with only one word: CHITCHAT or KNOWLEDGE.

    User message:
    "{question}"
    """

    response = llm.chat.completions.create(
        model="gpt4o",
        messages=[{"role": "user", "content": prompt}])

    if response.choices[0].message.content.strip().upper() not in {"CHITCHAT", "KNOWLEDGE"}:
        return "KNOWLEDGE"  # safe default

    return response.choices[0].message.content.strip().upper()


def chitchat_answer(question: str) -> str:
    llm = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))

    system_prompt = (
        "You are a friendly assistant. Keep answers very short and helpful. "
        "Do NOT offer scientific, laboratory, or chemical advice. "
        "Only handle social conversation like greetings or small talk. " \
        "Do not discuss about any country or politics or conflict"\
        "Your name is ELN help agent"\
        "if use asks about how can you help, say you can answer questions about ELN and lab work." \
        "If you are not sure how to answer, respond that you are not sure and suggest the user to ask questions about ELN or lab work."
    )

    user_prompt = f"User said: {question}"
    response =  llm.chat.completions.create(
        
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.6
    )

    return response.choices[0].message.content.strip()
