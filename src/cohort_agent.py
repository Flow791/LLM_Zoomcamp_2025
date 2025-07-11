#%%
import random
import os
from mistralai import Mistral
from chat_assistant import ChatAssistant, ChatInterface, Tools

# %%
known_weather_data = {
    'berlin': 20.0
}

def get_weather(city: str) -> float:
    city = city.strip().lower()

    if city in known_weather_data:
        return known_weather_data[city]

    return round(random.uniform(-5, 35), 1)
# %%
get_weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather temperature for a given city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to get weather for"
            }
        },
        "required": ["city"],
        "additionalProperties": False
    }
}

# %%
# Configuration du client Mistral
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

#%%
# Création de l'assistant avec les outils
tools = Tools()
tools.add_tool(get_weather, "Get weather for a city")

assistant = ChatAssistant(
    tools=tools, 
    developer_prompt="You are a helpful assistant that can answer questions about the weather.", 
    chat_interface=ChatInterface(), 
    client=client
)

# Test avec une question
question = "Quelle est la météo à Paris?"
response = assistant.run(question)
print(f"Question: {question}")
print(f"Réponse: {response}")
# %%

