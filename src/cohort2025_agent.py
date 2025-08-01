# %%
import random
import os
from openai import OpenAI
from chat_assistant import ChatAssistant, ChatInterface, Tools
from fastmcp import FastMCP
import threading
import subprocess
import json
import time

# %%
known_weather_data = {
    'berlin': 20.0
}

def get_weather(city: str) -> float:
    city = city.strip().lower()

    if city in known_weather_data:
        return known_weather_data[city]

    return round(random.uniform(-5, 35), 1)

def run_chat():
    chat_interface = ChatInterface()

    chat = ChatAssistant(
        tools=tools,
        developer_prompt="You are a helpful assistant that can answer questions about the weather.",
        chat_interface=chat_interface,
        client=client
    )

    chat.run()
    
    
api_key = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=api_key)

tools = Tools()

# %%
##Question 1:
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
tools.add_tool(get_weather, get_weather_tool)
run_chat()

# %%
def set_weather(city: str, temp: float) -> None:
    city = city.strip().lower()
    known_weather_data[city] = temp
    return 'OK'

# %%
##Question 2:
set_weather_tool = {
    "type": "function",
    "name": "set_weather",
    "description": "Set the given weather temperature for a given city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to set weather for"
            },
            "temp": {
                "type": "number",
                "description": "The temperature to set for the city"
            }
        },
        "required": ["city", "temp"],
        "additionalProperties": False
    }
}

tools.add_tool(set_weather, set_weather_tool)
run_chat()

# %%
##Question 3:
import fastmcp
print(fastmcp.__version__)

# %%
##Question 4:
#See mcp_server.py
#Launch it with terminal command:
!python mcp_server.py

# %%
proc = subprocess.Popen(
    ["python", "weather_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

def send_msg(msg):
    json_msg = json.dumps(msg)
    proc.stdin.write(json_msg + "\n")
    proc.stdin.flush()
    print(f">>> Envoyé : {json_msg}")
    time.sleep(0.2)
    response = proc.stdout.readline()
    print(f"<<< Reçu : {response.strip()}")
    return response

# %%
##Question 5:
! cat mcp_server_tools_test.txt | python mcp_server.py

# %%
#Question 6:
from fastmcp import Client
import mcp_server 

async def main():
    async with Client(mcp_server.mcp) as mcp_client:
        tools = await mcp_client.list_tools()
        for tool in tools:
            print(tool)

import asyncio
await main()
# %%
