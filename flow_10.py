import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

if __name__ == "__main__":

    os.environ["OPENAI_API_BASE_URL"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"]      = "lm-studio"

    model = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        temperature=0.5,
    )
    
    tools = [get_weather]
    graph = create_react_agent(model, tools=tools)
    display(Image(graph.get_graph().draw_mermaid_png()))
    
    inputs = {"messages": [("user", "what is the weather in sf")]}
    print_stream(graph.stream(inputs, stream_mode="values"))

    inputs = {"messages": [("user", "who built you?")]}
    print_stream(graph.stream(inputs, stream_mode="values"))