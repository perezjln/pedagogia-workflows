from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import re
from typing import Optional
import logging

# State model
class State(BaseModel):
    prompt: str = Field(description="The initial user prompt")
    initial_response: str = Field(default="", description="Raw response from the initial LLM")
    reasoning: str = Field(default="", description="Extracted reasoning")
    answer: str = Field(default="", description="Extracted answer")
    introspection_prompt: str = Field(default="", description="Prompt for introspection")
    introspection: dict | None = Field(default=None, description="Structured introspection output")

# Structured output schema
class Relationship(BaseModel):
    description: str = Field(description="How a reasoning step relates to the answer")

class IntrospectionResult(BaseModel):
    relationships: list[Relationship] = Field(description="List of reasoning-answer relationships")

# Core functions
def generate_initial_response(prompt: str, llm) -> str:
    full_prompt = (
        "Provide your reasoning and final answer. "
        "Use <think> and </think> tags to denote reasoning steps.\n\n"
        f"Prompt: {prompt}"
    )
    return llm.invoke(full_prompt).content

def extract_reasoning_and_answer(response: str) -> tuple[str, str]:
    thinks = response.split("</think>")
    if len(thinks) <= 1:
        return "", response.strip()
    reasoning = thinks[0].replace("<think>", "").replace("</think>", "")
    answer    = thinks[1].replace("<think>", "").replace("</think>", "")
    return reasoning, answer

# Node functions
def initial_response_node(state: State, llm, logger: Optional[logging.Logger] = None) -> State:
    try:
        state.initial_response = generate_initial_response(state.prompt, llm)
        if logger:
            logger.info(f"Generated initial response: {state.initial_response}")
    except Exception as e:
        if logger:
            logger.error(f"Error in initial response: {e}")
        state.initial_response = "Error generating response"
    return state

def introspection_prompt_node(state: State) -> State:
    state.reasoning, state.answer = extract_reasoning_and_answer(state.initial_response)
    if not state.reasoning:
        state.introspection_prompt = (
            "The initial response lacks explicit reasoning. "
            "Analyze the answer based on the prompt alone."
        )
    else:
        state.introspection_prompt = (
            f"Original prompt: {state.prompt}\n"
            f"Reasoning: {state.reasoning}\n"
            f"Answer: {state.answer}\n"
            "Explain how each reasoning step contributes to the answer. "
            "Be specific about connections between reasoning and answer parts."
        )
    return state

def introspection_node(state: State, llm_introspect) -> State:
    introspection = llm_introspect.with_structured_output(IntrospectionResult).invoke(state.introspection_prompt)
    state.introspection = introspection.dict()
    return state

# Graph builder
def build_graph(llm, llm_introspect) -> Graph:
    graph = Graph()
    graph.add_node("initial_response", lambda state: initial_response_node(state, llm))
    graph.add_node("introspection_prompt", introspection_prompt_node)
    graph.add_node("introspection", lambda state: introspection_node(state, llm_introspect))
    graph.add_edge("initial_response", "introspection_prompt")
    graph.add_edge("introspection_prompt", "introspection")
    graph.set_entry_point("initial_response")
    graph.set_finish_point("introspection")
    return graph.compile()

# Example usage
import os
if __name__ == "__main__":

    os.environ["OPENAI_API_BASE_URL"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"] = "lm-studio"

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        temperature=0.5,
        model="deepseek-r1-distill-qwen-7b",
    )

    llm_introspect = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        temperature=0.5,
        model="llama-3.2-3b-instruct",
    )
    
    compiled_graph = build_graph(llm, llm_introspect)
    
    # Example prompt
    prompt = "A triangle has sides of lengths 3, 4, and 5. Is it a right triangle?"
    initial_state = State(prompt=prompt)
    final_state = compiled_graph.invoke(initial_state)
    
    print(final_state)