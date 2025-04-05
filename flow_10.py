from langgraph.graph import Graph
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field

# Schema for structured output to use in evaluation
class InstrospectPiece(BaseModel):
    reasoning: str = Field(
        description="An element of reasoning associated to the answer piece.",
    )
    answer: str = Field(
        description="The answer piece associated to the reasoning elements.",
    )


# Define node functions
def initial_response_node(state):
    prompt = f"""Given the following prompt, provide your reasoning and then your final answer. 
    Use <think> and </think> tokens to denote your reasoning steps.\n\n
    Prompt: {state['prompt']}\n\n"""

    # Generate the initial response using the prompt chain
    initial_response = llm.invoke(prompt)
    state["initial_response"] = initial_response.content
    return state

def introspection_prompt_node(state):
    prompt = state["prompt"]

    initial_reflexion = state["initial_response"].split("</think>")[0].replace("<think>", "").replace("</think>", "").strip()
    initial_response = state["initial_response"].split("</think>")[1].replace("<think>", "").replace("</think>", "").strip()

    introspection_prompt = (
        f"Here is the original prompt: {prompt}\n\n"
        f"Here is the reasoning: {initial_reflexion}\n\n"
        f"Here is the answer: {initial_response}\n\n"
        "Introspect this reasoning. Make links between the reasoning information and the elements of answers.\n\n"
        "Use the given format to make links between the reasoning and answer pieces."
    )
    state["introspection_prompt"] = introspection_prompt
    return state

def introspection_node(state):
    introspection_prompt = state["introspection_prompt"]
    introspection = llm_introspect.invoke(introspection_prompt)
    state["introspection"] = introspection
    return state


import os
if __name__ == "__main__":

    os.environ["OPENAI_API_BASE_URL"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"] = "lm-studio"

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        temperature=0.5,
        model="deepseek-r1-distill-qwen-1.5b",
    )

    llm_introspect = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        temperature=0.5,
        model="hermes-3-llama-3.2-3b",
    )

    llm_introspect = llm_introspect.with_structured_output(InstrospectPiece)

    # Create and configure the graph
    graph = Graph()
    graph.add_node("initial_response", initial_response_node)
    graph.add_node("introspection_prompt", introspection_prompt_node)
    graph.add_node("introspection", introspection_node)
    
    graph.add_edge("initial_response", "introspection_prompt")
    graph.add_edge("introspection_prompt", "introspection")
    
    graph.set_entry_point("initial_response")
    graph.set_finish_point("introspection")

    # Compile the graph
    compiled_graph = graph.compile()

    # Execute the compiled graph
    initial_state = {"prompt": "if a train leaves the station at 3pm and travels at 60 mph, how far will it go in 2 hours?"}
    final_state = compiled_graph.invoke(initial_state)

    print("introspection:", final_state["introspection"])
