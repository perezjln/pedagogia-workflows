from typing import List
from typing_extensions import Literal

import os
from langchain_openai import ChatOpenAI


from langgraph.func import entrypoint, task
from pydantic import BaseModel, Field


# Schema for structured output to use in evaluation
class Answer(BaseModel):
    question: str = Field(
        description="store the question that is asked.",
    )
    answer: str = Field(
        description="the answer to the question.",
    )


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["true", "false"] = Field(
        description="Fact check the given sentence.",
    )
    feedback: str = Field(
        description="Explain why it is true or false.",
    )


# Nodes
@task
def llm_call_generator(topic: str):
    """LLM generates a joke"""
    answer = answerer.invoke(f"Answer the following question: {topic}")
    return answer


@task
def llm_call_evaluator(answer: str):
    """LLM evaluates the joke"""
    feedback = evaluator.invoke(f"Fact check the following fact : {answer}")
    return feedback


@entrypoint()
def optimizer_workflow(topic: str):
    answer = llm_call_generator(topic).result()
    feedback = llm_call_evaluator(answer.answer).result()
    return (answer, feedback)

if __name__ == "__main__":

    os.environ["OPENAI_API_BASE_URL"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"]      = "lm-studio"

    llm1 = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        temperature=0.5,
    )

    llm2 = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
        temperature=0.5,
    )

    # Augment the LLM with schema for structured output
    answerer = llm1.with_structured_output(Answer,method='function_calling')
    evaluator = llm2.with_structured_output(Feedback,method='function_calling')

    # Invoke
    idx = 1
    for step in optimizer_workflow.stream("What is the capital of france, make a complete answer and fill all the slots.", stream_mode="updates"):
        print(f"{idx} - {step}")
        print("\n")
        idx += 1
