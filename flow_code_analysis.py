from typing import List
import os
from langchain_openai import ChatOpenAI

from langgraph.func import entrypoint, task
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Schéma pour la sortie structurée de l'analyse
class AnalysisResult(BaseModel):
    reasoning: str = Field(
        description="Le raisonnement de l'analyse, contenu entre <think> et </think>."
    )
    answer: str = Field(
        description="La réponse finale de l'analyse."
    )

# Task qui appelle le LLM pour analyser le rendu de code de l'étudiant
@task
def analyzer(rendu_code: str, aspect: str) -> AnalysisResult:
    prompt_system = SystemMessage(
         content=(
           "Vous êtes un assistant chargé d'analyser le rendu de code d'un étudiant. "
           "Le rendu de code et l'aspect à analyser sont fournis. "
           "Merci de structurer votre réponse en deux parties : "
           "une partie raisonnement entre <think> et </think> et une partie réponse."
         )
    )
    input_text = (
         f"Voici le rendu de code de l'étudiant:\n{rendu_code}\n\n"
         f"Analysez cet ensemble en vous concentrant sur l'aspect: {aspect}."
    )
    messages = [prompt_system, HumanMessage(content=input_text)]
    
    # Appel du LLM
    result = llm.invoke(messages)
    content = result.content

    # Extraction robuste du raisonnement et de la réponse
    parts = content.split("</think>")
    if len(parts) <= 1:
         return AnalysisResult(reasoning="", answer=content.strip())
    reasoning = parts[0].replace("<think>", "").strip()
    answer = parts[1].strip()
    return AnalysisResult(reasoning=reasoning, answer=answer)

# Task de synthèse pour formater la réponse finale
@task
def synthesizer(analysis: AnalysisResult) -> AnalysisResult:    
    prompt_system = SystemMessage(
         content=(
           "Vous êtes un assistant chargé d'analyser le rendu de code d'un étudiant. "
           "Le rendu de code et l'analyse déja effectuée sont fournis. "
           "effectue un résumé de l'analyse."
         )
    )
    input_text = (
         f"Voici le raisonnement opéré durant l'analyse:\n{analysis.reasoning}\n\n"
         f"Voici le résultat de l'analyse sur le code étudiant: {analysis.answer}."
    )
    messages = [prompt_system, HumanMessage(content=input_text)]
    
    # Appel du LLM
    result = llm.invoke(messages)
    content = result.content

    # Extraction robuste du raisonnement et de la réponse
    parts = content.split("</think>")
    if len(parts) <= 1:
         return AnalysisResult(reasoning="", answer=content.strip())
    reasoning = parts[0].replace("<think>", "").strip()
    answer = parts[1].strip()
    return AnalysisResult(reasoning=reasoning, answer=answer)

# Entrypoint de l'orchestrateur
@entrypoint()
def orchestrator_worker(rendu_code: str):
    analysis_result = analyzer(rendu_code, aspect).result()
    final_report = synthesizer(analysis_result).result()
    return (analysis_result, final_report)

if __name__ == "__main__":
    
    os.environ["OPENAI_API_BASE_URL"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"]      = "lm-studio"

    llm = ChatOpenAI(
         api_key=os.getenv("OPENAI_API_KEY"),
         base_url=os.getenv("OPENAI_API_BASE_URL"),
         temperature=0.5,
    )

    # Exemple d'entrée : rendu de code de l'étudiant et aspect à analyser
    rendu_code = (
         "Fichier main.py:\n"
         "def main():\n"
         "    print('Hello World')\n\n"
         "Fichier utils.py:\n"
         "def helper():\n"
         "    return 42\n"
    )
    aspect = "lisibilité"

    analysis_result, final_report = orchestrator_worker.invoke(rendu_code)
    print(final_report)
