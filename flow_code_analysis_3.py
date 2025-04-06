import os
from typing import Dict, List, Optional
from typing_extensions import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langgraph.func import task
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# --- Définition des modèles de données ---

# Schéma pour la sortie structurée de l'analyse
class AnalysisResult(BaseModel):
    reasoning: str = Field(
        description="Le raisonnement de l'analyse, contenu entre <think> et </think>."
    )
    answer: str = Field(
        description="La réponse finale de l'analyse."
    )

# Type pour l'état du workflow avec plusieurs aspects
class State(TypedDict):
    rendu_code: str
    aspects: List[str]
    analysis_results: Optional[Dict[str, AnalysisResult]]
    final_reports: Optional[Dict[str, AnalysisResult]]
    combined_output: str

# --- Définition des tâches ---

# Task qui appelle le LLM pour analyser le rendu de code pour un aspect donné
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
    
    result = llm.invoke(messages)
    content = result.content

    # Extraction du raisonnement et de la réponse
    parts = content.split("</think>")
    if len(parts) <= 1:
         return AnalysisResult(reasoning="", answer=content.strip())
    reasoning = parts[0].replace("<think>", "").strip()
    answer = parts[1].strip()
    return AnalysisResult(reasoning=reasoning, answer=answer)

# Task de synthèse pour formater la réponse finale d'une analyse
@task
def synthesizer(analysis: AnalysisResult) -> AnalysisResult:
    prompt_system = SystemMessage(
         content=(
           "Vous êtes un assistant chargé d'analyser le rendu de code d'un étudiant. "
           "Le rendu de code et l'analyse déjà effectuée sont fournis. "
           "Effectuez un résumé de l'analyse."
         )
    )
    input_text = (
         f"Voici le raisonnement opéré durant l'analyse:\n{analysis.reasoning}\n\n"
         f"Voici le résultat de l'analyse sur le code étudiant: {analysis.answer}."
    )
    messages = [prompt_system, HumanMessage(content=input_text)]
    
    result = llm.invoke(messages)
    content = result.content

    parts = content.split("</think>")
    if len(parts) <= 1:
         return AnalysisResult(reasoning="", answer=content.strip())
    reasoning = parts[0].replace("<think>", "").strip()
    answer = parts[1].strip()
    return AnalysisResult(reasoning=reasoning, answer=answer)

# --- Nœuds pour le workflow en parallèle ---

def analyzer_node_parallel(state: State) -> dict:
    """
    Noeud qui exécute l'analyse pour chaque aspect en parallèle.
    Le résultat est stocké dans un dictionnaire associant chaque aspect à son analyse.
    """
    results = {}
    for aspect in state["aspects"]:
        # On appelle l'analyse pour chaque aspect
        results[aspect] = analyzer(state["rendu_code"], aspect).result()
    return {"analysis_results": results}

def synthesizer_node_parallel(state: State) -> dict:
    """
    Noeud qui synthétise les analyses déjà réalisées pour chaque aspect.
    Le résultat final de chaque branche est stocké dans 'final_reports'.
    """
    final_reports = {}
    for aspect, analysis in state["analysis_results"].items():
        final_reports[aspect] = synthesizer(analysis).result()
    return {"final_reports": final_reports}

def aggregator_node(state: State) -> dict:
    """
    Noeud qui combine les rapports finaux de toutes les analyses en un rapport global.
    """
    combined = "Rapport Final de l'Analyse du Rendu de Code\n\n"
    for aspect, report in state["final_reports"].items():
        combined += f"--- Aspect: {aspect} ---\n"
        combined += f"Analyse initiale:\n  Raisonnement: {state['analysis_results'][aspect].reasoning}\n"
        combined += f"  Réponse: {state['analysis_results'][aspect].answer}\n"
        combined += f"Synthèse:\n  Raisonnement: {report.reasoning}\n"
        combined += f"  Réponse: {report.answer}\n\n"
    return {"combined_output": combined}

# --- Initialisation de l'API LLM ---
if __name__ == "__main__":
    
    os.environ["OPENAI_API_BASE_URL"] = "http://localhost:1234/v1"
    os.environ["OPENAI_API_KEY"]      = "lm-studio"

    llm = ChatOpenAI(
         api_key=os.getenv("OPENAI_API_KEY"),
         base_url=os.getenv("OPENAI_API_BASE_URL"),
         temperature=0.5,
    )

    # --- Construction du workflow en parallèle ---
    workflow = StateGraph(State)
    workflow.add_node("analyzer_parallel", analyzer_node_parallel)
    workflow.add_node("synthesizer_parallel", synthesizer_node_parallel)
    workflow.add_node("aggregator", aggregator_node)

    # Définition des arêtes pour exécuter en parallèle :
    # On commence par analyser tous les aspects, ensuite synthétiser, puis agréger.
    workflow.add_edge(START, "analyzer_parallel")
    workflow.add_edge("analyzer_parallel", "synthesizer_parallel")
    workflow.add_edge("synthesizer_parallel", "aggregator")
    workflow.add_edge("aggregator", END)
    compiled_workflow = workflow.compile()

    # --- Exemple d'entrée ---
    sample_rendu_code = (
         "Fichier main.py:\n"
         "def main():\n"
         "    print('Hello World')\n\n"
         "Fichier utils.py:\n"
         "def helper():\n"
         "    return 42\n"
    )
    sample_aspects = ["lisibilité", "performance", "maintenabilité"]

    initial_state: State = {
         "rendu_code": sample_rendu_code,
         "aspects": sample_aspects,
         "analysis_results": None,
         "final_reports": None,
         "combined_output": ""
    }

    state = compiled_workflow.invoke(initial_state)
    print(state["combined_output"])
