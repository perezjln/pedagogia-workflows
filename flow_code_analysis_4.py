import os
import operator
from typing import Dict, List, Any
from typing_extensions import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langgraph.func import entrypoint, task
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# --- Définition des modèles de données ---

# Schéma pour la sortie structurée de l'analyse
class AnalysisResult(BaseModel):
    reasoning: str = Field(
        description="Le raisonnement de l'analyse, contenu entre <think> et </think>."
    )
    answer: str = Field(
        description="La réponse finale de l'analyse."
    )

# Opérateur personnalisé pour fusionner des dictionnaires
def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    return {**dict1, **dict2}

# Définition de l'état du workflow
class State(TypedDict):
    rendu_code: str
    aspects: List[str]
    # Le résultat de chaque branche sera fusionné dans ce dictionnaire
    aspect_outputs: Annotated[Dict[str, str], merge_dicts]
    combined_output: str

# --- Définition des tâches ---

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

    parts = content.split("</think>")
    if len(parts) <= 1:
         return AnalysisResult(reasoning="", answer=content.strip())
    reasoning = parts[0].replace("<think>", "").strip()
    answer = parts[1].strip()
    return AnalysisResult(reasoning=reasoning, answer=answer)

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

# --- Nœud de traitement d'un aspect (analyse + synthèse) ---
def process_aspect_node(state: Dict[str, Any]) -> dict:
    aspect = state["aspect"]
    analysis = analyzer(state["rendu_code"], aspect).result()
    synthesis = synthesizer(analysis).result()
    # Retourne le résultat sous forme d'un dictionnaire dont la clé est l'aspect
    return {"aspect_outputs": {aspect: synthesis.answer}}

# --- Nœud fan-out : crée un Send pour chaque aspect ---
def fanout_process_aspects(state: State) -> List[Send]:
    return [Send("process_aspect", {"aspect": a, "rendu_code": state["rendu_code"]})
            for a in state["aspects"]]

# --- Nœud agrégateur : combine les résultats des branches ---
def aggregator_node(state: State) -> dict:
    combined = "Rapport Final de l'Analyse du Rendu de Code\n\n"
    for aspect, answer in state["aspect_outputs"].items():
        combined += f"--- Aspect: {aspect} ---\n  Réponse: {answer}\n\n"
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

    # --- Construction du workflow ---
    # Le workflow utilise un fan-out pour lancer en parallèle l'analyse sur chaque aspect.
    workflow = StateGraph(State)
    workflow.add_node("fanout", lambda state: {})  # nœud vide pour déclencher le fan-out
    workflow.add_conditional_edges("fanout", fanout_process_aspects, ["process_aspect"])
    workflow.add_node("process_aspect", process_aspect_node)
    workflow.add_edge("process_aspect", "aggregator")
    workflow.add_node("aggregator", aggregator_node)

    # Relier les nœuds du workflow
    workflow.add_edge(START, "fanout")
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
         "aspect_outputs": {},  # initialisé vide; fusionné via merge_dicts
         "combined_output": ""
    }

    state = compiled_workflow.invoke(initial_state)
    print(state["combined_output"])
