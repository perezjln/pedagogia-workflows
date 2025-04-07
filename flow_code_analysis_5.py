import os
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
    # Le dictionnaire mappe le nom de l'aspect à la description du prompt pour l'analyse
    aspects: Dict[str, str]
    # Le résultat de chaque branche sera fusionné dans ce dictionnaire
    aspect_outputs: Annotated[Dict[str, str], merge_dicts]
    combined_output: str

# --- Définition des tâches ---

@task
def analyzer(rendu_code: str, aspect: str, prompt_description: str) -> AnalysisResult:
    prompt_system = SystemMessage(
         content=(
           "Vous êtes un assistant chargé d'analyser le rendu de code d'un étudiant. "
           "Le rendu de code, l'aspect à analyser et des instructions spécifiques pour cet aspect sont fournis. "
           "Merci de structurer votre réponse en deux parties : "
           "une partie raisonnement entre <think> et </think> et une partie réponse."
         )
    )
    input_text = (
         f"Voici le rendu de code de l'étudiant:\n{rendu_code}\n\n"
         f"Analysez cet ensemble en vous concentrant sur l'aspect: '{aspect}'.\n"
         f"Instructions spécifiques pour cet aspect:\n{prompt_description}"
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
           "Le raisonnement et le résultat de l'analyse déjà effectuée sont fournis. "
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
    prompt_description = state["prompt_description"]
    analysis = analyzer(state["rendu_code"], aspect, prompt_description).result()
    synthesis = synthesizer(analysis).result()
    # Retourne le résultat sous forme d'un dictionnaire dont la clé est l'aspect
    return {"aspect_outputs": {aspect: synthesis.answer}}

# --- Nœud fan-out : crée un Send pour chaque aspect sélectionné ---
def fanout_process_aspects(state: State) -> List[Send]:
    sends = []
    # Itération sur le dictionnaire d'aspects sélectionnés
    for aspect, prompt_description in state["aspects"].items():
        sends.append(Send("process_aspect", {
            "aspect": aspect, 
            "rendu_code": state["rendu_code"],
            "prompt_description": prompt_description
        }))
    return sends

# --- Nœud agrégateur : combine les résultats des branches ---
def aggregator_node(state: State) -> dict:
    combined = "Rapport Final de l'Analyse du Rendu de Code\n\n"
    for aspect, answer in state["aspect_outputs"].items():
        combined += f"--- Aspect: {aspect} ---\nRéponse: {answer}\n\n"
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
    # Dictionnaire complet regroupant de nombreux aspects
    all_aspects = {
         "lisibilité": "Analysez la clarté, la structure, et la cohérence du code.",
         "performance": "Évaluez l'efficacité du code, l'optimisation des boucles et l'utilisation des ressources.",
         "maintenabilité": "Examinez la modularité, la facilité de modification, et l'extensibilité du code.",
         "documentation": "Vérifiez la présence et la qualité de la documentation, ainsi que la clarté des commentaires.",
         "testabilité": "Analysez la couverture des tests et la facilité de mise en place d'une stratégie de test.",
         "sécurité": "Évaluez les vulnérabilités potentielles, la gestion des exceptions et les accès sécurisés.",
         "architecture": "Examinez la structure globale du projet, la séparation des responsabilités et la scalabilité.",
         "conception": "Analysez les choix de conception, les patterns utilisés et la cohérence de l'implémentation.",
         "compatibilité": "Vérifiez la compatibilité avec divers environnements et la portabilité du code."
    }
    # Sélection de quelques aspects pour l'analyse courante.
    # Par exemple, nous choisissons d'analyser uniquement la lisibilité, la performance et la maintenabilité.
    selected_keys = {"lisibilité", "performance", "maintenabilité"}
    selected_aspects = {aspect: prompt for aspect, prompt in all_aspects.items() if aspect in selected_keys}

    initial_state: State = {
         "rendu_code": sample_rendu_code,
         "aspects": selected_aspects,
         "aspect_outputs": {},  # initialisé vide; fusionné via merge_dicts
         "combined_output": ""
    }

    state = compiled_workflow.invoke(initial_state)
    print(state["combined_output"])
