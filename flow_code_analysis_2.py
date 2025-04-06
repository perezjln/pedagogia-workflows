import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.func import task
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# --- Définition des modèles de données ---

# Schéma pour la sortie structurée de l'analyse
class AnalysisResult(BaseModel):
    reasoning: str = Field(
        description="Le raisonnement de l'analyse, contenu entre <think> et </think>."
    )
    answer: str = Field(
        description="La réponse finale de l'analyse."
    )

# Type pour l'état du workflow
class State(TypedDict):
    rendu_code: str
    aspect: str
    analysis_result: AnalysisResult
    final_report: AnalysisResult
    combined_output: str

# --- Définition des tâches ---

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
           "Le rendu de code et l'analyse déjà effectuée sont fournis. "
           "Effectuez un résumé de l'analyse."
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

# --- Définition des noeuds du workflow ---
def analyzer_node(state: State) -> dict:
    """
    Noeud qui lance l'analyse du rendu de code.
    """
    result = analyzer(state["rendu_code"], state["aspect"]).result()
    return {"analysis_result": result}

def synthesizer_node(state: State) -> dict:
    """
    Noeud qui synthétise le résultat de l'analyse.
    """
    result = synthesizer(state["analysis_result"]).result()
    return {"final_report": result}

def aggregator_node(state: State) -> dict:
    """
    Noeud qui combine les résultats de l'analyse et de la synthèse pour produire un rapport final.
    """
    combined = "Rapport Final de l'Analyse du Rendu de Code\n\n"
    combined += f"--- Analyse Initiale ---\nRaisonnement : {state['analysis_result'].reasoning}\n"
    combined += f"Réponse : {state['analysis_result'].answer}\n\n"
    combined += f"--- Synthèse ---\nRaisonnement : {state['final_report'].reasoning}\n"
    combined += f"Réponse : {state['final_report'].answer}"
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
    workflow.add_node("analyzer_node", analyzer_node)
    workflow.add_node("synthesizer_node", synthesizer_node)
    workflow.add_node("aggregator_node", aggregator_node)

    # Définition du graphe des dépendances
    workflow.add_edge(START, "analyzer_node")
    workflow.add_edge("analyzer_node", "synthesizer_node")
    workflow.add_edge("synthesizer_node", "aggregator_node")
    workflow.add_edge("aggregator_node", END)
    compiled_workflow = workflow.compile()

    # --- Invocation du workflow avec un exemple d'entrée ---
    sample_rendu_code = (
         "Fichier main.py:\n"
         "def main():\n"
         "    print('Hello World')\n\n"
         "Fichier utils.py:\n"
         "def helper():\n"
         "    return 42\n"
    )
    sample_aspect = "lisibilité"

    initial_state: State = {
         "rendu_code": sample_rendu_code,
         "aspect": sample_aspect,
         "analysis_result": None,  # sera rempli par le noeud analyzer_node
         "final_report": None,     # sera rempli par le noeud synthesizer_node
         "combined_output": ""
    }

    state = compiled_workflow.invoke(initial_state)
    print(state["analysis_result"])
    print(state["final_report"])
