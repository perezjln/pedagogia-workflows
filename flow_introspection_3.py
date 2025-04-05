from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Tuple
import logging
import re
import os

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntrospectionPattern")

# Modèles de données
class State(BaseModel):
    prompt: str = Field(description="Requête initiale de l'utilisateur")
    initial_response: str = Field(default="", description="Réponse brute de la première LLM")
    reasoning: str = Field(default="", description="Raisonnement extrait")
    answer: str = Field(default="", description="Réponse extraite")
    introspection_prompt: str = Field(default="", description="Prompt pour l'introspection")
    introspection: dict | None = Field(default=None, description="Résultat structuré de l'introspection")
    error: str | None = Field(default=None, description="Message d'erreur éventuel")

class Relationship(BaseModel):
    description: str = Field(description="Relation entre une étape de raisonnement et la réponse")

class IntrospectionResult(BaseModel):
    relationships: list[Relationship] = Field(description="Liste des relations raisonnement-réponse")
    comment: str | None = Field(default=None, description="Commentaire global sur la cohérence de la réponse")

    # Extraction robuste du raisonnement et de la réponse
def extract_reasoning_and_answer(response: str) -> tuple[str, str]:
    thinks = response.split("</think>")
    if len(thinks) <= 1:
        return "", response.strip()
    reasoning = thinks[0].replace("<think>", "").replace("</think>", "")
    answer    = thinks[1].replace("<think>", "").replace("</think>", "")
    return reasoning, answer

def generate_initial_response(prompt: str, llm) -> str:
    full_prompt = (
        "Fournis ton raisonnement et ta réponse finale. "
        "Utilise les balises <think> et </think> pour marquer les étapes de ton raisonnement.\n\n"
        f"Requête : {prompt}"
    )
    return llm.invoke(full_prompt).content

# Nœud de génération de la réponse initiale
def initial_response_node(state: State, llm) -> State:
    try:
        state.initial_response = generate_initial_response(state.prompt, llm)
        logger.info(f"Réponse initiale générée : {state.initial_response}")
    except Exception as e:
        state.error = f"Erreur lors de la génération : {str(e)}"
        state.initial_response = "Erreur lors de la génération"
        logger.error(state.error)
    return state

# Construction du prompt d'introspection avec des instructions plus précises
def introspection_prompt_node(state: State) -> State:
    state.reasoning, state.answer = extract_reasoning_and_answer(state.initial_response)
    logger.info(f"Raisonnement extrait : {state.reasoning or 'Aucun raisonnement trouvé'}")
    logger.info(f"Réponse extraite : {state.answer}")
    
    if not state.reasoning:
        state.introspection_prompt = (
            f"Requête initiale : {state.prompt}\n"
            f"Réponse : {state.answer}\n"
            "Aucun raisonnement explicite n'a été fourni. "
            "Analyse la réponse par rapport à la requête et explique sa pertinence."
        )
    else:
        state.introspection_prompt = (
            f"Requête initiale : {state.prompt}\n"
            f"Raisonnement complet :\n{state.reasoning}\n"
            f"Réponse finale : {state.answer}\n"
            "Pour chaque étape du raisonnement, explique précisément comment elle contribue à la réponse finale. "
            "Indique les liens logiques entre les différentes étapes et la cohérence globale."
        )
    logger.info(f"Prompt d'introspection construit : {state.introspection_prompt}")
    return state

# Nœud d'introspection avec gestion améliorée des erreurs et du format de sortie
def introspection_node(state: State, llm_introspect) -> State:
    try:
        introspection = llm_introspect.with_structured_output(IntrospectionResult).invoke(state.introspection_prompt)
        state.introspection = introspection.dict()
        logger.info(f"Résultat de l'introspection : {state.introspection}")
    except Exception as e:
        state.error = f"Erreur lors de l'introspection : {str(e)}"
        state.introspection = {"relationships": [], "comment": "Échec de l'introspection"}
        logger.error(state.error)
    return state

# Construction du graphe
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

# Affichage formaté des résultats
def format_output(state: State) -> str:
    output = ["### Résultat de l'introspection ###"]
    output.append(f"**Requête** : {state.prompt}")
    output.append(f"**Réponse initiale** : {state.initial_response}")
    output.append(f"**Raisonnement extrait** : {state.reasoning or 'Aucun'}")
    output.append(f"**Réponse extraite** : {state.answer}")
    
    if state.error:
        output.append(f"**Erreur** : {state.error}")
    
    if state.introspection:
        output.append("**Introspection** :")
        if state.introspection.get("relationships"):
            for rel in state.introspection["relationships"]:
                output.append(f"- {rel['description']}")
        if state.introspection.get("comment"):
            output.append(f"Commentaire : {state.introspection['comment']}")
    
    return "\n".join(output)

# Exemple d'utilisation
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
        model="llama-3.2-3b-instruct",
    )
    
    compiled_graph = build_graph(llm, llm_introspect)
    
    prompt = "Problème d'arithmétique : Une baignoire se remplit avec un robinet qui verse 12 litres d'eau par minute. En même temps, une fuite laisse échapper 4 litres d'eau par minute. Combien de temps faudra-t-il pour remplir complètement une baignoire de 120 litres ?"    
    initial_state = State(prompt=prompt)
    final_state = compiled_graph.invoke(initial_state)
    
    #print(format_output(final_state))

    # Affichage détaillé des relations introspectives
    print("\n--- Détails des relations introspectives ---")
    if final_state.introspection and final_state.introspection.get("relationships"):
        for i, rel in enumerate(final_state.introspection["relationships"], 1):
            print(f"Relation {i}: {rel['description']}")
    else:
        print("Aucune relation n'a été identifiée.")

    # Affichage du commentaire global sur la cohérence
    if final_state.introspection and final_state.introspection.get("comment"):
        print("\n--- Commentaire global ---")
        print(final_state.introspection["comment"])
