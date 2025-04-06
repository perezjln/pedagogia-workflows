from langchain_ollama import OllamaLLM

if __name__ == "__main__":
    llm = OllamaLLM(model="llama3.2:latest")
    llm.invoke("The first man on the moon was ...")
