# Language Model Evaluation and Workflow Examples

This repository contains a collection of Python scripts demonstrating the use of language models (LLMs) with the `langchain` library. The scripts showcase various workflows, including simple LLM invocations, structured outputs, tool integration, and complex graph-based workflows using `langgraph`. These examples are designed to work with local LLM instances (e.g., via LM Studio or Ollama).

## Prerequisites

- **Python 3.8+**
- **Dependencies**:
  - `langchain-openai`
  - `langchain-ollama`
  - `langgraph`
  - `pydantic`
  - `typing-extensions`
  - (Optional) `IPython` for displaying graphs in some scripts

Install dependencies using:
```bash
pip install langchain-openai langchain-ollama langgraph pydantic typing-extensions IPython
```

Local LLM Setup:
- For ChatOpenAI examples: Run a local LLM server (e.g., LM Studio) at http://localhost:1234/v1 with API key lm-studio.
- For OllamaLLM examples: Install and run Ollama with the llama3.2:latest model.

## Files Overview

1. **eval_lmstudio.py**  
   **Purpose:** Basic example of invoking a local LLM using ChatOpenAI to answer a medical-related question.  
   **Features:** Configures environment variables for a local LLM server and queries the model.  
   **Example Usage:**
   ```bash
   python eval_lmstudio.py
   ```
   **Output:** Response to "How does Calcium CT score relate to high cholesterol?"

2. **eval_ollama.py**  
   **Purpose:** Simple demonstration of invoking an Ollama-hosted LLM (llama3.2:latest).  
   **Features:** Queries the model to complete a sentence.  
   **Example Usage:**
   ```bash
   python eval_ollama.py
   ```
   **Output:** Completion of "The first man on the moon was ..."

3. **flow_1.py**  
   **Purpose:** Introduces structured output and tool binding with OllamaLLM.  
   **Features:** Generates a search query with justification and demonstrates a multiplication tool.  
   **Example Usage:**
   ```bash
   python flow_1.py
   ```

4. **flow_1b.py**  
   **Purpose:** Similar to flow_1.py but uses ChatOpenAI instead of OllamaLLM.  
   **Features:** Structured output and tool usage with a local LLM server.  
   **Example Usage:**
   ```bash
   python flow_1b.py
   ```

5. **flow_2.py**  
   **Purpose:** Demonstrates a StateGraph workflow to generate and improve a joke.  
   **Features:** Conditional edges based on punchline detection, multiple LLM calls for refinement.  
   **Example Usage:**
   ```bash
   python flow_2.py
   ```

6. **flow_3.py**  
   **Purpose:** Parallel workflow generating a joke, story, and poem about a topic.  
   **Features:** Uses StateGraph to run tasks concurrently and aggregates results.  
   **Example Usage:**
   ```bash
   python flow_3.py
   ```

7. **flow_4.py**  
   **Purpose:** Dynamic routing workflow using structured output to decide between poem, story, or joke.  
   **Features:** Conditional routing with OllamaLLM.  
   **Example Usage:**
   ```bash
   python flow_4.py
   ```

8. **flow_4b.py**  
   **Purpose:** Variant of flow_4.py using ChatOpenAI.  
   **Example Usage:**
   ```bash
   python flow_4b.py
   ```

9. **flow_4c.py**  
   **Purpose:** Simplified routing workflow using langgraph.func decorators.  
   **Features:** Task-based routing with ChatOpenAI.  
   **Example Usage:**
   ```bash
   python flow_4c.py
   ```

10. **flow_5.py**  
    **Purpose:** Tool-augmented LLM workflow for arithmetic operations.  
    **Features:** Uses StateGraph with conditional edges to handle tool calls.  
    **Example Usage:**
    ```bash
    python flow_5.py
    ```

11. **flow_6.py**  
    **Purpose:** Alternative tool-augmented workflow using langgraph.func.  
    **Features:** Streams updates during execution.  
    **Example Usage:**
    ```bash
    python flow_6.py
    ```

12. **flow_7.py**  
    **Purpose:** Multi-step report generation workflow.  
    **Features:** Plans sections, generates content, and synthesizes a final report.  
    **Example Usage:**
    ```bash
    python flow_7.py
    ```

13. **flow_8.py**  
    **Purpose:** Joke optimization workflow with feedback loop.  
    **Features:** Generates and evaluates jokes until a "funny" one is produced.  
    **Example Usage:**
    ```bash
    python flow_8.py
    ```

14. **flow_9.py**  
    **Purpose:** Question-answering with fact-checking.  
    **Features:** Generates an answer and evaluates its accuracy.  
    **Example Usage:**
    ```bash
    python flow_9.py
    ```

15. **flow_10.py**  
    **Purpose:** Demonstrates ReAct-style tool usage with a local LLM using ChatOpenAI and `langgraph`.  
    **Features:** Incorporates a custom `get_weather` tool, shows graph rendering, and prints streaming outputs for different user queries.  
    **Example Usage:**
    ```bash
    python flow_10.py
    ```
    **Output:** Weather information for specific cities and LLM responses to general questions.


## Usage Notes

- Ensure your local LLM server (e.g., LM Studio or Ollama) is running before executing scripts.
- Modify model names or API endpoints in the scripts as needed to match your setup.
- Some scripts (e.g., flow_3.py, flow_4.py) include commented-out code to visualize workflows using `IPython.display`. Uncomment and install IPython to use this feature.

## License

This project is unlicensed and provided as-is for educational purposes.

This `README.md` provides a clear structure, explains each file's purpose, and includes setup instructions, making it easy for users to understand and run the code.
