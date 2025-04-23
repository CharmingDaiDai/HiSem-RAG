# HiSem-RAG

Hierarchical Semantic Retrieval-Augmented Generation system for power engineering question answering.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/CharmingDaiDai/HiSem-RAG.git
cd HiSem-RAG
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- numpy
- pydantic
- httpx
- openai
- PyYAML
- tqdm
- loguru

## Configuration

Before running the code, you need to configure the model API credentials in the `config.yaml` file:

1. **Local Embedding Model**:

   - Update the `base_url` to your embedding model URL
   - Make sure the `embedding_model` is properly set (default is "bge-m3")

   ```yaml
   models:
     local:
       api_key: "not empty"
       base_url: "your embedding model url"
       embedding_model: "bge-m3"
   ```
2. **LLM Model (Zhipu AI)**:

   - Set your Zhipu AI API key
   - The base URL is pre-configured as "https://open.bigmodel.cn/api/paas/v4/"
   - You can change the model name (default is "glm-4-flash")

   ```yaml
   models:
     zhipu:
       api_key: "your zhipu ai key"
       base_url: "https://open.bigmodel.cn/api/paas/v4/"
       model_name: "glm-4-flash"
   ```
3. **Select Active LLM**:

   - Set the `active_llm` parameter to choose which LLM to use (default is "zhipu")

   ```yaml
   models:
     active_llm: "zhipu"
   ```

## Running the Code

To run the HiSem-RAG system:

```bash
python main.py
```

Additional Configuration Options

The `config.yaml` file contains several other parameters you can customize:

- **Path Configuration**: Base directories for questions, documents, logs, and results
- **Retrieval Algorithm Parameters**: Thresholds, sensitivity coefficients, and search depth
- **System Parameters**: Threading configuration and logging levels
