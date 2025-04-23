import openai
from loguru import logger
# from tqdm import tqdm


class EmbeddingTool:
    def __init__(self, api_key, base_url, model_name):
        """
        初始化 OpenAI 客户端
        Initialize OpenAI client

        :param api_key: OpenAI API 密钥 (OpenAI API key)
        :param base_url: OpenAI API 基础 URL (OpenAI API base URL)
        :param model_name: 使用的模型名称 (Model name to be used)
        """
        self.client = openai.Client(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def create_embedding(self, input_texts, batch_size=8):
        """
        创建嵌入向量
        Create embedding vectors

        :param input_texts: 输入文本列表 (List of input texts)
        :param batch_size: 每个批次的大小，默认为 8，经测试：文本长度很长时，batch_size过大可能会导致'Model not found'错误
                          (Batch size, default is 8. As tested: when text is very long, a large batch size may cause 'Model not found' errors)
        :return: 嵌入向量列表 (List of embedding vectors)
        """
        embeddings = []

        # 检查 batch_size 是否大于 8
        # Check if batch_size is greater than 8
        if batch_size > 8:
            logger.warning(
                "Batch size is greater than 8. This may cause 'Model not found' errors."
            )

        try:
            # for i in tqdm(range(0, len(input_texts), batch_size), desc="Creating embeddings"):
            for i in range(0, len(input_texts), batch_size):
                batch = input_texts[i : i + batch_size]
                response = self.client.embeddings.create(
                    model=self.model_name, input=batch
                )
                batch_embeddings = [e.embedding for e in response.data]
                embeddings.extend(batch_embeddings)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None


class LLMTool:
    def __init__(self, api_key, base_url, model_name):
        """
        初始化 OpenAI 客户端
        Initialize OpenAI client

        :param api_key: OpenAI API 密钥 (OpenAI API key)
        :param base_url: OpenAI API 基础 URL (OpenAI API base URL)
        :param model_name: 使用的模型名称 (Model name to be used)
        """
        self.client = openai.Client(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def chat(self, prompt, max_tokens=None):
        """
        使用 OpenAI 的聊天 API 生成回复
        Generate responses using OpenAI's chat API

        :param prompt: 用户输入的提示 (User input prompt)
        :param max_tokens: 生成的最大 token 数量 (Maximum number of tokens to generate)
        :return: 生成的回复 (Generated response)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat API: {e}")
            return None


if __name__ == "__main__":
    api_key = "not empty"
    base_url = "http://localhost:9997/v1"

    embedding_name = "bge-m3"
    embeddingtool = EmbeddingTool(
        api_key=api_key, base_url=base_url, model_name=embedding_name
    )

    input_texts = ["What is the capital of China?", "主变压器安装的工艺标准是什么", ""]

    embeddings = embeddingtool.create_embedding(input_texts=input_texts)
    if embeddings:
        logger.info("Embedding created successfully:")
        logger.info(embeddings)
    else:
        logger.error("Failed to create embedding.")

    # llm_name = "Qwen2.5-72B-Instruct-GPTQ-Int4"
    # llm_tool = LLMTool(api_key=api_key, base_url=base_url, model_name=llm_name)

    # prompt = "主变压器安装的工艺标准是什么?"
    # response = llm_tool.chat(prompt=prompt)
    # if response:
    #     logger.info(f"LLM response: {response}")
    # else:
    #     logger.error("Failed to get LLM response.")
