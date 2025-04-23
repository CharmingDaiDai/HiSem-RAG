import os
import json
import numpy as np
import time
import yaml
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
import re
from datetime import datetime

from MyNode import MyNode
from xinference_utils import EmbeddingTool, LLMTool
from evaluate import AccuracyEvaluator


# 配置集中管理类
# Configuration Management Class
class ExperimentConfig:
    def __init__(self, config_path=None):
        """
        初始化配置，从文件加载
        Initialize configuration, load from file
        """
        self.config = {}

        # 如果提供了配置文件，则加载它
        # If configuration file is provided, load it
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        else:
            logger.warning(
                f"配置文件 {config_path} 不存在，请检查路径是否正确"
            )  # Configuration file {config_path} does not exist, please check if the path is correct
            # 如果无法加载配置文件，尝试使用默认配置路径
            # If the configuration file cannot be loaded, try using the default configuration path
            default_config_path = "config.yaml"
            if os.path.exists(default_config_path):
                logger.info(
                    f"使用默认配置文件路径: {default_config_path}"
                )  # Using default configuration file path: {default_config_path}
                self.load_from_file(default_config_path)
            else:
                logger.error(
                    "无法找到任何配置文件，请确保 config.yaml 文件存在"
                )  # Cannot find any configuration file, please ensure config.yaml exists
                raise FileNotFoundError(
                    "无法找到配置文件 config.yaml"
                )  # Cannot find configuration file config.yaml

        # 确保目录存在
        # Ensure directories exist
        self.ensure_directories()

    def load_from_file(self, config_path):
        """
        从YAML文件加载配置
        Load configuration from YAML file
        """
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file)

            print(
                f"从 {config_path} 加载配置成功"
            )  # Successfully loaded configuration from {config_path}
        except Exception as e:
            print(
                f"从 {config_path} 加载配置失败: {e}"
            )  # Failed to load configuration from {config_path}: {e}
            raise

    def ensure_directories(self):
        """
        确保所有需要的目录存在
        Ensure all required directories exist
        """
        base_path = self.config["paths"]["base_path"]
        os.makedirs(base_path, exist_ok=True)

        # 创建相对于base_path的子目录
        # Create subdirectories relative to base_path
        for dir_key in ["logs_dir", "results_dir"]:
            dir_path = os.path.join(base_path, self.config["paths"][dir_key])
            os.makedirs(dir_path, exist_ok=True)

    def get_path(self, key, create=False):
        """
        获取特定路径，可选创建目录
        Get specific path, optionally create directory

        :param key: 路径键名 (Path key)
        :param create: 是否创建目录 (Whether to create directory)
        :return: 路径字符串 (Path string)
        """
        if key == "base_path":
            path = self.config["paths"]["base_path"]
        elif key in self.config["paths"]:
            path = os.path.join(
                self.config["paths"]["base_path"], self.config["paths"][key]
            )
        else:
            raise ValueError(f"未知的路径键: {key}")  # Unknown path key: {key}

        if create and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return path

    def get_active_llm_config(self):
        """
        获取当前激活的LLM配置
        Get configuration of the currently active LLM
        """
        active_model = self.config["models"]["active_llm"]
        try:
            return self.config["models"][active_model]
        except Exception as e:
            raise ValueError(
                f"未知的激活模型: {active_model}, {e}"
            )  # Unknown active model: {active_model}

    def get(self, section, key=None, default=None):
        """
        获取配置值，支持默认值
        Get configuration value, supports default values

        :param section: 配置节点 (Configuration section)
        :param key: 配置键，为None时返回整个节点 (Configuration key, returns the whole section when None)
        :param default: 默认值 (Default value)
        :return: 配置值或默认值 (Configuration value or default value)
        """
        if key is None:
            return self.config[section]

        if section in self.config:
            if key in self.config[section]:
                return self.config[section][key]

        return default

    def log_config(self):
        """
        将配置记录到日志中
        Log configuration
        """
        logger.info("========== 实验配置 ==========")  # Experiment Configuration
        logger.info(
            f"实验名称: {self.config['system']['experiment_name']}"
        )  # Experiment Name
        logger.info(
            f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )  # Experiment Time
        logger.info(f"实验路径: {self.config['paths']['base_path']}")  # Experiment Path

        # 记录模型信息
        # Log model information
        active_model = self.config["models"]["active_llm"]
        model_config = self.get_active_llm_config()
        logger.info(
            f"使用模型: {active_model} ({model_config['model_name']})"
        )  # Using Model
        logger.info(
            f"嵌入模型: {self.config['models']['local']['embedding_model']}"
        )  # Embedding Model

        # 记录检索参数
        # Log retrieval parameters
        logger.info("\n检索算法参数:")  # Retrieval Algorithm Parameters
        for key, value in self.config["retrieval"].items():
            logger.info(f"  {key}: {value}")

        # 记录系统参数
        # Log system parameters
        logger.info("\n系统参数:")  # System Parameters
        for key, value in self.config["system"].items():
            logger.info(f"  {key}: {value}")

        logger.info("================================")

        # 完整配置导出到文件
        # Export complete configuration to file
        config_path = os.path.join(self.get_path("base_path"), "experiment_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)


# 初始化配置
# Initialize configuration
CONFIG = ExperimentConfig("config.yaml")

# 设置日志
# Set up logging
logger.remove()
logger.add(
    os.path.join(CONFIG.get_path("base_path"), "experiment.log"),
    rotation="100MB",
    level=CONFIG.get("system", "log_level"),
)
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    colorize=True,
    level=CONFIG.get("system", "log_level"),
)

# 记录配置到日志
# Log configuration
CONFIG.log_config()


def load_nodes_from_file(
    json_file_path: str,
) -> Tuple[Optional[MyNode], Dict[str, MyNode]]:
    """从JSON文件恢复节点结构"""
    # Restore node structure from JSON file
    try:
        # 检查文件是否存在
        # Check if the file exists
        if not os.path.exists(json_file_path):
            logger.error(
                f"文件不存在: {json_file_path}"
            )  # File does not exist: {json_file_path}
            return None, {}

        # 读取JSON文件
        # Read JSON file
        with open(json_file_path, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)

        # 将字典数据转换为MyNode对象
        # Convert dictionary data to MyNode objects
        nodes_dict = {}
        for node_id, node_data in nodes_data.items():
            nodes_dict[node_id] = MyNode.from_dict(node_data)

        # 找到根节点
        # Find the root node
        root_node = None
        for node in nodes_dict.values():
            if node.parent_id is None:
                root_node = node
                break

        if root_node is None:
            logger.warning(
                f"文件 {json_file_path} 中未找到根节点"
            )  # Root node not found in file {json_file_path}
            return None, nodes_dict

        return root_node, nodes_dict

    except Exception as e:
        logger.error(
            f"从文件加载节点时出错 {json_file_path}: {e}"
        )  # Error loading nodes from file {json_file_path}: {e}
        import traceback

        logger.error(traceback.format_exc())
        return None, {}


def calculate_adaptive_threshold(similarities, level, log_file=None):
    """计算分布感知的自适应阈值，并应用安全机制"""
    # Calculate distribution-aware adaptive threshold and apply safety mechanisms
    # 从配置获取参数
    # Get parameters from configuration
    beta = CONFIG.get("retrieval", "beta")
    gamma = CONFIG.get("retrieval", "gamma")
    k_min = CONFIG.get("retrieval", "k_min")
    k_max = CONFIG.get("retrieval", "k_max")
    theta_min = CONFIG.get("retrieval", "theta_min")

    if len(similarities) == 0:
        return 0.0

    # 计算统计指标
    # Calculate statistical indicators
    s_max = np.max(similarities)
    mu = np.mean(similarities)
    sigma = np.std(similarities)
    cv = sigma / mu if mu != 0 else 0  # 变异系数 (Coefficient of variation)

    # 计算初始自适应阈值
    # Calculate initial adaptive threshold
    raw_threshold = beta * s_max - (1 - gamma * cv) * (s_max - mu)

    # 应用绝对阈值下限安全机制
    # Apply absolute threshold lower limit safety mechanism
    threshold = max(raw_threshold, theta_min)

    # 应用节点数量安全机制
    # Apply node count safety mechanism
    # 对相似度进行降序排列
    # Sort similarities in descending order
    sorted_similarities = np.sort(similarities)[::-1]

    # 确保至少保留k_min个节点
    # Ensure at least k_min nodes are kept
    if np.sum(similarities >= threshold) < k_min:
        if len(similarities) >= k_min:
            threshold = sorted_similarities[k_min - 1]
        else:
            threshold = sorted_similarities[0]

    # 确保最多保留k_max个节点
    # Ensure at most k_max nodes are kept
    if np.sum(similarities >= threshold) > k_max:
        threshold = sorted_similarities[k_max - 1]

    return threshold


def is_leaf_node(node: MyNode, nodes_dict: Dict[str, MyNode]) -> bool:
    """
    判断节点是否为叶子节点
    Determine if a node is a leaf node

    参数:
        node: 待判断的节点 (Node to check)
        nodes_dict: 节点字典 (Node dictionary)

    返回:
        是否为叶子节点 (Whether it is a leaf node)
    """
    # 如果节点没有子节点或者所有子节点ID都不存在于nodes_dict中，则为叶子节点
    # If the node has no children or all child IDs do not exist in nodes_dict, it is a leaf node
    return not node.children or all(
        child_id not in nodes_dict for child_id in node.children
    )


def compute_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    # Calculate the cosine similarity between two vectors
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    # 避免除以零
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def hierarchical_search(
    query: str,
    root_node: MyNode,
    nodes_dict: Dict[str, MyNode],
    embedding_tool: EmbeddingTool,
    log_file=None,
) -> List[Tuple[MyNode, float]]:
    """使用自适应阈值的层级检索算法"""
    # Hierarchical search algorithm using adaptive threshold

    # 获取配置参数
    # Get configuration parameters
    max_depth = CONFIG.get("retrieval", "max_depth")
    max_results = CONFIG.get("retrieval", "max_results")

    # 第一步：将查询转为向量
    # Step 1: Convert query to vector
    query_embedding = embedding_tool.create_embedding([query])[0]

    # 用于存储检索结果
    # Used to store retrieval results
    results = []  # 只存储叶子节点 (Store only leaf nodes)
    all_matched_nodes = (
        []
    )  # 存储所有匹配的节点（包括中间节点），用于调试 (Store all matched nodes (including intermediate nodes) for debugging)
    visited_ids = set()  # 避免重复检索 (Avoid duplicate retrieval)

    # 搜索路径追踪，记录节点的父子关系和搜索路径
    # Search path tracking, record node parent-child relationships and search paths
    search_paths = []

    def search_level(current_nodes: List[MyNode], level: int, path_prefix: str = ""):
        """在当前层级搜索相关节点"""
        # Search for relevant nodes at the current level
        if level >= max_depth or not current_nodes:
            return

        # 记录父节点ID，用于输出更清晰的搜索路径
        # Record parent node ID for clearer search path output
        parent_info = {}
        for node in current_nodes:
            for child_id in node.children:
                if child_id in nodes_dict:
                    parent_info[child_id] = node.id

        # 计算当前层级所有节点的相似度
        # Calculate similarity for all nodes at the current level
        nodes_with_similarity = []
        for node in current_nodes:
            if node.id in visited_ids:
                continue

            # 标记当前检索路径
            # Mark the current retrieval path
            current_path = (
                f"{path_prefix} -> {node.title}" if path_prefix else node.title
            )

            if not node.embedding or len(node.embedding) == 0:
                node.embedding = embedding_tool.create_embedding(
                    [node.title + node.knowledge_summary]
                )[0]

            # 计算相似度
            # Calculate similarity
            similarity = compute_cosine_similarity(query_embedding, node.embedding)
            nodes_with_similarity.append((node, similarity, current_path))

        # 提取相似度数组用于计算阈值
        # Extract similarity array to calculate threshold
        if nodes_with_similarity:
            similarities = np.array([sim for _, sim, _ in nodes_with_similarity])

            # 计算自适应阈值
            # Calculate adaptive threshold
            threshold = calculate_adaptive_threshold(similarities, level)

            # 筛选通过阈值的节点
            # Filter nodes that pass the threshold
            passed_nodes = []

            for i, (node, sim, path) in enumerate(nodes_with_similarity):
                if sim >= threshold:
                    passed_nodes.append((node, path))
                    # 判断是否为叶子节点
                    # Determine if it is a leaf node
                    if is_leaf_node(node, nodes_dict):
                        results.append((node, sim))
                    else:
                        all_matched_nodes.append((node, sim))

                    visited_ids.add(node.id)
                    search_paths.append(
                        f"路径 {len(search_paths)+1}: {path} [相似度: {sim:.4f}]"  # Path {len(search_paths)+1}: {path} [Similarity: {sim:.4f}]
                    )

            # 收集通过筛选节点的所有子节点，并分别递归处理每个节点的子节点树
            # Collect all child nodes of the filtered nodes and recursively process each node's subtree separately
            for passed_node, path in passed_nodes:
                next_level_nodes = []
                for child_id in passed_node.children:
                    if child_id in nodes_dict and child_id not in visited_ids:
                        next_level_nodes.append(nodes_dict[child_id])

                # 对每个通过节点的子节点树分别进行递归搜索，保持路径独立
                # Recursively search the subtree of each passed node separately, keeping paths independent
                if next_level_nodes:
                    search_level(next_level_nodes, level + 1, path)

    # 从根节点开始搜索
    # Start searching from the root node
    search_level([root_node], 0)

    # 按相似度排序结果
    # Sort results by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    # 如果没有找到叶子节点，但找到了其他节点，使用最匹配的非叶子节点
    # If no leaf nodes are found, but other nodes are found, use the best matching non-leaf nodes
    if not results and all_matched_nodes:
        # 按相似度排序
        # Sort by similarity
        all_matched_nodes.sort(key=lambda x: x[1], reverse=True)
        # 取最相关的几个节点
        # Take the most relevant nodes
        for node, score in all_matched_nodes[:max_results]:
            results.append((node, score))

    return results


def parse_llm_response(answer):
    """解析大模型返回的JSON响应，增强健壮性"""
    # Parse the JSON response returned by the large model, enhance robustness
    if not answer:
        return {
            "rag_answer": "系统错误",
            "rag_explain": "大模型未返回回答",
        }  # System error, Large model did not return an answer

    # 1. 尝试直接解析完整JSON
    # 1. Try to parse the complete JSON directly
    try:
        # 如果回答完全是合法JSON
        # If the answer is completely valid JSON
        if answer.strip().startswith("{") and answer.strip().endswith("}"):
            result = json.loads(answer)
            if "rag_answer" in result and "rag_explain" in result:
                return result
    except json.JSONDecodeError:
        pass

    # 2. 在回答中寻找JSON，处理可能的多个JSON对象
    # 2. Look for JSON in the answer, handle possible multiple JSON objects
    try:
        # 使用正则表达式找出所有的JSON对象
        # Use regular expressions to find all JSON objects
        json_matches = re.findall(r"({[^{}]*(?:{[^{}]*}[^{}]*)*})", answer)

        if json_matches:
            # 尝试解析每一个匹配到的JSON对象
            # Try to parse each matched JSON object
            for json_str in json_matches:
                try:
                    result = json.loads(json_str)
                    # 检查是否包含我们需要的键
                    # Check if it contains the keys we need
                    if "rag_answer" in result and "rag_explain" in result:
                        return result
                except:
                    continue
    except Exception:
        pass

    # 3. 处理可能包含在代码块中的JSON
    # 3. Handle JSON possibly contained in code blocks
    try:
        # 提取代码块中的内容
        # Extract content from code blocks
        code_blocks = re.findall(r"```(?:json)?\s*\n([\s\S]*?)\n```", answer)
        if code_blocks:
            for block in code_blocks:
                try:
                    result = json.loads(block)
                    if "rag_answer" in result and "rag_explain" in result:
                        return result
                except:
                    continue
    except Exception:
        pass

    # 4. 如果是多个答案，只取第一个
    # 4. If there are multiple answers, take only the first one
    if "rag_answer" in answer and answer.count('"rag_answer"') > 1:
        try:
            # 找出第一个完整的JSON对象结束位置
            # Find the end position of the first complete JSON object
            first_json_end = answer.find("}\n\n{")
            if first_json_end > 0:
                first_json = answer[: first_json_end + 1]
                result = json.loads(first_json)
                if "rag_answer" in result and "rag_explain" in result:
                    return result
        except Exception:
            pass

    # 5. 尝试直接提取字段
    # 5. Try to extract fields directly
    try:
        answer_match = re.search(r'"rag_answer"\s*:\s*"([^"]*)"', answer)
        explain_match = re.search(r'"rag_explain"\s*:\s*"([^"]*)"', answer)

        if answer_match and explain_match:
            return {
                "rag_answer": answer_match.group(1),
                "rag_explain": explain_match.group(1),
            }
    except Exception:
        pass

    # 所有方法都失败，返回原始答案
    # All methods failed, return the original answer
    return {"rag_answer": "解析错误", "rag_explain": answer}  # Parsing error


def generate_answer_from_context(
    query: str,
    search_results: List[Tuple[MyNode, float]],
    llm_tool: LLMTool,
    log_file=None,
):
    """基于检索的上下文，生成问题的回答，添加重试机制"""
    # Generate answer based on retrieved context, add retry mechanism

    def log_msg(msg):
        if log_file:
            log_file.write(f"{msg}\n")

    # 从配置获取参数
    # Get parameters from configuration
    max_results = CONFIG.get("retrieval", "max_results")

    # 限制使用的检索结果数量
    # Limit the number of retrieval results used
    search_results = search_results[:max_results]

    if not search_results:
        return {
            "rag_answer": "无法回答",
            "rag_explain": "未找到相关信息",
        }  # Cannot answer, No relevant information found

    # 构建上下文信息
    # Build context information
    context_blocks = []
    for i, (node, score) in enumerate(search_results, 1):
        # 使用标题和原始内容，不使用摘要
        # Use title and original content, not summary
        block = f"【文档{i}】\n标题: {node.title}\n内容:\n{node.page_content}\n"  # [Document {i}]\nTitle: {node.title}\nContent:\n{node.page_content}\n
        context_blocks.append(block)

    # 合并上下文信息
    # Merge context information
    context = "\n".join(context_blocks)

    # 构建提示词
    # Build prompt
    prompt = f"""你是一个专业的电力工程考试辅助系统，请根据我提供的考试题目和参考资料生成答案。

题目: {query}

参考资料:
{context}

请严格遵循以下规则:
1. 只基于提供的参考资料回答问题，不使用其他背景知识
2. 如果参考资料不足以回答题目，请回答"无法回答"
3. 根据题目类型提供适当的答案格式:
   - 单选题：直接给出选项字母(A/B/C/D)
   - 多选题：给出所有正确选项字母(如ABC)
   - 判断题：回答"正确"或"错误"
   - 填空题：直接给出填空答案
4. 解释答案的理由，引用参考资料中的关键信息

请用以下JSON格式回答，确保只输出一个JSON对象:
{{
  "rag_answer": "你的答案",
  "rag_explain": "你的解释"
}}"""
    # You are a professional power engineering exam assistance system. Please generate answers based on the exam questions and reference materials I provide.
    #
    # Question: {query}
    #
    # Reference Materials:
    # {context}
    #
    # Please strictly follow these rules:
    # 1. Answer questions based solely on the provided reference materials, do not use other background knowledge.
    # 2. If the reference materials are insufficient to answer the question, please answer "无法回答" (Cannot answer).
    # 3. Provide the appropriate answer format based on the question type:
    #    - Single choice: Directly provide the option letter (A/B/C/D).
    #    - Multiple choice: Provide all correct option letters (e.g., ABC).
    #    - True/False: Answer "正确" (Correct) or "错误" (Incorrect).
    #    - Fill-in-the-blank: Directly provide the fill-in answer.
    # 4. Explain the reason for the answer, citing key information from the reference materials.
    #
    # Please respond in the following JSON format, ensuring only one JSON object is output:
    # {{
    #   "rag_answer": "Your answer",
    #   "rag_explain": "Your explanation"
    # }}

    # 实现重试机制
    # Implement retry mechanism
    max_retries = 3
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            # 调用大模型生成回答（使用原始LLMTool类）
            # Call the large model to generate an answer (using the original LLMTool class)
            answer = llm_tool.chat(prompt=prompt)

            # 记录原始回答
            # Log the original answer
            log_msg("\n原始大模型回答:")  # Original large model answer:
            log_msg(answer)

            if not answer:
                raise Exception(
                    "大模型返回空回答"
                )  # Large model returned an empty answer

            # 使用增强的解析函数处理返回结果
            # Use the enhanced parsing function to process the returned result
            result = parse_llm_response(answer)
            return result

        except Exception as e:
            retry_count += 1
            last_error = e

            if retry_count < max_retries:
                # 警告日志
                # Warning log
                warning_msg = f"调用大模型失败 (第{retry_count}次重试): {str(e)}"  # Failed to call large model (Retry {retry_count}): {str(e)}
                log_msg(f"\n警告: {warning_msg}")  # Warning: {warning_msg}
                logger.warning(warning_msg)
                # 短暂等待后重试
                # Wait briefly before retrying
                time.sleep(2 * retry_count)  # 递增等待时间 (Increasing wait time)
            else:
                # 错误日志
                # Error log
                error_msg = f"调用大模型失败，已达到最大重试次数 ({max_retries}): {str(last_error)}"  # Failed to call large model, maximum retry count reached ({max_retries}): {str(last_error)}
                log_msg(f"\n错误: {error_msg}")  # Error: {error_msg}
                logger.error(error_msg)
                return {
                    "rag_answer": "系统错误",  # System error
                    "rag_explain": f"调用大模型达到最大重试次数: {str(last_error)}",  # Calling large model reached maximum retry count: {str(last_error)}
                }


def load_all_questions():
    """加载所有问答数据，跳过标记有error的题目以及被排除的ID"""
    # Load all question-answer data, skip questions marked with 'error' and excluded IDs
    questions_dir = CONFIG.get("paths", "questions_dir")
    all_questions = []
    skipped_count = 0  # 记录跳过的题目数量 (Record the number of skipped questions)
    excluded_count = (
        0  # 记录被排除ID的题目数量 (Record the number of questions with excluded IDs)
    )

    # 获取要排除的问题ID列表
    # Get the list of question IDs to exclude
    excluded_ids = CONFIG.get("system", "excluded_question_ids", [])

    # 检查目录是否存在
    # Check if the directory exists
    if not os.path.exists(questions_dir):
        logger.error(
            f"问题数据目录不存在: {questions_dir}"
        )  # Question data directory does not exist: {questions_dir}
        return []

    # 遍历所有JSON文件
    # Iterate through all JSON files
    for filename in os.listdir(questions_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(questions_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    questions = json.load(f)

                    # 过滤掉有error字段的题目和被排除ID的题目
                    # Filter out questions with the 'error' field and excluded IDs
                    valid_questions = []
                    for question in questions:
                        # 检查docId是否在排除列表中
                        # Check if docId is in the exclusion list
                        question_id = str(question.get("docId", ""))

                        if question_id in excluded_ids:
                            excluded_count += 1
                            continue

                        if "error" not in question:
                            valid_questions.append(question)
                        else:
                            skipped_count += 1

                    all_questions.extend(valid_questions)
            except Exception as e:
                logger.error(
                    f"加载问题文件失败 {file_path}: {e}"
                )  # Failed to load question file {file_path}: {e}

    return all_questions


def group_questions_by_docid(questions):
    """按文档ID分组问题"""
    # Group questions by document ID
    grouped = {}
    for q in questions:
        doc_id = q.get("docId", "")
        if doc_id:
            if doc_id not in grouped:
                grouped[doc_id] = []
            grouped[doc_id].append(q)

    logger.info(
        f"问题分组完成，共有 {len(grouped)} 个文档"
    )  # Question grouping completed, total {len(grouped)} documents
    return grouped


def process_single_document(
    doc_id: str, questions: List[Dict], embedding_tool: EmbeddingTool, llm_tool: LLMTool
):
    """处理单个文档的所有问题"""
    # Process all questions for a single document
    logger.info(f"开始处理文档 {doc_id}")  # Start processing document {doc_id}

    # 创建日志文件
    # Create log file
    logs_dir = CONFIG.get_path("logs_dir", create=True)
    log_path = os.path.join(logs_dir, f"{doc_id}_log.txt")

    # 加载文档
    # Load document
    processed_docs_dir = CONFIG.get("paths", "processed_docs_dir")
    doc_path = os.path.join(processed_docs_dir, f"{doc_id}_processed.json")

    if not os.path.exists(doc_path):
        logger.error(f"文档不存在: {doc_path}")  # Document does not exist: {doc_path}
        return False, questions

    with open(log_path, "w", encoding="utf-8") as log_file:
        # 记录文档信息
        # Log document information
        log_file.write(
            f"=== 处理文档 {doc_id} ===\n"
        )  # === Processing document {doc_id} ===
        log_file.write(
            f"题目数量: {len(questions)}\n"
        )  # Number of questions: {len(questions)}

        # 加载文档结构
        # Load document structure
        root_node, nodes_dict = load_nodes_from_file(doc_path)
        if not root_node:
            log_file.write(
                f"加载文档失败: {doc_path}\n"
            )  # Failed to load document: {doc_path}
            logger.error(
                f"加载文档失败: {doc_path}"
            )  # Failed to load document: {doc_path}
            return False, questions

        log_file.write(
            f"成功加载文档，共 {len(nodes_dict)} 个节点\n"
        )  # Successfully loaded document, total {len(nodes_dict)} nodes

        # 逐个处理问题
        # Process questions one by one
        for i, question in enumerate(questions):
            question_id = question.get("id", str(i))
            question_text = question.get("question", "")
            question_type = question.get("type", "未知类型")  # Unknown type

            log_file.write(
                f"\n\n==== 问题 {i+1}/{len(questions)} ====\n"
            )  # ==== Question {i+1}/{len(questions)} ====
            log_file.write(f"ID: {question_id}\n")
            log_file.write(f"类型: {question_type}\n")  # Type: {question_type}
            log_file.write(f"问题: {question_text}\n")  # Question: {question_text}

            # 构建查询
            # Build query
            query = f"{question_type}:{question_text}"
            log_file.write(f"\n查询: {query}\n")  # Query: {query}

            # 执行检索
            # Execute retrieval
            try:
                results = hierarchical_search(
                    query, root_node, nodes_dict, embedding_tool, log_file=log_file
                )

                if not results:
                    log_file.write("未找到相关内容\n")  # No relevant content found
                    question["rag_answer"] = "无法回答"  # Cannot answer
                    question["rag_explain"] = (
                        "未找到相关内容"  # No relevant content found
                    )
                    continue

                # 生成回答 - 内部已包含重试机制
                # Generate answer - retry mechanism included internally
                answer_result = generate_answer_from_context(
                    query, results, llm_tool, log_file=log_file
                )
                log_file.write(
                    f"\n生成的回答: {json.dumps(answer_result, ensure_ascii=False, indent=2)}\n"  # Generated answer: ...
                )

                # 存储检索结果
                # Store retrieval results
                context_blocks = []
                for node, score in results:
                    context_blocks.append(node.page_content)

                # 更新问题
                # Update question
                question["rag_answer"] = answer_result.get("rag_answer", "")
                question["rag_explain"] = answer_result.get("rag_explain", "")
                question["context"] = context_blocks

            except Exception as e:
                log_file.write(
                    f"处理问题时出错: {e}\n"
                )  # Error processing question: {e}
                logger.error(
                    f"处理文档 {doc_id} 问题 {question_id} 时出错: {e}"
                )  # Error processing question {question_id} in document {doc_id}: {e}
                import traceback

                log_file.write(traceback.format_exc() + "\n")

                question["rag_answer"] = "错误"  # Error
                question["rag_explain"] = (
                    f"处理问题时出错: {str(e)}"  # Error processing question: {str(e)}
                )

    # 保存结果
    # Save results
    results_dir = CONFIG.get_path("results_dir", create=True)
    result_path = os.path.join(results_dir, f"{doc_id}_rag.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    logger.info(
        f"文档 {doc_id} 处理完成，结果已保存到 {result_path}"
    )  # Document {doc_id} processing completed, results saved to {result_path}
    return True, questions


def main():
    # 初始化API工具
    # Initialize API tools
    # 获取模型配置
    # Get model configuration
    local_config = CONFIG.get("models", "local")
    active_llm_config = CONFIG.get_active_llm_config()

    # 初始化工具
    # Initialize tools
    embedding_tool = EmbeddingTool(
        api_key=local_config["api_key"],
        base_url=local_config["base_url"],
        model_name=local_config["embedding_model"],
    )

    llm_tool = LLMTool(
        api_key=active_llm_config["api_key"],
        base_url=active_llm_config["base_url"],
        model_name=active_llm_config["model_name"],
    )

    # 加载所有问题
    # Load all questions
    all_questions = load_all_questions()
    if not all_questions:
        logger.error("没有找到题目数据，退出")  # No question data found, exiting
        return

    # 按文档ID分组
    # Group by document ID
    grouped_questions = group_questions_by_docid(all_questions)

    # 创建进度条
    # Create progress bar
    total_docs = len(grouped_questions)
    progress_bar = tqdm(total=total_docs, desc="处理文档")  # Processing documents

    # 记录处理结果
    # Record processing results
    results = {}

    # 使用多线程处理文档
    # Use multithreading to process documents
    threads = CONFIG.get("system", "threads")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # 提交所有任务
        # Submit all tasks
        future_to_doc_id = {
            executor.submit(
                process_single_document, doc_id, questions, embedding_tool, llm_tool
            ): doc_id
            for doc_id, questions in grouped_questions.items()
        }

        # 处理完成的任务 - 使用as_completed等待任务完成
        # Process completed tasks - use as_completed to wait for task completion
        for future in as_completed(future_to_doc_id):
            doc_id = future_to_doc_id[future]
            try:
                success, processed_questions = future.result()
                results[doc_id] = {
                    "success": success,
                    "questions_count": len(processed_questions),
                }
            except Exception as e:
                logger.error(
                    f"处理文档 {doc_id} 失败: {e}"
                )  # Failed to process document {doc_id}: {e}
                results[doc_id] = {"success": False, "error": str(e)}
            finally:
                progress_bar.update(1)

    # 打印总结
    # Print summary
    success_count = sum(1 for r in results.values() if r.get("success", False))
    logger.info(
        f"处理完成! 成功: {success_count}/{total_docs}"
    )  # Processing completed! Success: {success_count}/{total_docs}

    # 保存处理结果摘要
    # Save processing result summary
    with open(
        os.path.join(CONFIG.get_path("base_path"), "experiment_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "total_documents": total_docs,
                "success_count": success_count,
                "details": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 创建评估器并运行评估
    # Create evaluator and run evaluation
    eval_threads = CONFIG.get("system", "eval_threads")
    evaluator = AccuracyEvaluator(base_path=CONFIG.get_path("base_path"))
    evaluator.evaluate_all(max_workers=eval_threads)


if __name__ == "__main__":
    main()
