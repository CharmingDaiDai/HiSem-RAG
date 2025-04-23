import os
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import Levenshtein
from tqdm import tqdm
from loguru import logger
import yaml

# 导入大模型工具类
# Import large model tool class
from xinference_utils import LLMTool


class AccuracyEvaluator:
    def __init__(self, base_path):
        """
        初始化评估器
        Initialize the evaluator
        """
        self.base_path = base_path
        self.results_dir = os.path.join(self.base_path, "res")
        self.file_results = (
            {}
        )  # 存储每个文件的评估结果 (Store evaluation results for each file)
        self.overall_results = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )  # 按题型存储总体结果 (Store overall results by question type)
        self.lock = threading.Lock()  # 用于线程安全 (For thread safety)
        self.config = self._load_config()  # 加载配置 (Load config)

        # 配置日志
        # Configure logging
        logger.remove()
        logger.add(
            os.path.join(self.base_path, "evaluation_results.log"), rotation="10MB"
        )
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

        # 初始化大模型工具 (放在加载配置之后)
        # Initialize large model tool (after loading config)
        self.llm_tool = self._init_llm_tool()

    def _load_config(self):
        """
        加载 config.yaml 配置文件
        Load config.yaml configuration file
        """
        # 假设 config.yaml 在 base_path 的上一级目录
        # Assume config.yaml is in the parent directory of base_path
        config_path = os.path.join(os.path.dirname(self.base_path), "config.yaml")
        if not os.path.exists(config_path):
            # 如果上一级没有，尝试在当前脚本同级目录查找
            # If not in parent, try the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config.yaml")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logger.info(
                    f"成功加载配置文件: {config_path}"
                )  # Successfully loaded configuration file
                return config
        except FileNotFoundError:
            logger.error(
                f"配置文件未找到: {config_path}"
            )  # Configuration file not found
            return None
        except yaml.YAMLError as e:
            logger.error(
                f"解析配置文件时出错 {config_path}: {e}"
            )  # Error parsing configuration file
            return None
        except Exception as e:
            logger.error(
                f"加载配置文件时发生未知错误 {config_path}: {e}"
            )  # Unknown error loading configuration file
            return None

    def _init_llm_tool(self):
        """
        初始化大模型工具 (从配置加载)
        Initialize large model tool (load from config)
        """
        if not self.config or "models" not in self.config:
            logger.error(
                "配置未加载或缺少 'models' 部分，无法初始化大模型工具"
            )  # Config not loaded or 'llm' section missing, cannot initialize LLM tool
            return None

        active_llm = self.config["models"]["active_llm"]
        model_config = self.config["models"][active_llm]

        api_key = model_config.get("api_key")
        base_url = model_config.get("base_url")
        model_name = model_config.get("model_name")

        if not api_key or not base_url or not model_name:
            logger.error(
                "LLM 配置不完整 (缺少 api_key, base_url, 或 model_name)"
            )  # LLM configuration incomplete (missing api_key, base_url, or model_name)
            return None

        try:
            logger.info(
                f"使用模型 '{model_name}' 初始化 LLM 工具，URL: {base_url}"
            )  # Initializing LLM tool with model '{model_name}', URL: {base_url}
            return LLMTool(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
            )
        except Exception as e:
            logger.error(
                f"初始化大模型工具失败: {e}"
            )  # Failed to initialize large model tool
            return None

    def evaluate_fill_blank_with_llm(
        self, standard_answer, rag_answer, question_text=""
    ):
        """
        使用大模型评估填空题答案，失败时最多重试3次
        Evaluate fill-in-the-blank answers using large model, retry up to 3 times on failure
        """
        if not standard_answer or not rag_answer:
            return False

        # 如果大模型工具初始化失败，回退到传统方法
        # If large model tool initialization failed, fall back to traditional method
        if self.llm_tool is None:
            return self.evaluate_fill_blank_traditional(standard_answer, rag_answer)

        # 构建提示词（提前构建避免重复）
        # Build prompt (build in advance to avoid repetition)
        prompt = f"""作为一个客观公正的专业评判员，请评估以下填空题的回答是否正确。

    题目: {question_text}

    标准答案: {standard_answer}
    学生回答: {rag_answer}

    请考虑以下因素:
    1. 语义一致性: 答案表达的核心意思是否一致
    2. 关键词匹配: 关键术语是否存在
    3. 专业术语: 专业术语是否使用准确

    请只回答"正确"或"错误"，不要解释。"""

        # 添加重试机制
        # Add retry mechanism
        max_retries = 3
        for attempt in range(
            max_retries + 1
        ):  # +1 是因为第一次不算重试 (+1 because the first time is not counted as a retry)
            try:
                # 调用大模型评估
                # Call large model for evaluation
                result = self.llm_tool.chat(prompt=prompt)

                # 判断回答是否表示正确
                # Determine if the answer indicates correctness
                return any(
                    keyword in result.lower()
                    for keyword in ["正确", "对", "是的", "yes", "correct", "right"]
                )

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"{question_text} 大模型评估填空题时出错 (尝试 {attempt+1}/{max_retries}): {e}，正在重试..."  # Error evaluating fill-in-the-blank question (attempt {attempt+1}/{max_retries}): {e}, retrying...
                    )
                else:
                    logger.error(
                        f"大模型评估填空题时出错: 已达到最大重试次数({max_retries})，回退到传统方法"  # Error evaluating fill-in-the-blank question: maximum number of retries reached ({max_retries}), falling back to traditional method
                    )
                    return self.evaluate_fill_blank_traditional(
                        standard_answer, rag_answer
                    )

    def evaluate_fill_blank_traditional(self, standard_answer, rag_answer):
        """
        传统方法评估填空题答案，采用文本相似度方法
        Traditional method for evaluating fill-in-the-blank answers using text similarity approach
        """
        # 清理和标准化答案
        # Clean and standardize answers
        std_ans = re.sub(r"\s+", "", str(standard_answer).lower())
        rag_ans = re.sub(r"\s+", "", str(rag_answer).lower())

        # 如果答案完全匹配
        # If answers match exactly
        if std_ans == rag_ans:
            return True

        # 计算相似度
        # Calculate similarity
        similarity = self._calculate_similarity(std_ans, rag_ans)

        # 如果相似度超过阈值，认为是正确的
        # If similarity exceeds threshold, consider it correct
        return similarity > 0.8

    def _calculate_similarity(self, text1, text2):
        """
        计算两个文本的相似度
        Calculate similarity between two texts
        """
        # 使用Levenshtein距离计算相似度
        # Calculate similarity using Levenshtein distance
        distance = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0  # 两个都是空字符串 (Both are empty strings)
        similarity = 1 - (distance / max_len)
        return similarity

    def evaluate_choice_question(self, standard_answer, rag_answer):
        """
        评估选择题答案，要求完全匹配
        Evaluate multiple-choice question answers, requiring exact match
        """
        if not standard_answer or not rag_answer:
            return False

        # 清理和标准化答案
        # Clean and standardize answers
        std_ans = re.sub(r"\s+", "", str(standard_answer).upper())
        rag_ans = re.sub(r"\s+", "", str(rag_answer).upper())

        # 从文本中提取选项字母
        # Extract option letters from text
        std_options = re.findall(r"[A-D]", std_ans)
        rag_options = re.findall(r"[A-D]", rag_ans)

        # 排序以处理答案顺序不同的情况（例如"AB"和"BA"）
        # Sort to handle cases where answer order differs (e.g., "AB" and "BA")
        std_options.sort()
        rag_options.sort()

        return "".join(std_options) == "".join(rag_options)

    def evaluate_true_false(self, standard_answer, rag_answer):
        """
        评估判断题答案
        Evaluate true/false question answers
        """
        if not standard_answer or not rag_answer:
            return False

        # 标准化答案
        # Standardize answers
        std_ans = str(standard_answer).lower().strip()
        rag_ans = str(rag_answer).lower().strip()

        # 将各种可能的表达方式标准化
        # Standardize various possible expressions
        true_patterns = ["正确", "对", "t", "true", "√", "是"]
        false_patterns = ["错误", "错", "f", "false", "×", "否"]

        std_is_true = any(pattern in std_ans for pattern in true_patterns)
        std_is_false = any(pattern in std_ans for pattern in false_patterns)

        rag_is_true = any(pattern in rag_ans for pattern in true_patterns)
        rag_is_false = any(pattern in rag_ans for pattern in false_patterns)

        # 如果标准答案和RAG答案的判断一致
        # If the judgment of standard answer and RAG answer are consistent
        return (std_is_true and rag_is_true) or (std_is_false and rag_is_false)

    def evaluate_file(self, filename):
        """
        评估单个文件中所有题目的准确率
        Evaluate accuracy of all questions in a single file
        """
        file_path = os.path.join(self.results_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                questions = json.load(f)

            if not isinstance(questions, list):
                logger.warning(
                    f"文件 {filename} 不是有效的题目列表，跳过评估"
                )  # File {filename} is not a valid question list, skipping evaluation
                return

            # 初始化当前文件的结果统计
            # Initialize result statistics for current file
            file_result = defaultdict(lambda: {"correct": 0, "total": 0})

            # 处理每个问题
            # Process each question
            for question in questions:
                q_type = question.get("type", "未知类型")  # Unknown type
                std_answer = question.get("answer", "")
                rag_answer = question.get("rag_answer", "")
                question_text = question.get("question", "")

                # 如果没有RAG答案，跳过
                # Skip if there is no RAG answer
                if not rag_answer:
                    continue

                is_correct = False

                # 根据题型调用不同的评估方法
                # Call different evaluation methods based on question type
                if q_type == "填空题":  # Fill-in-the-blank question
                    is_correct = self.evaluate_fill_blank_with_llm(
                        std_answer, rag_answer, question_text
                    )
                elif q_type in [
                    "单选题",
                    "多选题",
                ]:  # Single-choice or multiple-choice question
                    is_correct = self.evaluate_choice_question(std_answer, rag_answer)
                elif q_type == "判断题":  # True/false question
                    is_correct = self.evaluate_true_false(std_answer, rag_answer)

                # 更新文件统计数据
                # Update file statistics
                file_result[q_type]["total"] += 1
                if is_correct:
                    file_result[q_type]["correct"] += 1

                # 在文件结果中也保存每个问题的评估结果，方便后续分析
                # Also save evaluation results for each question in file results for later analysis
                question["evaluation"] = {
                    "is_correct": is_correct,
                    "standard_answer": std_answer,
                    "rag_answer": rag_answer,
                }

            # 计算文件总准确率
            # Calculate overall file accuracy
            total_correct = sum(result["correct"] for result in file_result.values())
            total_questions = sum(result["total"] for result in file_result.values())

            if total_questions > 0:
                file_result["总体"] = {  # Overall
                    "correct": total_correct,
                    "total": total_questions,
                    "accuracy": total_correct / total_questions,
                }

            # 保存文件结果
            # Save file results
            with self.lock:  # 线程安全地更新共享数据 (Thread-safe update of shared data)
                self.file_results[filename] = file_result

                # 更新总体结果
                # Update overall results
                for q_type, stats in file_result.items():
                    if (
                        q_type != "总体"
                    ):  # 避免重复计算总体数据 (Avoid recalculating overall data)
                        self.overall_results[q_type]["correct"] += stats["correct"]
                        self.overall_results[q_type]["total"] += stats["total"]

            # 将评估结果保存到文件
            # Save evaluation results to file
            output_path = os.path.join(
                self.results_dir, f"{os.path.splitext(filename)[0]}_evaluated.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)

            return file_result

        except Exception as e:
            logger.error(
                f"评估文件 {filename} 时出错: {e}"
            )  # Error evaluating file {filename}: {e}
            return None

    def evaluate_all(self, max_workers=80):
        """
        评估所有结果文件
        Evaluate all result files
        """
        if not os.path.exists(self.results_dir):
            logger.error(
                f"结果目录不存在: {self.results_dir}"
            )  # Result directory does not exist: {self.results_dir}
            return

        # 获取所有JSON文件
        # Get all JSON files
        json_files = [
            f
            for f in os.listdir(self.results_dir)
            if f.endswith(".json") and not f.endswith("_evaluated.json")
        ]

        if not json_files:
            logger.warning(
                f"在 {self.results_dir} 中没有找到任何JSON文件"
            )  # No JSON files found in {self.results_dir}
            return

        logger.info(
            f"开始评估 {len(json_files)} 个结果文件"
        )  # Start evaluating {len(json_files)} result files

        # 使用线程池并行处理文件
        # Use thread pool to process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                tqdm(
                    executor.map(self.evaluate_file, json_files),
                    total=len(json_files),
                    desc="评估进度",  # Evaluation progress
                )
            )

        # 计算总体准确率
        # Calculate overall accuracy
        total_correct = sum(
            result["correct"] for result in self.overall_results.values()
        )
        total_questions = sum(
            result["total"] for result in self.overall_results.values()
        )

        if total_questions > 0:
            self.overall_results["总体"] = {  # Overall
                "correct": total_correct,
                "total": total_questions,
                "accuracy": total_correct / total_questions,
            }

        # 输出结果
        # Output results
        self.print_results()

        # 保存总体结果
        # Save overall results
        self.save_results()

    def print_results(self):
        """
        打印评估结果（按正确率从高到低排序）
        Print evaluation results (sorted by accuracy from high to low)
        """
        logger.info("\n=== 实验结果评估 ===")  # Experiment Result Evaluation

        # 打印每个文件的结果（按正确率排序）
        # Print results for each file (sorted by accuracy)
        logger.info(
            "\n--- 各文档准确率（从高到低）---"
        )  # Document Accuracy (from high to low)

        # 按准确率排序文件
        # Sort files by accuracy
        sorted_files = sorted(
            [
                (filename, results)
                for filename, results in self.file_results.items()
                if "总体" in results
            ],
            key=lambda x: x[1]["总体"].get("accuracy", 0),
            reverse=True,
        )

        for filename, results in sorted_files:
            accuracy = results["总体"].get("accuracy", 0) * 100
            logger.info(f"{filename}: {accuracy:.2f}%")

            # 按准确率排序各题型
            # Sort question types by accuracy
            sorted_types = sorted(
                [
                    (q_type, stats)
                    for q_type, stats in results.items()
                    if q_type != "总体" and stats["total"] > 0
                ],
                key=lambda x: (
                    x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0
                ),
                reverse=True,
            )

            for q_type, stats in sorted_types:
                type_acc = stats["correct"] / stats["total"] * 100
                logger.info(
                    f"  - {q_type}: {type_acc:.2f}% ({stats['correct']}/{stats['total']})"
                )

        # 打印总体结果（按正确率排序）
        # Print overall results (sorted by accuracy)
        logger.info(
            "\n--- 总体准确率（从高到低）---"
        )  # Overall Accuracy (from high to low)
        if "总体" in self.overall_results:
            total_acc = self.overall_results["总体"]["accuracy"] * 100
            logger.info(f"总体准确率: {total_acc:.2f}%")  # Overall accuracy

            # 按准确率排序各题型总体结果
            # Sort overall results by question type accuracy
            sorted_overall_types = sorted(
                [
                    (q_type, stats)
                    for q_type, stats in self.overall_results.items()
                    if q_type != "总体" and stats["total"] > 0
                ],
                key=lambda x: (
                    x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0
                ),
                reverse=True,
            )

            for q_type, stats in sorted_overall_types:
                type_acc = stats["correct"] / stats["total"] * 100
                logger.info(
                    f"{q_type}: {type_acc:.2f}% ({stats['correct']}/{stats['total']})"
                )

    def save_results(self):
        """
        保存评估结果到JSON文件
        Save evaluation results to JSON file
        """
        output = {
            "overall": {k: dict(v) for k, v in self.overall_results.items()},
            "by_file": {
                k: {q_type: dict(stats) for q_type, stats in v.items()}
                for k, v in self.file_results.items()
            },
        }

        # 计算每个题型的准确率
        # Calculate accuracy for each question type
        for results in [output["overall"]] + list(output["by_file"].values()):
            for q_type, stats in results.items():
                if stats.get("total", 0) > 0:
                    stats["accuracy"] = stats["correct"] / stats["total"]

        with open(
            os.path.join(self.base_path, "evaluation_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(
            f"评估结果已保存到 {os.path.join(self.base_path, 'evaluation_summary.json')}"  # Evaluation results have been saved to
        )


if __name__ == "__main__":
    # 创建评估器并运行评估
    # Create evaluator and run evaluation
    # 确保 base_path 设置正确，以便能找到 config.yaml
    # Ensure base_path is set correctly so config.yaml can be found
    evaluator = AccuracyEvaluator(base_path="res/1")
    if (
        evaluator.llm_tool
    ):  # 检查 LLM 工具是否成功初始化 (Check if LLM tool initialized successfully)
        evaluator.evaluate_all(max_workers=1)
    else:
        logger.error(
            "LLM 工具初始化失败，无法进行评估。请检查配置和日志。"
        )  # LLM tool initialization failed, cannot perform evaluation. Please check config and logs.
