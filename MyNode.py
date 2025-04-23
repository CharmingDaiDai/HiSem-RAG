from typing import Any, Dict, List, Optional
import uuid


class MyNode:
    """
    层级文档节点类，用于表示具有层级结构的文档节点，支持标题层级、父子关系等特性。
    Hierarchical document node class for representing document nodes with hierarchical structure,
    supporting features such as title hierarchy, parent-child relationships, etc.

    属性 (Attributes):
        id: 节点唯一标识符 (Unique identifier for the node)
        parent_id: 父节点ID，顶层节点为None (Parent node ID, None for top-level nodes)
        level: 标题等级，0表示非标题节点 (Title level, 0 indicates non-title node)
        title: 标题文本 (Title text)
        block_number: 块内序号 (Block number)
        knowledge_summary: 核心知识点和摘要 (Core knowledge points and summary)
        page_content: 节点内容 (Node content)
        embedding: 节点的向量表示 (Vector representation of the node)
        children: 子节点ID列表 (List of child node IDs)
        metadata: 其他元数据 (Other metadata)
    """

    def __init__(
        self,
        page_content: str = "",
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        level: int = 0,
        title: str = "",
        block_number: str = "",
        knowledge_summary: str = "",
        embedding: Optional[List[float]] = None,
        children: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化层级文档节点 (Initialize hierarchical document node)"""
        # 基本属性 (Basic attributes)
        self.id = id if id else str(uuid.uuid4())
        self.parent_id = parent_id
        self.level = level
        self.title = title
        self.block_number = block_number
        self.knowledge_summary = knowledge_summary
        self.page_content = page_content
        self.embedding = embedding if embedding is not None else []
        self.children = children if children is not None else []
        self.metadata = metadata if metadata is not None else {}

    def get_semantic_representation(self) -> str:
        """获取节点的语义表示，用于生成嵌入向量 (Get semantic representation of the node for generating embedding vectors)"""
        return f"{self.title}\n{self.knowledge_summary}".strip()

    def update_embedding(self, embedding_model: Any) -> None:
        """使用提供的嵌入模型更新节点的向量表示 (Update node's vector representation using the provided embedding model)"""
        text = self.get_semantic_representation()
        self.embedding = embedding_model.embed_query(text)

    def add_child(self, child_id: str) -> None:
        """添加子节点ID到子节点列表 (Add child node ID to the children list)"""
        if child_id not in self.children:
            self.children.append(child_id)

    def remove_child(self, child_id: str) -> bool:
        """从子节点列表中移除指定子节点ID (Remove specified child node ID from the children list)"""
        if child_id in self.children:
            self.children.remove(child_id)
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为字典表示 (Convert node to dictionary representation)"""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "level": self.level,
            "title": self.title,
            "block_number": self.block_number,
            "knowledge_summary": self.knowledge_summary,
            "page_content": self.page_content,
            "children": self.children,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MyNode":
        """从字典创建节点实例 (Create node instance from dictionary)"""
        return cls(
            page_content=data.get("page_content", ""),
            id=data.get("id"),
            parent_id=data.get("parent_id"),
            level=data.get("level", 0),
            title=data.get("title", ""),
            block_number=data.get("block_number", ""),
            knowledge_summary=data.get("knowledge_summary", ""),
            embedding=data.get("embedding", []),
            children=data.get("children", []),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """返回节点的字符串表示 (Return string representation of the node)"""
        preview = (
            self.page_content[:50] + "..."
            if len(self.page_content) > 50
            else self.page_content
        )
        return f"[{self.id}] {self.title} - {preview}"

    def __repr__(self) -> str:
        """提供详细的对象表示，方便调试 (Provide detailed object representation for debugging)"""
        return (
            f"MyNode(id='{self.id}', level={self.level}, "
            f"title='{self.title}', children={len(self.children)})"
        )


# from langchain_core.documents import Document

# def to_langchain_document(self) -> Document:
#     """转换为LangChain的Document对象 (Convert to LangChain Document object)"""
#     metadata = {
#         "doc_id": self.id,
#         "parent_id": self.parent_id,
#         "level": self.level,
#         "title": self.title,
#         "block_number": self.block_number,
#         "knowledge_summary": self.knowledge_summary,
#         "children": self.children,
#         "embedding": self.embedding
#     }
#     # 合并其他metadata (Merge other metadata)
#     metadata.update(self.metadata)

#     return Document(page_content=self.page_content, metadata=metadata)

# # 添加到MyNode类 (Add to MyNode class)
# MyNode.to_langchain_document = to_langchain_document
