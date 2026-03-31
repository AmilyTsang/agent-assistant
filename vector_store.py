from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List

class SimpleEmbeddings:
    """简单的本地嵌入类，使用 TF-IDF 向量化文本"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384)  # 限制特征数量
        self.is_fitted = False  # 标记向量器是否已经训练过
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档列表转换为嵌入向量"""
        try:
            if not self.is_fitted:
                # 第一次调用，训练向量器
                vectors = self.vectorizer.fit_transform(texts).toarray()
                self.is_fitted = True
            else:
                # 后续调用，使用已训练的向量器
                vectors = self.vectorizer.transform(texts).toarray()
            return vectors.tolist()
        except Exception as e:
            print(f"嵌入文档时出错: {e}")
            # 如果失败，返回零向量
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """将查询文本转换为嵌入向量"""
        try:
            if not self.is_fitted:
                # 如果向量器还没有训练过，返回零向量
                return [0.0] * 384
            # 使用已训练的向量器
            vector = self.vectorizer.transform([text]).toarray()
            return vector[0].tolist()
        except Exception as e:
            print(f"嵌入查询时出错: {e}")
            # 如果失败，返回零向量
            return [0.0] * 384
    
    def __call__(self, text: str) -> List[float]:
        """使 SimpleEmbeddings 成为可调用对象，用于 FAISS 内部调用"""
        return self.embed_query(text)

class VectorStoreManager:
    """向量存储管理器，用于文档向量化和检索"""
    
    def __init__(self, api_key=None, base_url=None):
        """初始化向量存储管理器"""
        # 初始化嵌入模型 - 使用简单的本地嵌入
        self.embeddings = SimpleEmbeddings()
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # 初始化向量存储
        self.vector_store = None
    
    def add_document(self, text, metadata=None):
        """添加文档到向量存储"""
        try:
            # 分割文本
            texts = self.text_splitter.split_text(text)
            
            # 添加元数据
            metadatas = [metadata] * len(texts) if metadata else None
            
            # 创建或更新向量存储
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
            else:
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
            
            return f"文档已成功添加到向量存储，共分割为 {len(texts)} 个片段"
        except Exception as e:
            return f"添加文档到向量存储时出错: {str(e)}"
    
    def search(self, query, k=3):
        """从向量存储中检索相关内容"""
        try:
            if self.vector_store is None:
                return "向量存储为空，请先添加文档"
            
            # 检索相关内容
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(f"结果 {i+1}:\n{result.page_content}\n")
            
            return "\n".join(formatted_results)
        except Exception as e:
            return f"检索文档时出错: {str(e)}"
    
    def clear(self):
        """清空向量存储"""
        try:
            self.vector_store = None
            # 重置嵌入模型，以便下次添加文档时重新训练
            self.embeddings = SimpleEmbeddings()
            return "向量存储已清空"
        except Exception as e:
            return f"清空向量存储时出错: {str(e)}"
