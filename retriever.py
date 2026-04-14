import jieba  # 题目是中文，用 jieba 分词
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QuestionRetriever:
    def __init__(self, train_data):
        self.train_data = train_data
        self.questions = [item['question'] for item in train_data]
        # 初始化 TF-IDF 向量化工具
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: jieba.lcut(x))
        # 预先计算训练集所有题目的向量
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def search(self, query: str, top_k: int = 1):
        """寻找最相似的题目。"""
        # 将新题目向量化
        query_vec = self.vectorizer.transform([query])
        # 计算余弦相似度
        similarities = cosine_similarity(query_vec, self.question_vectors).flatten()
        # 找到最匹配的索引
        best_idx = np.argsort(similarities)[-top_k:]
        
        results = []
        for idx in best_idx:
            if similarities[idx] > 0.3:  # 设置一个阈值，太不像的就不给了
                results.append(self.train_data[idx])
        return results