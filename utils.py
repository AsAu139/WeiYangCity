import json
import re
from sympy import symbols, solve, Eq

def clean_text(text: str) -> str:
    """清理物理题目中的特殊字符，规范化 LaTeX 格式，提高检索匹配精度。"""
    if not text:
        return ""
    
    # 1. 基础清理：去除首尾空格，合并多余空格
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # 2. 符号规范化：处理物理题目中常见的混写符号
    text = text.replace("×", "\\times").replace("·", "\\cdot")
    text = text.replace("÷", "\\div").replace("≤", "\\le").replace("≥", "\\ge")
    
    # 3. LaTeX 符号间距处理：
    # 检索时，"$F = ma$" 和 "$F=ma$" 应该被视为一样的。
    # 我们统一把符号两边的空格去掉，方便计算相似度。
    text = re.sub(r'\s*=\s*', '=', text)
    text = re.sub(r'\s*\+\s*', '+', text)
    text = re.sub(r'\s*-\s*', '-', text)
    
    # 4. 纠正 LaTeX 语法的小瑕疵
    # 有时候 OCR 会把 \times 识别成 \ times (多了空格)
    text = text.replace("\\ ", "\\")
    
    return text

def load_and_clean_data(file_path: str):
    """加载并清洗训练集。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # 清理题目文本
            item['question'] = clean_text(item.get('question', ""))
            # 如果 train_data 里的推理过程也乱，也可以顺便清理一下
            if 'reasoning_process' in item:
                item['reasoning_process'] = item['reasoning_process'].strip()
        return data
    except Exception as e:
        print(f"读取数据失败: {e}")
        return []

def solve_physics_eqs(eq_strings: list, var_strings: list):
    """
    (保留作为后期研究) 利用 SymPy 辅助计算
    """
    try:
        # 这里以后可以接入更高级的解析逻辑
        return "逻辑待后续开发"
    except:
        return None