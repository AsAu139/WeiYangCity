"""比赛 baseline 智能体实现（可实例化类，不使用脚本入口）。

输入接口（单题）:
{
  "question_id": str,
  "type": str,
  "difficulty": str,
    "question": str,
    "image": str  # 可选字段，仅含图题提供，内容为图片相对路径
}

输出接口（单题）:
{
  "question_id": str,
  "reasoning_process": str,
  "answer": str
}
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import operator
import re
from typing import Any, Dict, List
import os
from dotenv import load_dotenv  # 新增：用于读取 .env 文件
from openai import OpenAI  # 导入库

@dataclass
class BaselineConfig:
    """基础配置。当前 baseline 默认不依赖外部 API。"""

    max_reasoning_chars: int = 1200


class BaselineAgent:
    """一个可实例化的基础智能体。

    设计目标:
    - 保持赛题输入输出字段不变
    - 逻辑简单、稳定，方便选手二次开发
    - 不作为脚本运行，供评测框架直接实例化调用
    """
    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        
        # --- 新增安全加载逻辑 ---
        load_dotenv()  # 这一行会自动寻找根目录下的 .env 并读取
        api_key = os.getenv("KIMI_API_KEY") # 这里的字符串必须和 .env 里等号左边一致
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )
        # ----------------------



    def solve(self, item: Dict[str, Any]) -> Dict[str, str]:
        """解单题，使用优化的 Prompt 调用 Kimi API。"""
        q_id = str(item.get("question_id", ""))
        q_type = str(item.get("type", ""))
        difficulty = str(item.get("difficulty", ""))
        question = str(item.get("question", "")).strip()
        image_path = item.get("image")

        # 1. 设定系统级指令
        system_prompt = (
            "你是一个精通清华大学《基础物理学》课程的资深教授。\n"
            "你的任务是为学生提供严谨、准确的题目解答。请务必保持推导过程的严谨，不要发散。请遵循以下规范：\n"
            "1. 逻辑严密：从基本定律（如牛顿定律、基尔霍夫定律 KCL/KVL）出发进行推导。\n"
            "2. 公式规范：所有的数学公式、物理量、数值单位必须使用 LaTeX 格式（例如：$F=ma$, $10\\Omega$）。\n"
            "3. 结构清晰：推理过程需包含‘已知条件’、‘推导步骤’、‘数值计算’。\n"
            "4. 最终答案：请在回复的最后一行，独立起行并严格以‘答案：[具体结果]’的形式结束。"
        )

        # 2. 构造用户输入（提供题目信息）
        user_content = f"题目类型：{q_type}\n难度：{difficulty}\n题目内容：{question}"
        if image_path:
            user_content += f"\n（注意：本题配有图示，路径为 {image_path}，请根据题目文字描述及常识进行推理解析。）"

        try:
            completion = self.client.chat.completions.create(
                model="kimi-k2.5",  # 建议根据你的 API 实际权限确认模型名（模型名在哪看，我已急哭）
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=1,  # Kimi-k2.5模型的temperature只允许设置为1
            )
            
            full_response = completion.choices[0].message.content
            
            # 3. 简单的答案提取逻辑
            # 尝试寻找最后一行“答案：”之后的内容
            lines = full_response.strip().split('\n')
            last_line = lines[-1]
            if "答案：" in last_line:
                answer = last_line.split("答案：")[-1].strip()
            else:
                answer = "请见推理过程末尾"

            return {
                "question_id": q_id,
                "reasoning_process": full_response,
                "answer": answer,
            }

        except Exception as e:
            return {
                "question_id": q_id,
                "reasoning_process": f"API 调用发生错误：{str(e)}",
                "answer": "ERROR",
            }

    def _extract_math_expression(self, text: str) -> str | None:
        # 仅提取由数字与常见运算符构成的最基础表达式。
        normalized = text.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")")
        candidates = re.findall(r"[\d\.\s\+\-\*/\(\)\%\^]+", normalized)
        for candidate in candidates:
            expr = candidate.strip()
            if not expr:
                continue
            if re.search(r"\d", expr) and re.search(r"[\+\-\*/\^%]", expr):
                return expr.replace("^", "**")
        return None

    def _safe_eval(self, expr: str) -> float:
        node = ast.parse(expr, mode="eval")
        value = self._eval_ast(node.body)
        return float(value)

    def _eval_ast(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("表达式包含非法常量")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._allowed_ops:
                raise ValueError("表达式包含不支持的二元运算")
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)
            return float(self._allowed_ops[op_type](left, right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._allowed_unary_ops:
                raise ValueError("表达式包含不支持的一元运算")
            value = self._eval_ast(node.operand)
            return float(self._allowed_unary_ops[op_type](value))

        raise ValueError("表达式语法不受支持")


# agent 输出格式还需规范。