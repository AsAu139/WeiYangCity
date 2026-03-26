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

    _allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    _allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()

    def solve(self, item: Dict[str, Any]) -> Dict[str, str]:
        """解单题，返回赛题要求的标准字段。"""
        q_id = str(item.get("question_id", ""))
        q_type = str(item.get("type", ""))
        difficulty = str(item.get("difficulty", ""))
        question = str(item.get("question", "")).strip()
        image_path = item.get("image")

        reasoning_parts: List[str] = []
        reasoning_parts.append(f"题目类型: {q_type or '未知'}")
        reasoning_parts.append(f"难度: {difficulty or '未知'}")
        if image_path:
            reasoning_parts.append("检测到 image 字段（含图题）。baseline 仅记录路径，不处理图像内容。")

        answer = "请根据题意作答"

        # 基础能力: 仅做简单算式求值，作为可扩展起点。
        expr = self._extract_math_expression(question)
        if expr:
            try:
                value = self._safe_eval(expr)
                answer = str(value)
                reasoning_parts.append(f"识别到算式: {expr}")
                reasoning_parts.append("使用安全表达式求值器完成计算。")
                reasoning_parts.append(f"计算结果: {answer}")
            except Exception as exc:
                reasoning_parts.append(f"识别到算式: {expr}")
                reasoning_parts.append(f"计算失败，原因: {exc}")
                reasoning_parts.append("已退回基础回答模板。")
        else:
            reasoning_parts.append("未识别到可直接计算的标准算式。")
            reasoning_parts.append("该 baseline 不做复杂推理，等待选手扩展。")

        reasoning = "\n".join(reasoning_parts)
        if len(reasoning) > self.config.max_reasoning_chars:
            reasoning = reasoning[: self.config.max_reasoning_chars].rstrip() + "..."

        return {
            "question_id": q_id,
            "reasoning_process": reasoning,
            "answer": answer,
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
