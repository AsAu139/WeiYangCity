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

import ast
from dataclasses import dataclass
import operator
import os
import re
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass
class BaselineConfig:
    """基础配置。默认可离线运行，外部 API 仅作为可选增强。"""

    max_reasoning_chars: int = 1200
    enable_llm: bool = True
    llm_timeout: float = 30.0
    llm_model: str = "kimi-k2.5"
    llm_base_url: str = "https://api.moonshot.cn/v1"
    api_key_env: str = "KIMI_API_KEY"


class BaselineAgent:
    """一个可实例化的基础智能体。"""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self._allowed_ops: dict[type[ast.operator], Callable[[float, float], float]] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        self._allowed_unary_ops: dict[type[ast.unaryop], Callable[[float], float]] = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }
        self._client: Any | None = None
        self._client_init_error: str | None = None

        if load_dotenv is not None:
            load_dotenv()

    def solve(self, item: dict[str, Any]) -> dict[str, str]:
        """解单题，优先走本地兜底，再按条件调用可选 LLM。"""
        q_id = str(item.get("question_id", ""))
        q_type = str(item.get("type", "")).strip()
        difficulty = str(item.get("difficulty", "")).strip()
        question = str(item.get("question", "")).strip()
        image_path = item.get("image")

        if not question:
            return {
                "question_id": q_id,
                "reasoning_process": "题目内容为空，无法作答。",
                "answer": "无法确定",
            }

        local_result = self._solve_by_rules(question=question, q_type=q_type, image_path=image_path)
        if local_result is not None:
            local_result["question_id"] = q_id
            return local_result

        llm_response = self._solve_by_llm(
            question=question,
            q_type=q_type,
            difficulty=difficulty,
            image_path=image_path,
        )
        if llm_response is not None:
            reasoning_process = self._truncate_reasoning(llm_response)
            answer = self._extract_answer(reasoning_process)
            return {
                "question_id": q_id,
                "reasoning_process": reasoning_process,
                "answer": answer,
            }

        fallback_reasoning = self._truncate_reasoning(
            self._build_fallback_reasoning(question=question, image_path=image_path)
        )
        return {
            "question_id": q_id,
            "reasoning_process": fallback_reasoning,
            "answer": "无法确定",
        }

    def _solve_by_rules(
        self, question: str, q_type: str, image_path: Any | None
    ) -> dict[str, str] | None:
        """优先处理可直接提取表达式的基础计算题。"""
        if image_path:
            return None

        expr = self._extract_math_expression(question)
        if expr is None:
            return None

        try:
            value = self._safe_eval(expr)
        except Exception:
            return None

        answer = self._format_number(value)
        reasoning = (
            "已知条件：题目中可直接提取出数学表达式 "
            f"${expr.replace('**', '^')}$。\n"
            "推导步骤：该题可按四则运算顺序直接计算，无需额外物理建模。\n"
            f"数值计算：${expr.replace('**', '^')} = {answer}$。\n"
            f"答案：{answer}"
        )

        return {
            "reasoning_process": self._truncate_reasoning(reasoning),
            "answer": answer,
        }

    def _solve_by_llm(
        self,
        question: str,
        q_type: str,
        difficulty: str,
        image_path: Any | None,
    ) -> str | None:
        client = self._get_client()
        if client is None:
            return None

        system_prompt = (
            "你是一个精通大学基础物理的助教。"
            "请严格按照以下结构回答：\n"
            "已知条件：...\n"
            "推导步骤：...\n"
            "数值计算：...\n"
            "答案：...\n"
            "请保证最后一行必须是“答案：具体结果”，不要添加额外结束语。"
        )

        user_content = (
            f"题目类型：{q_type or '未知'}\n"
            f"难度：{difficulty or '未知'}\n"
            f"题目内容：{question}"
        )
        if image_path:
            user_content += (
                f"\n补充说明：本题给出了图片路径 `{image_path}`。"
                "当前基线版本不会直接读取图片，请仅依据题干文字中可确定的信息作答；"
                "若关键信息依赖图片，请明确说明。"
            )

        try:
            completion = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=1,
                timeout=self.config.llm_timeout,
            )
        except Exception as exc:
            self._client_init_error = f"LLM 调用失败：{exc}"
            return None

        raw_content = completion.choices[0].message.content
        if isinstance(raw_content, str):
            return raw_content.strip()
        return None

    def _get_client(self) -> Any | None:
        if not self.config.enable_llm:
            self._client_init_error = "配置中已关闭 LLM 调用。"
            return None

        if self._client is not None:
            return self._client

        if OpenAI is None:
            self._client_init_error = "未安装 openai，跳过 LLM 调用。"
            return None

        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            self._client_init_error = f"环境变量 {self.config.api_key_env} 未设置。"
            return None

        try:
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.llm_base_url,
            )
            return self._client
        except Exception as exc:
            self._client_init_error = f"LLM 客户端初始化失败：{exc}"
            return None

    def _extract_answer(self, response_text: str) -> str:
        patterns = [
            r"^\s*答案\s*[:：]\s*(.+?)\s*$",
            r"^\s*最终答案\s*[:：]\s*(.+?)\s*$",
            r"^\s*\*\*答案\*\*\s*[:：]?\s*(.+?)\s*$",
        ]
        for line in reversed([line.strip() for line in response_text.splitlines() if line.strip()]):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    return match.group(1).strip()

        for pattern in patterns:
            matches = re.findall(pattern, response_text, flags=re.MULTILINE)
            if matches:
                return matches[-1].strip()

        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        return lines[-1] if lines else "无法确定"

    def _build_fallback_reasoning(self, question: str, image_path: Any | None) -> str:
        lines = [
            f"已知条件：题目内容为“{question}”。",
            "推导步骤：当前基线未能从题干中提取出可直接计算的表达式，且外部模型不可用，因此无法完成可靠推导。",
        ]
        if image_path:
            lines.append(
                f"补充说明：题目还引用了图片 `{image_path}`，但当前基线版本不会直接读取图片内容。"
            )
        if self._client_init_error:
            lines.append(f"模型状态：{self._client_init_error}")
        lines.append("数值计算：无可执行的可靠计算。")
        lines.append("答案：无法确定")
        return "\n".join(lines)

    def _extract_math_expression(self, text: str) -> str | None:
        """仅提取由数字与常见运算符构成的基础表达式。"""
        normalized = (
            text.replace("×", "*")
            .replace("÷", "/")
            .replace("（", "(")
            .replace("）", ")")
            .replace("−", "-")
        )
        candidates = re.findall(r"[\d\.\s\+\-\*/\(\)\%\^]+", normalized)
        for candidate in candidates:
            expr = re.sub(r"\s+", "", candidate)
            if not expr:
                continue
            if not re.search(r"\d", expr):
                continue
            if not re.search(r"[\+\-\*/\^%]", expr):
                continue
            if expr[0] in "*/%^" or expr[-1] in "+-*/%^":
                continue
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

    def _truncate_reasoning(self, text: str) -> str:
        if len(text) <= self.config.max_reasoning_chars:
            return text
        truncated = text[: self.config.max_reasoning_chars].rstrip()
        if "答案：" in text and "答案：" not in truncated:
            answer = self._extract_answer(text)
            suffix = f"\n...\n答案：{answer}"
            keep = max(self.config.max_reasoning_chars - len(suffix), 0)
            truncated = text[:keep].rstrip() + suffix
        return truncated

    def _format_number(self, value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.10g}"
