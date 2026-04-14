"""成员 A 使用的批量评测脚本。

功能：
1. 读取题目 JSON 列表，循环调用 ``BaselineAgent.solve``。
2. 生成评测系统可用的 ``submission.json``。
3. 若输入数据包含标准答案，则自动计算准确率并导出错题集。
4. 支持失败重试、请求节流和限制评测样本数，方便本地调试。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from baseline_agent import BaselineAgent, BaselineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量调用 BaselineAgent 并生成评测结果。")
    parser.add_argument(
        "--input",
        default="train_data/train_data.json",
        help="输入题目 JSON 路径，文件内容需为题目对象列表。",
    )
    parser.add_argument(
        "--output",
        default="submission.generated.json",
        help="预测结果输出路径。",
    )
    parser.add_argument(
        "--metrics-output",
        default="metrics.json",
        help="评测指标输出路径；若无标准答案则不会生成有效指标。",
    )
    parser.add_argument(
        "--wrong-output",
        default="wrong_answers.json",
        help="错题集输出路径；若无标准答案则输出为空列表。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅评测前 N 题，便于调试。",
    )
    parser.add_argument(
        "--retry-times",
        type=int,
        default=3,
        help="单题失败后的最大重试次数。",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="重试前等待秒数。",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.0,
        help="每题完成后额外等待秒数，适合共享 API key 时限流。",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="关闭 LLM，只测试本地规则与兜底逻辑。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="覆盖默认模型名。",
    )
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"{path} 中的内容不是题目列表。")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} 第 {idx + 1} 项不是对象。")
        normalized.append(item)
    return normalized


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_answer(answer: Any) -> str:
    text = str(answer if answer is not None else "").strip()
    if not text:
        return ""

    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("：", ":").replace("，", ",")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip("。.;；,，")

    upper_letter = re.fullmatch(r"[A-Da-d]", text)
    if upper_letter:
        return upper_letter.group(0).upper()

    return text


def compare_answers(predicted: Any, gold: Any) -> bool:
    pred = normalize_answer(predicted)
    truth = normalize_answer(gold)
    if pred == truth:
        return True

    # 对纯数值做一个宽松比较，减少 7 与 7.0 一类差异造成的误判。
    try:
        return abs(float(pred) - float(truth)) <= 1e-8
    except ValueError:
        return False


def run_agent(
    items: list[dict[str, Any]],
    agent: BaselineAgent,
    retry_times: int,
    retry_sleep: float,
    request_sleep: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    total = len(items)
    for index, item in enumerate(items, start=1):
        qid = str(item.get("question_id", f"UNKNOWN_{index}"))
        last_error = ""

        for attempt in range(1, retry_times + 1):
            try:
                result = agent.solve(item)
                result.setdefault("question_id", qid)
                predictions.append(result)
                print(f"[{index}/{total}] question_id={qid} status=ok attempt={attempt}")
                break
            except Exception as exc:  # pragma: no cover - runtime protection
                last_error = str(exc)
                print(
                    f"[{index}/{total}] question_id={qid} status=error attempt={attempt} error={exc}",
                    file=sys.stderr,
                )
                if attempt < retry_times and retry_sleep > 0:
                    time.sleep(retry_sleep)
        else:
            fallback = {
                "question_id": qid,
                "reasoning_process": f"批量评测执行失败：{last_error}",
                "answer": "无法确定",
            }
            predictions.append(fallback)
            failures.append(
                {
                    "question_id": qid,
                    "error": last_error,
                    "raw_item": item,
                }
            )

        if request_sleep > 0:
            time.sleep(request_sleep)

    return predictions, failures


def evaluate_predictions(
    items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_map = {str(item.get("question_id", "")): item for item in items}
    has_gold = all("answer" in item for item in items)

    metrics: dict[str, Any] = {
        "total": len(items),
        "predicted": len(predictions),
        "failed_count": len(failures),
        "has_gold_answer": has_gold,
    }

    wrong_items: list[dict[str, Any]] = []
    if not has_gold:
        metrics["message"] = "输入数据未包含标准答案，仅生成 submission。"
        return metrics, wrong_items

    correct = 0
    for pred in predictions:
        qid = str(pred.get("question_id", ""))
        source_item = gold_map.get(qid, {})
        gold_answer = source_item.get("answer", "")
        is_correct = compare_answers(pred.get("answer", ""), gold_answer)
        if is_correct:
            correct += 1
            continue

        wrong_items.append(
            {
                "question_id": qid,
                "type": source_item.get("type", ""),
                "difficulty": source_item.get("difficulty", ""),
                "question": source_item.get("question", ""),
                "predicted_answer": pred.get("answer", ""),
                "gold_answer": gold_answer,
                "reasoning_process": pred.get("reasoning_process", ""),
            }
        )

    metrics["correct"] = correct
    metrics["wrong"] = len(wrong_items)
    metrics["accuracy"] = round(correct / len(items), 6) if items else 0.0
    return metrics, wrong_items


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    metrics_path = Path(args.metrics_output)
    wrong_path = Path(args.wrong_output)

    if not input_path.exists():
        print(f"输入文件不存在：{input_path}", file=sys.stderr)
        return 1

    items = load_items(input_path)
    if args.limit is not None:
        items = items[: args.limit]

    config = BaselineConfig(enable_llm=not args.disable_llm)
    if args.model:
        config.llm_model = args.model

    agent = BaselineAgent(config)
    predictions, failures = run_agent(
        items=items,
        agent=agent,
        retry_times=max(args.retry_times, 1),
        retry_sleep=max(args.retry_sleep, 0.0),
        request_sleep=max(args.request_sleep, 0.0),
    )

    dump_json(output_path, predictions)
    metrics, wrong_items = evaluate_predictions(items, predictions, failures)
    metrics["input_path"] = str(input_path)
    metrics["output_path"] = str(output_path)
    dump_json(metrics_path, metrics)
    dump_json(wrong_path, wrong_items)

    print("\n评测完成：")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if wrong_items:
        print(f"错题数：{len(wrong_items)}，已写入 {wrong_path}")
    if failures:
        print(f"运行失败题数：{len(failures)}，这些题已按兜底结果写入 submission。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
