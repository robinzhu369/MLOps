"""Data type classification tools — rule-based and LLM-based."""

import json
from typing import Any, Callable

from core.constants import DataType, PipelineType, FieldRole


# Rule scoring weights
_TRANSACTION_INDICATORS = [
    "transaction_id", "txn_id", "trans_id",
    "transaction_time", "txn_time", "trans_time", "trans_date",
    "transaction_amount", "txn_amt", "trans_amt",
    "debit_credit_flag", "debit_credit", "dc_flag",
    "merchant_category", "mcc", "channel",
    "balance_after_txn", "balance",
    "counterparty", "counterparty_account",
]

_LABEL_INDICATORS = [
    "bad_flag", "label", "target", "is_default", "is_overdue",
    "bad", "default_flag", "overdue_flag", "y",
]

_OBSERVATION_TIME_INDICATORS = [
    "apply_date", "application_date", "observation_date",
    "申请日期", "观察点",
]


def classify_data_type_by_rules(
    file_name: str,
    columns: list[str],
    column_profiles: dict,
    field_semantics: dict | None = None,
) -> dict:
    """Classify data type using rule-based scoring.

    Returns:
        dict with detected_data_type, confidence, reasoning, detected_roles.
    """
    col_lower = [c.lower() for c in columns]
    n_cols = len(columns)

    # Score each data type
    scores = {
        DataType.TRANSACTION_FLOW_TABLE: 0.0,
        DataType.STRUCTURED_MODELING_TABLE: 0.0,
        DataType.MAIN_TABLE: 0.0,
        DataType.AUXILIARY_TABLE: 0.0,
    }

    detected_roles = {
        "account_key": None,
        "customer_key": None,
        "transaction_time_col": None,
        "amount_col": None,
        "direction_col": None,
        "label_col": None,
        "base_time_col": None,
    }

    # Check for transaction flow indicators
    txn_matches = 0
    for indicator in _TRANSACTION_INDICATORS:
        for i, col in enumerate(col_lower):
            if indicator in col:
                txn_matches += 1
                # Assign detected roles
                if "time" in indicator or "date" in indicator:
                    detected_roles["transaction_time_col"] = columns[i]
                elif "amount" in indicator or "amt" in indicator:
                    detected_roles["amount_col"] = columns[i]
                elif "debit" in indicator or "credit" in indicator or "dc_flag" in indicator:
                    detected_roles["direction_col"] = columns[i]
                break

    # Check for account/customer keys
    for i, col in enumerate(col_lower):
        if "account_id" in col or "account_no" in col or "acct_id" in col:
            detected_roles["account_key"] = columns[i]
        if "customer_id" in col or "cust_id" in col or "client_id" in col:
            detected_roles["customer_key"] = columns[i]

    # Check for label
    has_label = False
    for indicator in _LABEL_INDICATORS:
        for i, col in enumerate(col_lower):
            if indicator == col or indicator in col:
                detected_roles["label_col"] = columns[i]
                has_label = True
                break
        if has_label:
            break

    # Check for observation time
    for indicator in _OBSERVATION_TIME_INDICATORS:
        for i, col in enumerate(col_lower):
            if indicator in col:
                detected_roles["base_time_col"] = columns[i]
                break

    # Also use field_semantics if available
    if field_semantics:
        for col_key, info in field_semantics.items():
            role = info.get("role", "")
            col_name = col_key.split(":")[-1] if ":" in col_key else col_key
            if role == FieldRole.LABEL:
                detected_roles["label_col"] = col_name
                has_label = True
            elif role == FieldRole.ACCOUNT_KEY:
                detected_roles["account_key"] = col_name
            elif role == FieldRole.CUSTOMER_KEY:
                detected_roles["customer_key"] = col_name
            elif role == FieldRole.TRANSACTION_TIME:
                detected_roles["transaction_time_col"] = col_name
            elif role == FieldRole.OBSERVATION_TIME:
                detected_roles["base_time_col"] = col_name
            elif role == FieldRole.AMOUNT:
                detected_roles["amount_col"] = col_name
            elif role == FieldRole.DIRECTION:
                detected_roles["direction_col"] = col_name

    # Check row-per-account ratio (transaction tables have many rows per account)
    account_col = detected_roles["account_key"] or detected_roles["customer_key"]
    high_row_per_key = False
    if account_col and account_col in column_profiles:
        profile = column_profiles[account_col]
        unique_rate = profile.get("unique_rate", 1.0)
        if unique_rate < 0.3:  # Many rows per key
            high_row_per_key = True

    # Scoring logic
    if txn_matches >= 3:
        scores[DataType.TRANSACTION_FLOW_TABLE] += 0.5
    elif txn_matches >= 1:
        scores[DataType.TRANSACTION_FLOW_TABLE] += 0.2

    if detected_roles["transaction_time_col"] and detected_roles["amount_col"]:
        scores[DataType.TRANSACTION_FLOW_TABLE] += 0.3

    if high_row_per_key:
        scores[DataType.TRANSACTION_FLOW_TABLE] += 0.2

    if has_label:
        scores[DataType.STRUCTURED_MODELING_TABLE] += 0.4
        scores[DataType.MAIN_TABLE] += 0.3

    if detected_roles["base_time_col"] and has_label:
        scores[DataType.STRUCTURED_MODELING_TABLE] += 0.2
        scores[DataType.MAIN_TABLE] += 0.2

    if not has_label and txn_matches == 0:
        scores[DataType.AUXILIARY_TABLE] += 0.3

    # Determine winner
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Normalize confidence
    total = sum(scores.values())
    confidence = best_score / total if total > 0 else 0.0
    confidence = min(confidence, 0.99)

    # Build reasoning
    reasoning_parts = []
    if txn_matches > 0:
        reasoning_parts.append(f"匹配到 {txn_matches} 个交易流水指标字段")
    if has_label:
        reasoning_parts.append(f"发现标签字段 {detected_roles['label_col']}")
    if high_row_per_key:
        reasoning_parts.append("同一主键对应多行记录")
    if detected_roles["base_time_col"]:
        reasoning_parts.append(f"发现观察点时间字段 {detected_roles['base_time_col']}")

    return {
        "file_name": file_name,
        "detected_data_type": best_type.value,
        "confidence": round(confidence, 2),
        "reasoning_summary": "；".join(reasoning_parts) if reasoning_parts else "基于字段特征综合判断",
        "detected_roles": detected_roles,
        "scores": {k.value: round(v, 3) for k, v in scores.items()},
    }


# LLM prompt for data type classification
_DATA_TYPE_PROMPT = """你是一个风控数据专家。请根据以下文件信息判断数据类型。

文件名: {file_name}
行数: {n_rows}
列数: {n_cols}
字段列表: {columns}
字段语义（如有）: {field_semantics_summary}

可选数据类型:
- structured_modeling_table: 普通结构化建模宽表（一行一个客户/申请，有label字段）
- transaction_flow_table: 交易流水明细表（一行一笔交易，通常无label）
- main_table: 多表建模主表（有label，需要关联其他表）
- auxiliary_table: 辅助信息表
- unknown_or_ambiguous: 不确定

请以JSON格式返回:
{{
  "detected_data_type": "类型",
  "confidence": 0.0-1.0,
  "reasoning_summary": "判断原因"
}}

只返回JSON。"""


def classify_data_type_by_llm(
    file_name: str,
    n_rows: int,
    n_cols: int,
    columns: list[str],
    field_semantics: dict | None = None,
    llm_call: Callable[[str], str] | None = None,
) -> dict:
    """Classify data type using LLM.

    Returns:
        dict with detected_data_type, confidence, reasoning_summary.
    """
    if llm_call is None:
        return {
            "detected_data_type": DataType.UNKNOWN.value,
            "confidence": 0.0,
            "reasoning_summary": "LLM 不可用",
        }

    semantics_summary = ""
    if field_semantics:
        parts = []
        for col, info in list(field_semantics.items())[:20]:
            col_name = col.split(":")[-1] if ":" in col else col
            role = info.get("role", "unknown")
            parts.append(f"{col_name}={role}")
        semantics_summary = ", ".join(parts)

    prompt = _DATA_TYPE_PROMPT.format(
        file_name=file_name,
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns[:30],
        field_semantics_summary=semantics_summary or "无",
    )

    response = llm_call(prompt)
    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        result = {
            "detected_data_type": DataType.UNKNOWN.value,
            "confidence": 0.0,
            "reasoning_summary": "LLM 响应解析失败",
        }

    return result


def merge_data_type_classification(
    rule_result: dict,
    llm_result: dict,
) -> dict:
    """Merge rule-based and LLM-based classification results.

    Strategy:
    - If both agree, use the agreed type with boosted confidence.
    - If they disagree, use the one with higher confidence.
    - If both have low confidence, mark as need_human_review.
    """
    rule_type = rule_result.get("detected_data_type", DataType.UNKNOWN.value)
    llm_type = llm_result.get("detected_data_type", DataType.UNKNOWN.value)
    rule_conf = rule_result.get("confidence", 0.0)
    llm_conf = llm_result.get("confidence", 0.0)

    if rule_type == llm_type and rule_type != DataType.UNKNOWN.value:
        # Agreement — boost confidence
        final_type = rule_type
        final_conf = min(0.99, (rule_conf + llm_conf) / 2 + 0.1)
        reasoning = f"规则和LLM一致判断为 {final_type}"
    elif llm_type == DataType.UNKNOWN.value or llm_conf == 0.0:
        # LLM unavailable or failed, use rules
        final_type = rule_type
        final_conf = rule_conf
        reasoning = rule_result.get("reasoning_summary", "")
    elif rule_conf >= llm_conf:
        final_type = rule_type
        final_conf = rule_conf
        reasoning = f"规则判断({rule_conf:.2f})优先于LLM({llm_conf:.2f}): {rule_result.get('reasoning_summary', '')}"
    else:
        final_type = llm_type
        final_conf = llm_conf
        reasoning = f"LLM判断({llm_conf:.2f})优先于规则({rule_conf:.2f}): {llm_result.get('reasoning_summary', '')}"

    need_human_review = final_conf < 0.7 or rule_type != llm_type

    # Determine recommended pipeline
    pipeline_map = {
        DataType.STRUCTURED_MODELING_TABLE.value: PipelineType.STRUCTURED_MODELING.value,
        DataType.TRANSACTION_FLOW_TABLE.value: PipelineType.TRANSACTION_FEATURE.value,
        DataType.MAIN_TABLE.value: PipelineType.MAIN_PLUS_TRANSACTION.value,
        DataType.AUXILIARY_TABLE.value: PipelineType.MANUAL_CONFIGURATION.value,
        DataType.UNKNOWN.value: PipelineType.MANUAL_CONFIGURATION.value,
    }

    return {
        "file_name": rule_result.get("file_name", ""),
        "detected_data_type": final_type,
        "confidence": round(final_conf, 2),
        "reasoning_summary": reasoning,
        "recommended_pipeline": pipeline_map.get(final_type, PipelineType.MANUAL_CONFIGURATION.value),
        "detected_roles": rule_result.get("detected_roles", {}),
        "need_human_review": need_human_review,
        "rule_result": {"type": rule_type, "confidence": rule_conf},
        "llm_result": {"type": llm_type, "confidence": llm_conf},
    }
