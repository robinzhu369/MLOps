"""LLM-based field semantic parsing tools."""

import json
from typing import Any

from core.constants import FieldRole, RiskLevel, LEAKAGE_KEYWORDS


# Prompt template for LLM field semantic parsing
FIELD_SEMANTIC_PROMPT = """你是一个风控建模专家。请根据以下字段信息，推断每个字段的业务含义和角色。

数据文件: {file_name}
总行数: {n_rows}
总列数: {n_cols}

字段信息:
{field_info}

请为每个字段判断其角色，可选角色包括:
- label: 标签字段（如是否逾期、是否违约）
- customer_key: 客户ID
- account_key: 账户ID
- loan_key: 贷款ID
- observation_time: 观察点时间（如申请日期）
- transaction_time: 交易时间
- amount: 金额字段
- direction: 交易方向字段（借贷标志）
- sensitive: 敏感字段（身份证、手机号等）
- possible_leakage_feature: 疑似标签泄露字段（贷后表现字段）
- post_loan_feature: 贷后特征
- normal: 普通特征字段

请以JSON格式返回，格式如下:
{{
  "field_semantics": {{
    "字段名": {{
      "role": "角色",
      "business_meaning": "业务含义",
      "confidence": 0.0-1.0的置信度
    }}
  }}
}}

只返回JSON，不要其他内容。"""


def build_field_info_text(column_profiles: dict, sample_values_masked: dict) -> str:
    """Build a text description of fields for the LLM prompt.

    Args:
        column_profiles: Column profiles from metadata extraction.
        sample_values_masked: Masked sample values.

    Returns:
        Formatted text describing each field.
    """
    lines = []
    for col_name, profile in column_profiles.items():
        samples = sample_values_masked.get(col_name, [])
        line = (
            f"- {col_name}: "
            f"类型={profile.get('dtype', 'unknown')}, "
            f"缺失率={profile.get('missing_rate', 0):.1%}, "
            f"唯一值数={profile.get('unique_count', 0)}, "
            f"唯一值比例={profile.get('unique_rate', 0):.1%}"
        )
        if samples:
            line += f", 样例值={samples[:3]}"
        lines.append(line)
    return "\n".join(lines)


def build_semantic_prompt(
    file_name: str,
    n_rows: int,
    n_cols: int,
    column_profiles: dict,
    sample_values_masked: dict,
) -> str:
    """Build the full prompt for LLM field semantic parsing."""
    field_info = build_field_info_text(column_profiles, sample_values_masked)
    return FIELD_SEMANTIC_PROMPT.format(
        file_name=file_name,
        n_rows=n_rows,
        n_cols=n_cols,
        field_info=field_info,
    )


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response JSON into field semantics.

    Args:
        response_text: Raw LLM response text (should be JSON).

    Returns:
        dict with field_semantics mapping.
    """
    # Try to extract JSON from the response
    text = response_text.strip()
    if text.startswith("```"):
        # Remove markdown code block
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
        else:
            return {"field_semantics": {}}

    return result if "field_semantics" in result else {"field_semantics": result}


def parse_field_semantics_by_llm(
    file_name: str,
    n_rows: int,
    n_cols: int,
    column_profiles: dict,
    sample_values_masked: dict,
    llm_call: Any = None,
) -> dict:
    """Parse field semantics using LLM.

    Args:
        file_name: Name of the data file.
        n_rows: Number of rows.
        n_cols: Number of columns.
        column_profiles: Column profiles from metadata extraction.
        sample_values_masked: Masked sample values.
        llm_call: Callable that takes a prompt string and returns LLM response text.
                  If None, falls back to rule-based parsing.

    Returns:
        dict with dictionary_uploaded=False, field_semantics, need_human_review.
    """
    if llm_call is not None:
        prompt = build_semantic_prompt(
            file_name, n_rows, n_cols, column_profiles, sample_values_masked
        )
        response_text = llm_call(prompt)
        llm_result = parse_llm_response(response_text)
        field_semantics = llm_result.get("field_semantics", {})
    else:
        # Fallback: rule-based parsing
        field_semantics = _rule_based_field_parsing(column_profiles, sample_values_masked)

    # Assess which fields need human review
    need_human_review = False
    for col, info in field_semantics.items():
        confidence = info.get("confidence", 0.5)
        if confidence < 0.8:
            need_human_review = True
            break

    # Add risk_level to each field
    for col, info in field_semantics.items():
        role = info.get("role", FieldRole.NORMAL)
        if role in (FieldRole.POSSIBLE_LEAKAGE, FieldRole.POST_LOAN_FEATURE):
            info["risk_level"] = RiskLevel.HIGH
        elif any(kw in col.lower() for kw in LEAKAGE_KEYWORDS):
            info["risk_level"] = RiskLevel.HIGH
        else:
            info["risk_level"] = RiskLevel.NORMAL

    return {
        "dictionary_uploaded": False,
        "field_semantics": field_semantics,
        "need_human_review": need_human_review,
    }


def merge_dictionary_and_llm_semantics(
    dictionary_result: dict,
    llm_result: dict,
) -> dict:
    """Merge dictionary-based and LLM-based field semantics.

    Dictionary results take priority. LLM fills in gaps.

    Returns:
        Merged field_semantics dict.
    """
    merged = {}

    # First, add all dictionary entries (higher priority)
    for entry in dictionary_result.get("parsed_columns", []):
        col_name = entry["column_name"]
        merged[col_name] = {
            "role": entry["role"],
            "business_meaning": entry.get("business_meaning", ""),
            "confidence": entry.get("confidence", 1.0),
            "risk_level": entry.get("risk_level", RiskLevel.NORMAL),
            "from_dictionary": True,
        }

    # Then, fill in from LLM for columns not in dictionary
    llm_semantics = llm_result.get("field_semantics", {})
    for col_name, info in llm_semantics.items():
        if col_name not in merged:
            merged[col_name] = {
                "role": info.get("role", FieldRole.NORMAL),
                "business_meaning": info.get("business_meaning", ""),
                "confidence": info.get("confidence", 0.5),
                "risk_level": info.get("risk_level", RiskLevel.NORMAL),
                "from_dictionary": False,
            }

    return {"field_semantics": merged}


def _rule_based_field_parsing(
    column_profiles: dict,
    sample_values_masked: dict,
) -> dict:
    """Fallback rule-based field semantic parsing when LLM is unavailable."""
    field_semantics = {}

    for col_name, profile in column_profiles.items():
        col_lower = col_name.lower()
        role = FieldRole.NORMAL
        confidence = 0.7
        meaning = ""

        # ID fields
        if col_lower in ("customer_id", "cust_id", "cid"):
            role, confidence, meaning = FieldRole.CUSTOMER_KEY, 0.95, "客户ID"
        elif col_lower in ("account_id", "acct_id", "acc_id"):
            role, confidence, meaning = FieldRole.ACCOUNT_KEY, 0.95, "账户ID"
        elif col_lower in ("loan_id", "loan_no"):
            role, confidence, meaning = FieldRole.LOAN_KEY, 0.95, "贷款ID"

        # Label fields
        elif col_lower in ("bad_flag", "label", "target", "is_default", "is_overdue"):
            role, confidence, meaning = FieldRole.LABEL, 0.90, "标签字段"
        elif any(kw in col_lower for kw in ["bad", "default", "overdue", "逾期", "违约"]):
            role, confidence, meaning = FieldRole.POSSIBLE_LEAKAGE, 0.85, "疑似标签泄露字段"

        # Time fields
        elif col_lower in ("apply_date", "application_date", "申请日期"):
            role, confidence, meaning = FieldRole.OBSERVATION_TIME, 0.90, "申请日期/观察点"
        elif "transaction_time" in col_lower or "txn_time" in col_lower:
            role, confidence, meaning = FieldRole.TRANSACTION_TIME, 0.90, "交易时间"
        elif col_lower.endswith("_date") or col_lower.endswith("_time"):
            role, confidence, meaning = FieldRole.OBSERVATION_TIME, 0.60, "时间字段"

        # Amount fields
        elif "amount" in col_lower or "amt" in col_lower:
            role, confidence, meaning = FieldRole.AMOUNT, 0.85, "金额字段"
        elif col_lower in ("transaction_amount", "txn_amt", "loan_amount"):
            role, confidence, meaning = FieldRole.AMOUNT, 0.92, "金额字段"

        # Direction fields
        elif col_lower in ("debit_credit_flag", "dc_flag", "direction"):
            role, confidence, meaning = FieldRole.DIRECTION, 0.90, "交易方向"

        # Leakage keywords
        elif any(kw in col_lower for kw in LEAKAGE_KEYWORDS):
            role, confidence, meaning = FieldRole.POSSIBLE_LEAKAGE, 0.80, "疑似泄露字段"

        # High unique rate likely ID
        elif profile.get("unique_rate", 0) > 0.95 and profile.get("dtype") in ("object", "int64"):
            role, confidence, meaning = FieldRole.CUSTOMER_KEY, 0.50, "疑似ID字段(高唯一值)"

        field_semantics[col_name] = {
            "role": role,
            "business_meaning": meaning,
            "confidence": confidence,
        }

    return field_semantics
