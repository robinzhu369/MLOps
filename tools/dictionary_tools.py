"""Data dictionary parsing tools."""

import pandas as pd

from core.constants import FieldRole, RiskLevel, LEAKAGE_KEYWORDS


# Expected columns in a data dictionary file
DICTIONARY_REQUIRED_COLUMNS = ["table_name", "column_name"]
DICTIONARY_OPTIONAL_COLUMNS = [
    "column_cn_name", "business_meaning", "data_type", "value_range",
    "enum_mapping", "is_label", "is_id", "is_time", "is_sensitive",
    "available_time", "source_system", "remark",
]


def parse_data_dictionary(dictionary_path: str) -> dict:
    """Parse a data dictionary file and extract field semantics.

    Args:
        dictionary_path: Path to the data dictionary CSV/Excel file.

    Returns:
        dict with parsed_columns list and warnings.
    """
    df = _read_dictionary_file(dictionary_path)
    parsed_columns = []
    warnings = []

    for _, row in df.iterrows():
        table_name = str(row.get("table_name", "")).strip()
        column_name = str(row.get("column_name", "")).strip()

        if not column_name:
            continue

        role = _infer_role_from_dictionary(row)
        risk_level = _assess_risk_level(column_name, role, row)
        business_meaning = str(row.get("business_meaning", row.get("column_cn_name", ""))).strip()
        available_time = str(row.get("available_time", "")).strip()

        parsed_columns.append({
            "table_name": table_name,
            "column_name": column_name,
            "business_meaning": business_meaning,
            "role": role,
            "confidence": 1.0,
            "risk_level": risk_level,
            "available_time": available_time,
            "from_dictionary": True,
        })

        if risk_level == RiskLevel.HIGH:
            warnings.append(
                f"字段 {column_name} (表 {table_name}) 被标记为高风险: "
                f"角色={role}, 可用时间={available_time}"
            )

    return {
        "dictionary_uploaded": True,
        "parsed_columns": parsed_columns,
        "warnings": warnings,
    }


def validate_dictionary_columns(dictionary_path: str) -> dict:
    """Validate that the dictionary file has the required columns.

    Returns:
        dict with is_valid, missing_required, available_optional.
    """
    df = _read_dictionary_file(dictionary_path)
    columns = [c.strip().lower() for c in df.columns]

    missing_required = [c for c in DICTIONARY_REQUIRED_COLUMNS if c not in columns]
    available_optional = [c for c in DICTIONARY_OPTIONAL_COLUMNS if c in columns]

    return {
        "is_valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "available_optional": available_optional,
        "all_columns": list(df.columns),
    }


def map_dictionary_to_dataset(
    parsed_columns: list[dict],
    dataset_columns: list[str],
) -> dict:
    """Map parsed dictionary entries to actual dataset columns.

    Returns:
        dict with matched, unmatched_in_dictionary, unmatched_in_dataset.
    """
    dict_col_names = {entry["column_name"] for entry in parsed_columns}
    dataset_col_set = set(dataset_columns)

    matched = [
        entry for entry in parsed_columns
        if entry["column_name"] in dataset_col_set
    ]
    unmatched_in_dictionary = [
        entry["column_name"] for entry in parsed_columns
        if entry["column_name"] not in dataset_col_set
    ]
    unmatched_in_dataset = [
        col for col in dataset_columns
        if col not in dict_col_names
    ]

    return {
        "matched": matched,
        "unmatched_in_dictionary": unmatched_in_dictionary,
        "unmatched_in_dataset": unmatched_in_dataset,
        "coverage_rate": len(matched) / len(dataset_columns) if dataset_columns else 0.0,
    }


def _read_dictionary_file(path: str) -> pd.DataFrame:
    """Read a dictionary file (CSV or Excel)."""
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(path)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported dictionary format: {ext}")


def _infer_role_from_dictionary(row: pd.Series) -> str:
    """Infer field role from dictionary metadata."""
    is_label = str(row.get("is_label", "")).strip().lower()
    is_id = str(row.get("is_id", "")).strip().lower()
    is_time = str(row.get("is_time", "")).strip().lower()
    is_sensitive = str(row.get("is_sensitive", "")).strip().lower()
    column_name = str(row.get("column_name", "")).lower()
    available_time = str(row.get("available_time", "")).strip().lower()

    if is_label in ("yes", "true", "1", "是"):
        return FieldRole.LABEL

    if is_id in ("yes", "true", "1", "是"):
        if "customer" in column_name:
            return FieldRole.CUSTOMER_KEY
        elif "account" in column_name:
            return FieldRole.ACCOUNT_KEY
        elif "loan" in column_name:
            return FieldRole.LOAN_KEY
        return FieldRole.CUSTOMER_KEY

    if is_time in ("yes", "true", "1", "是"):
        if "transaction" in column_name or "txn" in column_name:
            return FieldRole.TRANSACTION_TIME
        return FieldRole.OBSERVATION_TIME

    if is_sensitive in ("yes", "true", "1", "是"):
        return FieldRole.SENSITIVE

    # Check for post-loan / leakage indicators
    if available_time in ("after_loan", "post_loan", "贷后"):
        if any(kw in column_name for kw in LEAKAGE_KEYWORDS):
            return FieldRole.POSSIBLE_LEAKAGE
        return FieldRole.POST_LOAN_FEATURE

    return FieldRole.NORMAL


def _assess_risk_level(column_name: str, role: str, row: pd.Series) -> str:
    """Assess risk level for a field."""
    if role in (FieldRole.POSSIBLE_LEAKAGE, FieldRole.POST_LOAN_FEATURE):
        return RiskLevel.HIGH

    column_lower = column_name.lower()
    if any(kw in column_lower for kw in LEAKAGE_KEYWORDS):
        return RiskLevel.HIGH

    available_time = str(row.get("available_time", "")).strip().lower()
    if available_time in ("after_loan", "post_loan", "贷后"):
        return RiskLevel.MEDIUM

    return RiskLevel.NORMAL
