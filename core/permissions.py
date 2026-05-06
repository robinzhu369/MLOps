"""Agent-tool permission whitelist."""

AGENT_TOOL_PERMISSIONS: dict[str, list[str]] = {
    "DataIntakeAgent": [
        "save_uploaded_file",
        "inspect_file_format",
        "extract_file_metadata",
    ],
    "DataDictionaryParserAgent": [
        "parse_data_dictionary",
        "validate_dictionary_columns",
        "map_dictionary_to_dataset",
    ],
    "FieldSemanticParserAgent": [
        "parse_field_semantics_by_llm",
        "merge_dictionary_and_llm_semantics",
    ],
    "DataTypeClassifierAgent": [
        "classify_data_type_by_rules",
        "classify_data_type_by_llm",
        "merge_data_type_classification",
    ],
    "DataQualityAgent": [
        "analyze_duplicates",
        "analyze_missing_values",
        "analyze_outliers",
        "analyze_type_mismatch",
        "analyze_label_quality",
        "analyze_key_quality",
        "generate_data_quality_report",
    ],
    "TransactionQualityAgent": [
        "analyze_transaction_quality",
    ],
    "DataCleaningPlannerAgent": [
        "generate_cleaning_plan",
    ],
    "DataCleaningExecutorTool": [
        "execute_cleaning_plan",
    ],
    "TransactionFeatureAgent": [
        "infer_transaction_schema",
        "build_account_daily_features",
        "build_account_window_features",
        "validate_transaction_feature_cutoff",
        "profile_generated_features",
    ],
    "RiskGuardAgent": [
        "detect_leakage_columns",
        "detect_id_columns",
        "detect_high_missing_columns",
        "detect_time_leakage_candidates",
    ],
    "ModelingAgent": [
        "train_autogluon_binary",
    ],
    "EvaluationAgent": [
        "evaluate_binary_model",
    ],
    "StrategyAgent": [
        "build_threshold_table",
    ],
    "ExplainAgent": [
        "compute_feature_importance",
    ],
    "ReportAgent": [
        "generate_markdown_report",
    ],
    "PipelineRouterAgent": [
        "route_pipeline",
    ],
}


def check_tool_permission(agent_name: str, tool_name: str) -> bool:
    """Check if an agent is allowed to call a specific tool."""
    return tool_name in AGENT_TOOL_PERMISSIONS.get(agent_name, [])
