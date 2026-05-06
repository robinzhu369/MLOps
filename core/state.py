"""State definitions for the risk modeling agent pipeline."""

from typing import TypedDict, Optional, List, Dict, Any


class UploadedFileState(TypedDict, total=False):
    file_id: str
    file_name: str
    file_path: str
    file_format: str
    n_rows: int
    n_cols: int
    columns: List[str]
    column_profiles: Dict[str, Any]
    sample_values_masked: Dict[str, Any]
    rule_detected_type: str
    llm_detected_type: str
    final_detected_type: str
    confidence: float
    assigned_role: str
    recommended_pipeline: str
    detected_keys: Dict[str, Any]
    warnings: List[str]


class RiskModelingProjectState(TypedDict, total=False):
    project_name: str

    # File upload
    uploaded_files: List[UploadedFileState]

    # Data dictionary & field semantics
    data_dictionary_path: Optional[str]
    dictionary_uploaded: bool
    dictionary_parse_result: Dict[str, Any]
    field_semantics: Dict[str, Any]

    # Data type classification
    data_type_classification_result: Dict[str, Any]
    pipeline_type: str

    # Data quality
    data_quality_report: Dict[str, Any]
    transaction_quality_report: Optional[Dict[str, Any]]

    # Cleaning
    cleaning_plan: Dict[str, Any]
    cleaning_user_decision: Dict[str, Any]
    cleaned_data_paths: Dict[str, str]
    cleaning_log_path: str
    quality_recheck_report: Dict[str, Any]

    # Human confirmations
    human_confirmations: List[Dict[str, Any]]

    # Main table fields
    main_data_path: Optional[str]
    cleaned_main_data_path: Optional[str]
    label_col: Optional[str]
    id_col: Optional[str]
    account_col: Optional[str]
    customer_col: Optional[str]
    time_col: Optional[str]
    positive_label: int

    # Transaction table fields
    transaction_data_path: Optional[str]
    cleaned_transaction_data_path: Optional[str]
    transaction_schema: Dict[str, Any]
    transaction_feature_plan: Dict[str, Any]
    transaction_daily_feature_path: Optional[str]
    transaction_window_feature_path: Optional[str]

    # Modeling
    modeling_data_path: Optional[str]
    train_path: Optional[str]
    test_path: Optional[str]
    model_path: Optional[str]
    leaderboard_path: Optional[str]
    predictions_path: Optional[str]
    threshold_table_path: Optional[str]
    feature_importance_path: Optional[str]
    metrics: Dict[str, Any]
    report_path: Optional[str]

    # Trace
    agent_trace: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    next_step: str
