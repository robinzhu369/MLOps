"""Pipeline state definition for LangGraph orchestration.

Extends RiskModelingProjectState with pipeline-internal fields
(prefixed with underscore) used for passing runtime context.
"""

from typing import TypedDict, Optional, List, Dict, Any, IO, Callable


class PipelineState(TypedDict, total=False):
    """Full pipeline state — all fields optional for partial updates."""

    # Project identity
    project_name: str

    # Runtime context (not persisted, prefixed with _)
    _pending_files: List[str]  # List of file paths to process
    _llm_call: Optional[Callable]
    _time_limit: int

    # File upload
    uploaded_files: List[Dict[str, Any]]

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
    cleaned_data_path: Optional[str]
    cleaned_data_paths: Dict[str, str]
    cleaned_transaction_data_path: Optional[str]
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
    base_time_col: Optional[str]
    positive_label: int

    # Transaction table fields
    transaction_data_path: Optional[str]
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
    predictions: Any
    predictions_path: Optional[str]
    feature_columns: List[str]
    drop_columns: List[str]
    threshold_table_path: Optional[str]
    threshold_table: List[Dict[str, Any]]
    feature_importance_path: Optional[str]
    feature_importance: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    report_path: Optional[str]
    recommendations: List[str]

    # Trace & errors
    agent_trace: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    next_step: str
