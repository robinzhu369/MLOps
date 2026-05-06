"""Pydantic schemas for structured data exchange between agents and tools."""

from typing import Optional

from pydantic import BaseModel


class ColumnSemantic(BaseModel):
    table_name: str
    column_name: str
    business_meaning: str = ""
    role: str = "normal"
    confidence: float = 1.0
    risk_level: str = "normal"
    available_time: str = ""
    from_dictionary: bool = False


class FieldSemanticResult(BaseModel):
    dictionary_uploaded: bool
    parsed_columns: list[ColumnSemantic] = []
    field_semantics: dict[str, dict] = {}
    need_human_review: bool = False
    warnings: list[str] = []


class DataTypeClassificationResult(BaseModel):
    file_name: str
    detected_data_type: str
    confidence: float
    reasoning_summary: str = ""
    recommended_pipeline: str = ""
    detected_roles: dict[str, Optional[str]] = {}
    warnings: list[str] = []
    need_human_review: bool = False


class DuplicateAnalysis(BaseModel):
    duplicate_rows: int = 0
    duplicate_rate: float = 0.0
    duplicate_key_rows: int = 0
    suggestion: str = ""


class MissingColumnInfo(BaseModel):
    column: str
    missing_rate: float
    suggestion: str = ""


class OutlierColumnInfo(BaseModel):
    column: str
    method: str = "IQR"
    outlier_rate: float = 0.0
    suggestion: str = ""


class TypeMismatchInfo(BaseModel):
    column: str
    current_type: str
    suggested_type: str


class DataQualityReport(BaseModel):
    duplicate_analysis: DuplicateAnalysis = DuplicateAnalysis()
    missing_analysis: dict = {}
    outlier_analysis: dict = {}
    type_analysis: dict = {}
    label_analysis: dict = {}
    key_analysis: dict = {}
    overall_quality_score: int = 0
    need_cleaning_review: bool = True


class CleaningPlan(BaseModel):
    duplicate_strategy: dict = {}
    missing_strategy: dict = {}
    outlier_strategy: dict = {}
    type_fix_strategy: dict = {}
    protected_columns: list[str] = []
    warnings: list[str] = []


class CleaningResult(BaseModel):
    cleaned_data_path: str
    cleaning_log_path: str
    before_shape: list[int]
    after_shape: list[int]
    actions_applied: list[str] = []


class ModelMetrics(BaseModel):
    auc: float = 0.0
    ks: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class ThresholdRow(BaseModel):
    threshold: float
    pass_rate: float
    reject_rate: float
    pass_bad_rate: float
    bad_capture_rate: float


class AgentTraceEntry(BaseModel):
    agent_name: str
    reasoning_summary: str = ""
    action: str = ""
    action_input_summary: dict = {}
    observation_summary: str = ""
    decision: str = ""
    next_node: str = ""
    status: str = ""
    timestamp: str = ""


class HumanConfirmation(BaseModel):
    confirmation_type: str
    timestamp: str
    user_decision: str
    details: dict = {}
