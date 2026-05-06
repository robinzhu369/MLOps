"""Constants and enumerations for the risk modeling agent."""

from enum import Enum


class DataType(str, Enum):
    STRUCTURED_MODELING_TABLE = "structured_modeling_table"
    TRANSACTION_FLOW_TABLE = "transaction_flow_table"
    MAIN_TABLE = "main_table"
    AUXILIARY_TABLE = "auxiliary_table"
    UNKNOWN = "unknown_or_ambiguous"


class PipelineType(str, Enum):
    STRUCTURED_MODELING = "structured_modeling_pipeline"
    TRANSACTION_FEATURE = "transaction_feature_pipeline"
    MAIN_PLUS_TRANSACTION = "main_plus_transaction_pipeline"
    MANUAL_CONFIGURATION = "manual_configuration_pipeline"


class FieldRole(str, Enum):
    LABEL = "label"
    CUSTOMER_KEY = "customer_key"
    ACCOUNT_KEY = "account_key"
    LOAN_KEY = "loan_key"
    OBSERVATION_TIME = "observation_time"
    TRANSACTION_TIME = "transaction_time"
    AMOUNT = "amount"
    DIRECTION = "direction"
    SENSITIVE = "sensitive"
    POSSIBLE_LEAKAGE = "possible_leakage_feature"
    POST_LOAN_FEATURE = "post_loan_feature"
    NORMAL = "normal"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    NORMAL = "normal"


# Keywords that indicate potential label leakage
LEAKAGE_KEYWORDS = [
    "bad", "overdue", "default", "dpd",
    "逾期", "违约", "催收", "还款结果", "结清", "核销", "风险结果",
    "collection", "writeoff", "settlement", "repay_result",
]

# Supported file formats
SUPPORTED_FILE_FORMATS = ["csv", "xlsx", "xls", "parquet"]

# Transaction feature windows (days)
DEFAULT_FEATURE_WINDOWS = [7, 14, 30, 60, 90, 180]

# Data quality thresholds
DEFAULT_HIGH_MISSING_THRESHOLD = 0.3
DEFAULT_OUTLIER_METHOD = "IQR"
DEFAULT_WINSORIZE_LOWER = 0.01
DEFAULT_WINSORIZE_UPPER = 0.99
