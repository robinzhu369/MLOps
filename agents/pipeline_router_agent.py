"""PipelineRouterAgent — routes to the appropriate pipeline based on data type classification."""

from core.constants import DataType, PipelineType
from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError, PipelineRoutingError
from core.state import RiskModelingProjectState
from tools.trace_tools import write_agent_trace

AGENT_NAME = "PipelineRouterAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def route_pipeline(classifications: list[dict]) -> dict:
    """Determine which pipeline to use based on file classifications.

    Routing logic:
    - Single structured_modeling_table → structured_modeling_pipeline
    - Single transaction_flow_table → transaction_feature_pipeline
    - main_table + transaction_flow_table → main_plus_transaction_pipeline
    - structured_modeling_table + transaction_flow_table → main_plus_transaction_pipeline
    - unknown/ambiguous → manual_configuration_pipeline

    Returns:
        dict with pipeline_type, reasoning, file_roles.
    """
    if not classifications:
        return {
            "pipeline_type": PipelineType.MANUAL_CONFIGURATION,
            "reasoning": "没有文件分类结果",
            "file_roles": {},
        }

    types = [c.get("detected_data_type", DataType.UNKNOWN) for c in classifications]
    file_names = [c.get("file_name", "") for c in classifications]

    # Build file roles mapping
    file_roles = {}
    for c in classifications:
        file_roles[c.get("file_name", "")] = {
            "type": c.get("detected_data_type"),
            "role": c.get("detected_data_type"),
            "detected_roles": c.get("detected_roles", {}),
        }

    # Single file scenarios
    if len(classifications) == 1:
        data_type = types[0]
        if data_type == DataType.STRUCTURED_MODELING_TABLE:
            return {
                "pipeline_type": PipelineType.STRUCTURED_MODELING,
                "reasoning": f"单文件 {file_names[0]} 识别为结构化建模表，进入结构化建模 Pipeline",
                "file_roles": file_roles,
            }
        elif data_type == DataType.TRANSACTION_FLOW_TABLE:
            return {
                "pipeline_type": PipelineType.TRANSACTION_FEATURE,
                "reasoning": f"单文件 {file_names[0]} 识别为交易流水表，进入交易流水特征 Pipeline",
                "file_roles": file_roles,
            }
        elif data_type == DataType.MAIN_TABLE:
            return {
                "pipeline_type": PipelineType.STRUCTURED_MODELING,
                "reasoning": f"单文件 {file_names[0]} 识别为主表但无流水表，按结构化建模处理",
                "file_roles": file_roles,
            }
        else:
            return {
                "pipeline_type": PipelineType.MANUAL_CONFIGURATION,
                "reasoning": f"文件 {file_names[0]} 类型不确定，需要人工配置",
                "file_roles": file_roles,
            }

    # Multi-file scenarios
    has_main = any(t in (DataType.MAIN_TABLE, DataType.STRUCTURED_MODELING_TABLE) for t in types)
    has_transaction = DataType.TRANSACTION_FLOW_TABLE in types

    if has_main and has_transaction:
        return {
            "pipeline_type": PipelineType.MAIN_PLUS_TRANSACTION,
            "reasoning": "检测到主表和交易流水表组合，进入联合建模 Pipeline",
            "file_roles": file_roles,
        }
    elif has_main and not has_transaction:
        return {
            "pipeline_type": PipelineType.STRUCTURED_MODELING,
            "reasoning": "检测到主表但无交易流水表，进入结构化建模 Pipeline",
            "file_roles": file_roles,
        }
    elif has_transaction and not has_main:
        return {
            "pipeline_type": PipelineType.TRANSACTION_FEATURE,
            "reasoning": "仅检测到交易流水表，进入交易流水特征 Pipeline",
            "file_roles": file_roles,
        }
    else:
        return {
            "pipeline_type": PipelineType.MANUAL_CONFIGURATION,
            "reasoning": "无法确定文件组合的 Pipeline，需要人工配置",
            "file_roles": file_roles,
        }


def run(state: RiskModelingProjectState) -> dict:
    """Route to the appropriate pipeline based on classification results.

    Expects state to contain data_type_classification_result with classifications.

    Returns:
        dict with pipeline_type, reasoning, file_roles.
    """
    project_name = state["project_name"]
    classification_result = state.get("data_type_classification_result", {})
    classifications = classification_result.get("classifications", [])

    _check_permission("route_pipeline")
    result = route_pipeline(classifications)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="根据数据类型分类结果确定 Pipeline 路由",
        action="route_pipeline",
        action_input_summary={
            "file_types": {c.get("file_name"): c.get("detected_data_type") for c in classifications},
        },
        observation_summary=f"路由结果: {result['pipeline_type']}",
        decision=result["reasoning"],
        next_node=result["pipeline_type"],
        status="completed",
    )

    return result
