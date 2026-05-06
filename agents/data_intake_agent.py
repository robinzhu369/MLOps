"""DataIntakeAgent — coordinates file upload and metadata extraction."""

from typing import IO

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState, UploadedFileState
from tools.file_tools import save_uploaded_file, inspect_file_format
from tools.metadata_tools import extract_file_metadata
from tools.trace_tools import write_agent_trace

AGENT_NAME = "DataIntakeAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    file_obj: IO[bytes],
    file_name: str,
) -> UploadedFileState:
    """Process a single uploaded file: save, inspect, and extract metadata.

    Args:
        state: Current project state (must contain project_name).
        file_obj: File-like object with the uploaded content.
        file_name: Original file name.

    Returns:
        UploadedFileState with all metadata populated.
    """
    project_name = state["project_name"]

    # Step 1: Save uploaded file
    _check_permission("save_uploaded_file")
    save_result = save_uploaded_file(file_obj, file_name, project_name)

    file_path = save_result["file_path"]

    # Step 2: Inspect file format
    _check_permission("inspect_file_format")
    inspect_result = inspect_file_format(file_path)

    if not inspect_result["readable"]:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary=f"文件 {file_name} 无法读取",
            action="inspect_file_format",
            action_input_summary={"file": file_name},
            observation_summary=f"读取失败: {inspect_result['error']}",
            decision="标记文件不可读，等待用户处理",
            next_node="error_handling",
            status="error",
        )
        return UploadedFileState(
            file_id=save_result["file_id"],
            file_name=file_name,
            file_path=file_path,
            file_format=save_result["file_format"],
            warnings=[f"文件无法读取: {inspect_result['error']}"],
        )

    # Step 3: Extract metadata
    _check_permission("extract_file_metadata")
    metadata = extract_file_metadata(file_path)

    # Build the file state
    file_state = UploadedFileState(
        file_id=save_result["file_id"],
        file_name=file_name,
        file_path=file_path,
        file_format=save_result["file_format"],
        n_rows=metadata["n_rows"],
        n_cols=metadata["n_cols"],
        columns=metadata["columns"],
        column_profiles=metadata["column_profiles"],
        sample_values_masked=metadata["sample_values_masked"],
        warnings=[],
    )

    # Write trace
    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary=f"接收并解析上传文件 {file_name}",
        action="extract_file_metadata",
        action_input_summary={"file": file_name, "rows": metadata["n_rows"], "cols": metadata["n_cols"]},
        observation_summary=f"文件包含 {metadata['n_rows']} 行 {metadata['n_cols']} 列",
        decision="元数据提取完成，进入字段语义解析",
        next_node="FieldSemanticParserAgent",
        status="completed",
    )

    return file_state


def run_batch(
    state: RiskModelingProjectState,
    files: list[tuple[IO[bytes], str]],
) -> list[UploadedFileState]:
    """Process multiple uploaded files.

    Args:
        state: Current project state.
        files: List of (file_obj, file_name) tuples.

    Returns:
        List of UploadedFileState for each file.
    """
    results = []
    for file_obj, file_name in files:
        file_state = run(state, file_obj, file_name)
        results.append(file_state)
    return results
