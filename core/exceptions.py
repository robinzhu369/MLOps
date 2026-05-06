"""Custom exceptions for the risk modeling agent."""


class PermissionDeniedError(Exception):
    """Raised when an agent attempts to call a tool not in its whitelist."""

    def __init__(self, agent_name: str, tool_name: str):
        self.agent_name = agent_name
        self.tool_name = tool_name
        super().__init__(
            f"Agent '{agent_name}' is not permitted to call tool '{tool_name}'."
        )


class DataValidationError(Exception):
    """Raised when uploaded data fails validation checks."""


class PipelineRoutingError(Exception):
    """Raised when pipeline routing cannot determine the correct pipeline."""


class TimeLeakageError(Exception):
    """Raised when time leakage is detected in feature engineering."""


class CleaningExecutionError(Exception):
    """Raised when a cleaning action fails to execute."""
