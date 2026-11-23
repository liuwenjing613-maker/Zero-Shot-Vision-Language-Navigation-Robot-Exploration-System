"""V4 共享的字符串常量。

按方案 Q4 的决策：**用字符串常量，不用 Enum**。
好处是日志和 JSON 里直接可读，不需要 `.value` 转换；
代价是拼写错误只能在运行期发现，所以统一从这里引用，不要手写字符串。
"""

from __future__ import annotations


class ExecutionStatus:
    """ExecutionResult.status 的取值（方案 6.2 节）。"""

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"

    ALL = (RUNNING, SUCCEEDED, FAILED, CANCELED, TIMEOUT)


class ActionType:
    """ActionDecision.action_type 的取值。

    第一版只实现前三个；OBSERVE_TARGET 留给后续，
    其余类型在 TaskExecutor 里返回 UNSUPPORTED_ACTION。
    """

    EXPLORE_FRONTIER = "EXPLORE_FRONTIER"
    APPROACH_TARGET = "APPROACH_TARGET"
    STOP = "STOP"

    # 预留（第一版 TaskExecutor 会返回 UNSUPPORTED_ACTION）
    OBSERVE_TARGET = "OBSERVE_TARGET"

    ALL = (EXPLORE_FRONTIER, APPROACH_TARGET, STOP, OBSERVE_TARGET)


class ErrorCode:
    """ExecutionResult.error_code 的取值。"""

    NO_PATH = "NO_PATH"
    TIMEOUT = "TIMEOUT"
    STUCK = "STUCK"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    UNKNOWN_ID = "UNKNOWN_ID"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    ALL = (NO_PATH, TIMEOUT, STUCK, UNSUPPORTED_ACTION, UNKNOWN_ID, INTERNAL_ERROR)


class EntityKind:
    """pose_resolver 的 kind 参数，区分 target / frontier 两种 ID 命名空间。"""

    TARGET = "target"
    FRONTIER = "frontier"
