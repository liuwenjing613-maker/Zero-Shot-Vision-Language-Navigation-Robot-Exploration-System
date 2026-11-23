"""Explicit task-stage state machine; action selection stays in the planner."""

from __future__ import annotations

from enum import Enum

from v4.common.schemas import ActionType, ExecutionStatus


class TaskState(str, Enum):
    IDLE = "IDLE"
    PARSE_TASK = "PARSE_TASK"
    INITIALIZE = "INITIALIZE"
    OBSERVE = "OBSERVE"
    UPDATE_MEMORY = "UPDATE_MEMORY"
    DECIDE = "DECIDE"
    APPROACH = "APPROACH"
    EXPLORE = "EXPLORE"
    VERIFY = "VERIFY"
    RECOVERY = "RECOVERY"
    REPLAN = "REPLAN"
    SUCCESS = "SUCCESS"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"


class TaskStateMachine:
    """Tracks legal orchestration transitions without owning module side effects."""

    _ACTION_STATES = {
        ActionType.SCAN: TaskState.OBSERVE,
        ActionType.OBSERVE_TARGET: TaskState.OBSERVE,
        ActionType.REVISIT_TARGET: TaskState.OBSERVE,
        ActionType.APPROACH_TARGET: TaskState.APPROACH,
        ActionType.EXPLORE_FRONTIER: TaskState.EXPLORE,
        ActionType.VERIFY: TaskState.VERIFY,
        ActionType.RELOCALIZE: TaskState.RECOVERY,
        ActionType.RECOVER: TaskState.RECOVERY,
        ActionType.STOP: TaskState.SEARCH_EXHAUSTED,
    }

    def __init__(self) -> None:
        self.state = TaskState.IDLE

    def start_task(self) -> TaskState:
        self._transition(TaskState.PARSE_TASK)
        return self.state

    def task_parsed(self) -> TaskState:
        self._require(TaskState.PARSE_TASK)
        self._transition(TaskState.INITIALIZE)
        return self.state

    def initialized(self) -> TaskState:
        self._require(TaskState.INITIALIZE)
        self._transition(TaskState.OBSERVE)
        return self.state

    def observed(self) -> TaskState:
        self._require(TaskState.OBSERVE)
        self._transition(TaskState.UPDATE_MEMORY)
        return self.state

    def memory_updated(self) -> TaskState:
        self._require(TaskState.UPDATE_MEMORY)
        self._transition(TaskState.DECIDE)
        return self.state

    def apply_decision(self, action: ActionType) -> TaskState:
        self._require(TaskState.DECIDE, TaskState.REPLAN, TaskState.RECOVERY)
        self._transition(self._ACTION_STATES[action])
        return self.state

    def execution_finished(self, status: ExecutionStatus) -> TaskState:
        if self.state not in {TaskState.APPROACH, TaskState.EXPLORE, TaskState.RECOVERY}:
            raise RuntimeError(f"execution result is invalid from {self.state}")
        if status is ExecutionStatus.SUCCEEDED:
            self._transition(TaskState.OBSERVE)
        elif status in {ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT, ExecutionStatus.CANCELED}:
            self._transition(TaskState.RECOVERY)
        return self.state

    def verified(self, success: bool) -> TaskState:
        self._require(TaskState.VERIFY)
        self._transition(TaskState.SUCCESS if success else TaskState.REPLAN)
        return self.state

    def replan(self) -> TaskState:
        self._transition(TaskState.REPLAN)
        return self.state

    def _require(self, *states: TaskState) -> None:
        if self.state not in states:
            allowed = ", ".join(state.value for state in states)
            raise RuntimeError(f"state {self.state.value} is not one of: {allowed}")

    def _transition(self, state: TaskState) -> None:
        self.state = state
