"""Parser robustness tests for heterogeneous model outputs."""

from __future__ import annotations

from latentgoalops.baseline.parsers import extract_json_object, parse_action
from latentgoalops.models import TaskId


def test_extract_json_object_handles_markdown_fence():
    payload = extract_json_object(
        """
        Here is the action.
        ```json
        {"task_id":"task2_roadmap_priority","selected_item_ids":["item_1"],"rationale_summary":"Focus."}
        ```
        """
    )
    assert payload["task_id"] == "task2_roadmap_priority"


def test_extract_json_object_handles_prose_around_object():
    payload = extract_json_object(
        'I choose this action: {"task_id":"task4_capital_allocation","budget_allocations":{"renewal_rescue_pod":3},"rationale_summary":"Renewals first."} Thanks.'
    )
    assert payload["task_id"] == "task4_capital_allocation"


def test_parse_action_accepts_fenced_output():
    action = parse_action(
        """
        ```json
        {"labels":{},"priorities":{},"escalate_ids":[]}
        ```
        """,
        TaskId.TASK1,
    )
    assert action.task_id.value == "task1_feedback_triage"


def test_parse_action_accepts_compact_task2_shape():
    action = parse_action('{"selected":["item_1","item_2"]}', TaskId.TASK2)
    assert action.task_id == TaskId.TASK2
    assert action.selected_item_ids == ["item_1", "item_2"]


def test_parse_action_accepts_compact_task7_aliases():
    action = parse_action(
        '{"allocations":{"hire_sre_cluster":2},"focus":{"entity_ids":["team_infra"],"tags":["team_update"]},"belief":{"archetype_probs":{"efficiency":1.0},"shift_detected_confidence":0.3}}',
        TaskId.TASK7,
    )
    assert action.task_id == TaskId.TASK7
    assert action.budget_allocations == {"hire_sre_cluster": 2}
    assert len(action.memory_focus) == 1
    assert action.belief_report is not None
    assert action.belief_report.archetype_probs["efficiency"] == 1.0


def test_parse_action_repairs_truncated_compact_task6_json():
    action = parse_action(
        '{"chosen":["refactor_incident_tooling"],"msg":"retention_campaign","price":0.0,"support":"incident_swarm"',
        TaskId.TASK6,
    )
    assert action.task_id == TaskId.TASK6
    assert action.chosen_initiatives == ["refactor_incident_tooling"]
    assert action.messaging_action.value == "retention_campaign"
    assert action.support_policy.value == "incident_swarm"
