"""Robust parsing for compact benchmark-facing model actions."""

from __future__ import annotations

import json
import re

from latentgoalops.models import LatentGoalOpsAction, TaskId


CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


def _candidate_text_spans(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    spans: list[str] = []
    for fenced in CODE_FENCE_RE.findall(cleaned):
        fenced_clean = fenced.strip()
        if fenced_clean:
            spans.append(fenced_clean)
    spans.append(cleaned)
    return spans


def extract_json_object(text: str) -> dict:
    """Extract the most plausible top-level JSON object from text."""
    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for span in _candidate_text_spans(text):
        for index, char in enumerate(span):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(span[index:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                return parsed
            last_error = ValueError("Parsed JSON payload was not an object.")
    if last_error is not None:
        raise ValueError("Model response did not contain a valid JSON object.") from last_error
    raise ValueError("Model response did not contain a JSON object.")


def _repair_truncated_json(text: str) -> str | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    start = cleaned.find("{")
    if start < 0:
        return None
    candidate = cleaned[start:]
    repaired_chars: list[str] = []
    expected_closers: list[str] = []
    in_string = False
    escape = False
    for char in candidate:
        repaired_chars.append(char)
        if escape:
            escape = False
            continue
        if in_string:
            if char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            expected_closers.append("}")
            continue
        if char == "[":
            expected_closers.append("]")
            continue
        if char in "}]":
            if expected_closers and char == expected_closers[-1]:
                expected_closers.pop()
            else:
                return None
    repaired = "".join(repaired_chars).rstrip()
    if in_string:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired)
    if expected_closers:
        repaired += "".join(reversed(expected_closers))
    if repaired == candidate:
        return None
    return repaired


def extract_json_object_with_repair(text: str) -> dict:
    """Extract JSON, then try deterministic bracket-balancing repair for clipped payloads."""
    try:
        return extract_json_object(text)
    except ValueError as original_error:
        for span in _candidate_text_spans(text):
            repaired = _repair_truncated_json(span)
            if repaired is None:
                continue
            try:
                return extract_json_object(repaired)
            except ValueError:
                continue
        raise original_error


def _listify(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_present(payload: dict, *keys: str):
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _compact_labels(value) -> list[dict]:
    if isinstance(value, dict):
        return [{"item_id": str(item_id), "label": label} for item_id, label in value.items()]
    return _listify(value)


def _compact_priorities(value) -> list[dict]:
    if isinstance(value, dict):
        return [{"item_id": str(item_id), "priority": priority} for item_id, priority in value.items()]
    return _listify(value)


def _compact_allocations(value) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): amount for key, amount in value.items()}
    if isinstance(value, list):
        allocations: dict[str, float] = {}
        for entry in value:
            if isinstance(entry, dict):
                item_id = _first_present(entry, "item_id", "id", "program_id")
                amount = _first_present(entry, "amount", "allocation", "value")
                if item_id is not None and amount is not None:
                    allocations[str(item_id)] = amount
        return allocations
    return {}


def _compact_focus(value) -> list[dict]:
    return _listify(value)


def _compact_writes(value) -> list[dict]:
    return _listify(value)


def _coerce_payload(task_id: TaskId, payload: dict) -> dict:
    coerced: dict = {"task_id": task_id.value}

    if task_id == TaskId.TASK1:
        coerced["labels"] = _compact_labels(_first_present(payload, "labels", "label_map"))
        coerced["priorities"] = _compact_priorities(_first_present(payload, "priorities", "priority_map"))
        coerced["escalate_ids"] = [
            str(item_id)
            for item_id in _listify(_first_present(payload, "escalate_ids", "escalate", "escalations"))
            if item_id is not None
        ]
        return coerced

    if task_id == TaskId.TASK2:
        coerced["selected_item_ids"] = [
            str(item_id)
            for item_id in _listify(_first_present(payload, "selected_item_ids", "selected", "selected_ids", "items"))
            if item_id is not None
        ]
        rationale_summary = _first_present(payload, "rationale_summary", "summary", "note")
        if isinstance(rationale_summary, str) and rationale_summary.strip():
            coerced["rationale_summary"] = rationale_summary
        return coerced

    if task_id in {TaskId.TASK3, TaskId.TASK5, TaskId.TASK6}:
        coerced["chosen_initiatives"] = [
            str(item_id)
            for item_id in _listify(_first_present(payload, "chosen_initiatives", "chosen", "chosen_ids", "items"))
            if item_id is not None
        ]
        messaging_action = _first_present(payload, "messaging_action", "msg", "messaging")
        if messaging_action is not None:
            coerced["messaging_action"] = messaging_action
        pricing_change_pct = _first_present(payload, "pricing_change_pct", "price", "pricing")
        if pricing_change_pct is not None:
            coerced["pricing_change_pct"] = pricing_change_pct
        support_policy = _first_present(payload, "support_policy", "support")
        if support_policy is not None:
            coerced["support_policy"] = support_policy
        memory_focus = _first_present(payload, "memory_focus", "focus")
        if memory_focus is not None:
            coerced["memory_focus"] = _compact_focus(memory_focus)
        memory_writes = _first_present(payload, "memory_writes", "writes", "notes")
        if memory_writes is not None:
            coerced["memory_writes"] = _compact_writes(memory_writes)
        belief_report = _first_present(payload, "belief_report", "belief")
        if belief_report is not None:
            coerced["belief_report"] = belief_report
        rationale = _first_present(payload, "rationale", "note", "summary")
        if isinstance(rationale, str) and rationale.strip():
            coerced["rationale"] = rationale
        return coerced

    if task_id in {TaskId.TASK4, TaskId.TASK7}:
        coerced["budget_allocations"] = _compact_allocations(
            _first_present(payload, "budget_allocations", "allocations", "allocs")
        )
        rationale_summary = _first_present(payload, "rationale_summary", "summary", "note")
        if isinstance(rationale_summary, str) and rationale_summary.strip():
            coerced["rationale_summary"] = rationale_summary
        if task_id == TaskId.TASK7:
            memory_focus = _first_present(payload, "memory_focus", "focus")
            if memory_focus is not None:
                coerced["memory_focus"] = _compact_focus(memory_focus)
            memory_writes = _first_present(payload, "memory_writes", "writes", "notes")
            if memory_writes is not None:
                coerced["memory_writes"] = _compact_writes(memory_writes)
            belief_report = _first_present(payload, "belief_report", "belief")
            if belief_report is not None:
                coerced["belief_report"] = belief_report
        return coerced

    return {"task_id": task_id.value, **payload}


def parse_action(text: str, task_id: TaskId) -> LatentGoalOpsAction:
    """Parse a model response into a validated environment action."""
    return LatentGoalOpsAction.model_validate(_coerce_payload(task_id, extract_json_object_with_repair(text)))
