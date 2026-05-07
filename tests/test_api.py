"""FastAPI endpoint tests."""

from fastapi.testclient import TestClient

from latentgoalops.models import TaskId
from latentgoalops.server.app import app


def test_tasks_endpoint():
    client = TestClient(app)
    response = client.get("/tasks")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == len(TaskId)
    assert {item["task_id"] for item in payload} == {task.value for task in TaskId}


def test_reset_and_step_endpoints():
    client = TestClient(app)
    reset_response = client.post("/reset", json={"seed": 9, "task_id": "task1_feedback_triage"})
    assert reset_response.status_code == 200
    observation = reset_response.json()["observation"]
    assert observation["task_id"] == "task1_feedback_triage"

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "reset", "data": {"seed": 9, "task_id": "task1_feedback_triage"}})
        ws_reset = websocket.receive_json()
        assert ws_reset["type"] == "observation"
        ws_observation = ws_reset["data"]["observation"]
        labels = [{"item_id": item["item_id"], "label": "praise"} for item in ws_observation["inbox"]]
        priorities = [{"item_id": item["item_id"], "priority": 3} for item in ws_observation["inbox"]]
        websocket.send_json(
            {
                "type": "step",
                "data": {
                    "task_id": "task1_feedback_triage",
                    "labels": labels,
                    "priorities": priorities,
                    "escalate_ids": [],
                },
            }
        )
        ws_step = websocket.receive_json()
        assert ws_step["type"] == "observation"
        assert "done" in ws_step["data"]
