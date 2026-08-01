import json
import os

FEEDBACK_FILE = "feedback.json"


def save_feedback(query, answer, rating):
    data = {
        "query": query,
        "answer": answer,
        "rating": rating
    }

    # Load existing
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            feedback_list = json.load(f)
    else:
        feedback_list = []

    feedback_list.append(data)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_list, f, indent=4)


def get_bad_feedback(threshold=3):
    if not os.path.exists(FEEDBACK_FILE):
        return []

    with open(FEEDBACK_FILE, "r") as f:
        feedback_list = json.load(f)

    bad = [f for f in feedback_list if f["rating"] < threshold]
    return bad