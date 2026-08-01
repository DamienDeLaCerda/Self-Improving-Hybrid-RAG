from app.feedback import get_bad_feedback
from app.query_rewriter import rewrite_query


def improve_queries():
    bad_feedback = get_bad_feedback()

    if not bad_feedback:
        print("No bad feedback yet.")
        return

    print("\n🔧 Improving based on bad feedback:\n")

    for item in bad_feedback:
        original_query = item["query"]

        try:
            improved_query = rewrite_query(original_query)
            print(f"Original: {original_query}")
            print(f"Improved: {improved_query}\n")
        except:
            print(f"Could not improve: {original_query}")