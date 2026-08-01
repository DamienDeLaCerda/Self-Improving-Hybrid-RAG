from app.retriever import build_retriever, hybrid_search
from app.query_rewriter import rewrite_query
from app.generator import generate_answer
from app.evaluator import evaluate_rag
from app.feedback import save_feedback
from app.improver import improve_queries

if __name__ == "__main__":
    collection, bm25, chunks = build_retriever("data/arXiv_scientific dataset.csv")

    while True:
        user_query = input("\nEnter your query: ")

        # Rewrite
        try:
            rewritten_query = rewrite_query(user_query)
            print(f"\nRewritten Query: {rewritten_query}")
        except:
            rewritten_query = user_query

        # Retrieve
        results = hybrid_search(rewritten_query, collection, bm25, chunks)

        # Generate
        answer = generate_answer(user_query, results)

        print("\n🧠 Final Answer:\n")
        print(answer)

        # Evaluate
        try:
            eval_result = evaluate_rag(user_query, answer, results)
            print("\n📊 Evaluation Metrics:\n")
            print(eval_result)
        except:
            pass

        # 🔥 Feedback
        try:
            rating = int(input("\nRate the answer (1-5): "))
            save_feedback(user_query, answer, rating)
        except:
            print("Invalid rating")

        # 🔥 Improve system
        improve_queries()