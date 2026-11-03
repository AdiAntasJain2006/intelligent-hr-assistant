"""Command line demo of HR policy agent can be run using python main.py"""

from hr_assistant.logger import get_logger
from hr_assistant.pipeline import ask_agent, build_hr_assistant

logger = get_logger(__name__)


def main() :
    logger.info("Starting HR assistant CLI demo")
    agent = build_hr_assistant()
    print("Assistant ready")
    
    sample_questions = [
        "What are the key steps in the employee hiring process?",
        "What documents are generally required when joining a new company?",
        "How can an employee apply for leave?",
        "What should I do if my salary credited is incorrect?",
        "How should I report workplace harassment or misconduct?"
    ]

    print("Ask an HR question. Enter 'quit' or 'exit' to stop.")
    print("Sample questions:")
    for question in sample_questions:
        print(f"- {question}")

    chat_history: list[dict[str, str]] = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            print()
            break

        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        answer = ask_agent(
            agent,
            question,
            chat_history=chat_history,
        )
        print(f"Assistant: {answer}")
        chat_history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

    logger.info("HR assistant CLI demo completed")

if __name__ == "__main__" :
    main()