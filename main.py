"""Command line demo of HR policy agent can be run using python main.py"""

import hr_assistant

from hr_assistant.logger import get_logger
from hr_assistant.pipeline import ask_agent, build_hr_assistant, build_vec_store_for_docs

logger = get_logger(__name__)


def main() :
    logger.info("Starting HR assistant CLI demo")
    agent = build_hr_assistant()
    print("Assistant ready")
    
    demo_questions = [
        "What are the key steps in the employee hiring process?",
        "What documents are generally required when joining a new company?",
        "How can an employee apply for leave?",
        "What should I do if my salary credited is incorrect?",
        "How should I report workplace harassment or misconduct?"
    ]
    
    for question in demo_questions :
        print(ask_agent(agent, question))
    logger.info("HR assistant CLI demo completed")

if __name__ == "__main__" :
    main()