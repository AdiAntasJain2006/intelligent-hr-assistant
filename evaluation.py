from hr_assistant.run_eval import run_evaluation
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def main() :
    logger.info("Starting HR assistant evaluation")
    response = run_evaluation()
    logger.info("Evaluation completed")
    print("Evaluation results:")
    print(response)
    
if(__name__ == "__main__") :
    main()