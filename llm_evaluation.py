import mlflow
import openai
import os
import pandas as pd
from dotenv import load_dotenv
import dagshub
from openai import OpenAI
from openrouter_wrapper import OpenRouterWrapper

# Load .env file
load_dotenv()

# Initialize DagsHub
dagshub.init(repo_owner='jaggusuperhit', repo_name='LLM-EVAL_MLFLOW', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/jaggusuperhit/LLM-EVAL_MLFLOW.mlflow")

# Define evaluation data
eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)

# Set experiment
mlflow.set_experiment("LLM Evaluation")

# Start run
with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    
    # Initialize OpenAI client with OpenRouter configuration
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Create and log the custom model
    model = OpenRouterWrapper(system_prompt)
    model_path = "openrouter_model"
    mlflow.pyfunc.log_model(
        artifact_path=model_path,
        python_model=model,
        code_path=["openrouter_wrapper.py"],
        registered_model_name="OpenRouter-GPT3.5"
    )
    
    # Get the model URI
    model_uri = f"runs:/{run.info.run_id}/{model_path}"
    
    # Evaluate model
    results = mlflow.evaluate(
        model_uri,
        data=eval_data,
        targets="ground_truth",
        model_type="text",
        evaluators="default",
        # extra_metrics can be added here if you have custom metrics or plugins
        evaluator_config={
            "col_mapping": {
                "inputs": "inputs",
                "ground_truth": "ground_truth"
            }
        }
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # Save evaluation table
    eval_table = results.tables["eval_results_table"]
    df = pd.DataFrame(eval_table)
    df.to_csv('eval.csv')
    print(f"See evaluation table below: \n{eval_table}")