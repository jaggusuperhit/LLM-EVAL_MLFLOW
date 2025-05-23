import os
from openai import OpenAI
import mlflow.pyfunc

class OpenRouterWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def predict(self, context, model_input):
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        responses = []
        for question in model_input["inputs"]:
            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ]
            )
            responses.append(response.choices[0].message.content)
        return responses
