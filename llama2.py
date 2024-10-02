import boto3
import json

prompt_data="""
Act as a Shakespeare and write a poem on Machine Learning
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    "prompt": prompt_data,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body=json.dumps(payload)
model_id="meta.llama3-70b-instruct-v1:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    contentType="application/json",
    accept="application/json"
)

response_body=json.loads(response.get("body").read())
response_text=response_body['generation']
print(response_text)