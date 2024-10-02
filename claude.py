# Use the Converse API to send a text message to Claude 3 Sonnet.

import boto3
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="eu-west-2")

# Set the model ID, e.g., Titan Text Premier.
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Start a conversation with the user message.
user_message = """Act as a Shakespeare and write a poem on Machine Learning
"""
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=conversation,
        inferenceConfig={"maxTokens":2000,"temperature":0},
        additionalModelRequestFields={"top_k":250}
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
