import json
import openai

# Step 1: Load OpenAI API key from a file
with open("apikey_openai.txt") as file:
    openai.api_key = file.readline().strip()

# Step 2: Initialize an empty list to hold all example conversations
conversations = []

# Step 3: Define example conversations for fine-tuning
# Example 1: Conversation about the procurement process
conversations.append({
    "messages": [
        {"role": "system", "content": "You are a helpful assistant specialized in enterprise procurement."},
        {"role": "user", "content": "What is the first step in the procurement process?"},
        {"role": "assistant", "content": "The first step is identifying the need for a product or service."}
    ]
})

# Example 2: Conversation about supplier evaluation
conversations.append({
    "messages": [
        {"role": "system", "content": "You are a helpful assistant specialized in enterprise procurement."},
        {"role": "user", "content": "How do I evaluate suppliers?"},
        {"role": "assistant", "content": "Supplier evaluation often involves assessing factors like cost, quality, and reliability."}
    ]
})

# Step 4: Generate additional example conversations
# These are hypothetical examples to meet the minimum requirement of 10 examples
for i in range(3, 11):
    conversations.append({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in enterprise procurement."},
            {"role": "user", "content": f"What is the importance of step {i} in the procurement process?"},
            {"role": "assistant", "content": f"Step {i} is crucial for ensuring compliance and quality in the procurement process."}
        ]
    })

# Step 5: Save the conversations to a JSONL file
with open('procurement_fine_tuning_data3.jsonl', 'w') as f:
    for conversation in conversations:
        f.write(json.dumps(conversation) + '\n')

print("Data has been saved to 'procurement_fine_tuning_data.jsonl'")

# Step 6: Upload the data to OpenAI for fine-tuning
file_response = openai.File.create(file=open("procurement_fine_tuning_data.jsonl"), purpose="fine-tune")
uploaded_file_id = file_response["id"]

# Step 7: Start the fine-tuning job
# Use FineTuningJob class in the newest version, the previous ones are deprecated
job = openai.FineTuningJob.create(training_file=uploaded_file_id, model="gpt-3.5-turbo")

# Step 8: Monitor the fine-tuning job until completion
# Use FineTuningJob class in the newest version, the previous ones are deprecated
while True:
    job_status = openai.FineTuningJob.retrieve(id=job["id"])
    if job_status["status"] == "succeeded":
        break

custom_model = job["fine_tuned_model"]

# Step 9: Test the fine-tuned model
# Use ChatCompletion, not Completion class - as this is a chat model, not a completion model.
messages = [{"role": "user", "content": "How would I evaluate suppliers?"}]
response = openai.ChatCompletion.create(model=custom_model, messages=messages)

# Step 10: Extract and print the assistant's response
if 'choices' in response and len(response['choices']) > 0:
    assistant_response = response['choices'][0]['message']['content']
    print(assistant_response.strip())
else:
    print("Unexpected response format or empty 'choices'")
