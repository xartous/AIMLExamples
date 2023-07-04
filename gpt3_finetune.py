import json
import subprocess
import time

import openai

# Load OpenAI API key
with open("apikey_openai.txt") as file:
    openai.api_key = file.readline().strip()

training_data = [{
    "prompt": "Where is the billing ->",
    "completion": " You find the billing in the left-hand side menu.\n"
},{
    "prompt":"How do I upgrade my account ->",
    "completion": "Visit you user settings in the left-hand side menu, then click 'upgrade account' button at the "
                  "top.\n"
}]

file_name = "training_data.jsonl"

with open(file_name, "w") as output_file:
    for entry in training_data:
        json.dump(entry, output_file)
        output_file.write("\n")

#cmd = "openai -k " + api_key + " tools fine_tunes.prepare_data -f " + output_file.name
cmd = "openai -k " + api_key + " api fine_tunes.create -t " + output_file.name + " -m curie"
result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)

#!openai tools fine_tunes.prepare_data -f training_data.jsonl

while True:
    cmd = "openai -k " + api_key + " api fine_tunes.get -i " + id
    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    if "\"status\": \"succeeded\"" in str(result.stdout):
        print("Succeeded")
        break
    if result.stderr is not None:
        print("Error in cmd!")
        error = True
        break
    if "\"status\": \"failed\"" in str(result.stdout):
        print("Failed remote job!")
        error = True
        break
    time.sleep(20)
    print("Still training ...")
print("Done Training")