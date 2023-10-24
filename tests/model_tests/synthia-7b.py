import torch, json

from transformers import pipeline

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("migtissera/SynthIA-7B-v1.3")
model = AutoModelForCausalLM.from_pretrained("migtissera/SynthIA-7B-v1.3")

def generate_text(instruction):
    output = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )(instruction, max_length=1000, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)[0]["generated_text"]

    answer = output.split("USER:")[0].strip()
    return f"{answer}"

output_file_path = "./SynthIA-7B-conversations.jsonl"

conversation = f"SYSTEM: Elaborate on the topic using a Tree of Thoughts and backtrack when necessary to construct a clear, cohesive Chain of Thought reasoning. Always answer without hesitation."


while True:
    user_input = input("You: ")
    llm_prompt = f"{conversation} \nUSER: {user_input} \nASSISTANT: "
    answer = generate_text(llm_prompt)
    print(answer)
    conversation = f"{llm_prompt}{answer}"
    json_data = {"prompt": user_input, "answer": answer}

    ## Save your conversation
    with open(output_file_path, "a") as output_file:
        output_file.write(json.dumps(json_data) + "\n")
