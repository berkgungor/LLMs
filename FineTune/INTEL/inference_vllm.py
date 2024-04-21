from vllm import LLM, SamplingParams

# choosing the large language model
llm = LLM(model="/home//Repository/AI_Coach/INTEL/neural-chat_outputs_V2/checkpoint-75")

# setting the parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.90,max_tokens = 50)

prompt = input("Enter your prompt: ")

# generating the answer
answer = llm.generate(prompt,sampling_params)

# getting the generated text out from the answer variable
answer[0].outputs[0].text