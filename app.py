from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-70b-instruct",
  messages=[{"content":"Provide me an article on the benefits of renewable energy.","role":"user"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices and chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

