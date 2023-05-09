import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(9))
def get_summarization_instruct(text, model="text-curie-001"):
    response = openai.Completion.create(
        model=model,
        prompt=text + "\n\ntl;dr\n",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    if not isinstance(response, dict):
        raise TypeError()

    with open(".memories/summary.txt", "w") as summary_file:
        summary_file.write(text + "\n\n" + response["choices"][0]["text"])

    return response
