import os

import openai
from dotenv import load_dotenv

from gptme.utils.web_search import web_question_answer

load_dotenv(dotenv_path=".env")

openai.api_key = os.environ["OPENAI_KEY"]

print("qnaing")
print(web_question_answer("https://huggingface.co/gpt2", "How many downloads did gpt-2 get last month?"))
