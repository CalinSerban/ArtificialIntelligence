"""
Phase 4.1 — Gemini API basics
================================
First real call to an LLM API. Four pieces to get right:
  1. auth (load the key, create a client)
  2. a system prompt (instructions that shape behavior, separate from the
     user's actual message)
  3. generation params (temperature -- how random/creative the output is)
  4. token accounting (every API call costs tokens in + tokens out -- you
     need to be able to see these numbers, since they're what you pay for
     and what rate limits are measured in)

Docs for the SDK if you want to poke around: https://ai.google.dev/gemini-api/docs

Fill the TODOs, then run:  ./.venv/bin/python basics.py
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads .env in this directory, sets os.environ from it

MODEL = "gemini-2.5-flash"  # free-tier eligible; swap if this 404s for you

# TODO a: create the client. The SDK will look for the key itself if you
# pass it explicitly here -- read it from the environment (load_dotenv()
# above already loaded .env into os.environ).
#   client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# TODO b: build a GenerateContentConfig with:
#   - system_instruction: a short persona/instruction string of your choice
#     (e.g. "You are a terse assistant. Answer in one sentence.")
#   - temperature: try 0.0 first (near-deterministic) -- we'll experiment
#     with this value in a moment.
#   config = types.GenerateContentConfig(
#       system_instruction="...",
#       temperature=0.0,
#   )
config=types.GenerateContentConfig(
      system_instruction="You are a terse assistant. Answer in one sentence.",
      temperature=0.0,
   )

# TODO c: call client.models.generate_content(model=MODEL, contents=<your
# prompt string>, config=config) and store the result.
#   response = client.models.generate_content(
#       model=MODEL, contents="What's one good reason to unit test code?",
#       config=config,
#   )
response = client.models.generate_content(
    model=MODEL, contents="What's one good reason to unit test code?", config=config
)

print("--- response text ---")
print(response.text)

print("\n--- token usage ---")
print(response.usage_metadata)
