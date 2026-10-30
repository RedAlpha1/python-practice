from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "apply", "application_status", "salary_query", "documents", "complaint", "other"
]
# dev: look at misses, tune the prompt. test: held out, only read the score.
Split = Literal["dev", "test"]


class IntentResult(BaseModel):
    intent: Intent
    city: str | None = Field(
        description="Indian city, official name (e.g. Bengaluru), else null"
    )


class LabeledCase(BaseModel):
    id: str
    text: str
    intent: Intent  # ← typo wala bug yahin pakda jata
    city: str | None
    note: str | None = None
    split: Split


PROMPT_V1 = "Classify this job-seeker message.\n\nMessage: {text}"

# Rules written from dev-split labels and notes only, never from test cases.
PROMPT_V2 = """\
You classify messages that job seekers send to a blue-collar jobs helpline in India.
Messages are often Hinglish, Hindi or English, with spelling mistakes.

Pick exactly one intent:
- apply: wants a job, asks about openings, or asks how to apply. A message that is
  only a city name is a reply to "which city?", so it is apply.
- application_status: has already applied, uploaded documents, or been given an
  interview or joining date, and asks what happens next or when.
- salary_query: asks about pay: amount, per-order rate, weekly or monthly payment,
  incentives, bonus, fuel allowance. If a message wants a job but asks about pay,
  it is salary_query: answer the question first.
- documents: which documents are needed, how or where to submit them, or a problem
  uploading them. Entering bank account details counts as documents.
- complaint: something has already gone wrong: money not received, nobody replies,
  bad treatment, the app is broken.
- other: greetings, thanks, "ok", questions about the bot, anything off-topic.

city: the city where the user wants work, with its official name (Bombay -> Mumbai,
Bangalore -> Bengaluru). null if no city is mentioned.

Message: {text}"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
