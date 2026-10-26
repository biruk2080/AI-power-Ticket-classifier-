from langchain_typesafe import Noul, TypeSafeClassifier
from langchain_typesafe import Choice, TypeSafeClassifier
import os
from dotenv import load_dotenv
os.getenv("OPENAI_API_KEY")
load_dotenv()
api_key = os.getenv("TYPESAFE_API_KEY")
classifier = TypeSafeClassifier(api_key=api_key)

response = classifier.invoke({
    "state": (
        "The deploy failed twice and customers are seeing 500s. "
        "Can someone look now?"
    ),
    "questions": {
        "urgent": Noul(
            instructions="Does this need attention right now?"
        ),
    },
})

urgency = response.nouls["urgent"].noul

print("Urgency:", urgency)



from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)
4
router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model="openai:gpt-5.6-luna",
            criteria="Direct lookups, extraction, and localized changes.",
        ),
        "powerful": ModelChoice(
            model="openai:gpt-5.6-sol",
            criteria="Architecture and high-stakes decisions.",
        ),
    },
    instructions="Choose the least costly model that can complete the task.",
)

agent = create_agent(
    "openai:gpt-5.6-luna",
    middleware=[router],
)

agent = create_agent("openai:gpt-5.6-luna", middleware=[router])

test_cases = [
    "Extract the email address from this text: Contact us at help@example.com.",
    (
        "Design a reliable architecture for an online store that must "
        "handle traffic spikes, payment failures, and database outages."
    ),
]

for i, prompt in enumerate(test_cases, start=1):
    print(f"\n--- Test {i} ---")
    print("Prompt:", prompt)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    print("Selected route:", result["model_route"].choice)
    print("Confidence:", result["model_route"].confidence)
    

    DEPARTMENTS = {
    "sales": "Pricing, products, plans, purchasing, and sales enquiries.",
    "support": "Technical problems, bugs, troubleshooting, and account access.",
    "billing": "Invoices, payments, refunds, subscriptions, and charges.",
    "hr": "Employee policies, hiring, payroll, and leave.",
    "human_review": "Questions that do not clearly fit another department.",
}


def classify_department(question: str) -> str:
    result = classifier.invoke({
        "state": question,
        "questions": {
            "department": Choice(
                instructions=(
                    "Choose the single department best suited "
                    "to handle the user's question. if not sure,"
                    " give a low confidence score bellow 0.60 and route to human review. "

                    # "route to human review if uncertain."
                ),
                criteria=DEPARTMENTS,
            )
        },
    })
    

    answer = result.choices["department"]
    print("Department probabilities:", answer.probabilities)
    print("Classification confidence:", answer.confidence)
    # Avoid automatic routing when the classifier is uncertain.
    if answer == "human_review" or answer.confidence < 0.60:
        return "human_review"

    return answer.choice

test_cases = [
    "What plans and prices do you offer?",
    "My account crashes whenever I log in.",
    "I was charged twice for my subscription.",
    "How many annual leave days do employees receive?",
    "what is your name?",
]

for i, question in enumerate(test_cases, start=1):
    print(f"\n--- Test {i} ---")
    print("Question:", question)

    department = classify_department(question)
    print("Selected department:", department)