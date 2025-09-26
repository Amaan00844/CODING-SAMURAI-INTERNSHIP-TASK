#!/usr/bin/env python3
# advanced_rule_chatbot.py
# A more advanced rule-based chatbot with extra utilities and FAQs.

import re
import random
import datetime
import sys

# --------- Utilities ---------
def now_time():
    return datetime.datetime.now().strftime("%I:%M %p")

def now_date():
    return datetime.datetime.now().strftime("%A, %d %B %Y")

def calc(expr: str):
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\) \t]+", expr):
        return "Sorry, I can only do basic arithmetic."
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return "That expression looks invalid."

# --------- Knowledge Base (mini FAQ) ---------
FACTS = {
    "capital of france": "Paris is the capital of France 🇫🇷",
    "capital of germany": "Berlin is the capital of Germany 🇩🇪",
    "capital of india": "New Delhi is the capital of India 🇮🇳",
    "capital of usa": "Washington, D.C. is the capital of the United States 🇺🇸",
    "creator": "I was created by Amaan Chauhan",
    "meaning of life": "42 😉 (Hitchhiker’s Guide to the Galaxy reference)"
}

# --------- Intents ---------
INTENTS = [
    # Greetings
    (re.compile(r"\b(hi|hello|hey|yo|namaste)\b", re.I),
     lambda m, _: random.choice([
         "Hello! How can I help you today?",
         "Hey there ✨ What’s up?",
         "Hi! Ask me anything simple."
     ])),

    # Farewells
    (re.compile(r"\b(bye|goodbye|see ya|see you)\b", re.I),
     lambda m, _: random.choice([
         "Goodbye! Have a great day 👋",
         "See you soon!",
         "Bye! Take care."
     ])),

    # How are you
    (re.compile(r"\b(how are you|how's it going|how r u)\b", re.I),
     lambda m, _: random.choice([
         "I’m just a bunch of rules, but feeling helpful!",
         "Running at 100% 😊 How about you?"
     ])),

    # Name
    (re.compile(r"\b(what(?:'| i)s your name|who are you)\b", re.I),
     lambda m, _: "I’m RuleBot+, a smarter rule-based chatbot."),

    # Time / Date
    (re.compile(r"\b(what(?:'| i)s )?(the )?time\b", re.I),
     lambda m, _: f"It’s {now_time()} right now."),
    (re.compile(r"\b(what(?:'| i)s )?(the )?date\b", re.I),
     lambda m, _: f"Today is {now_date()}."),

    # Calculator
    (re.compile(r"\bcalc(?:ulate)?[: ]+(.+)", re.I),
     lambda m, _: f"Result: {calc(m.group(1))}"),
    (re.compile(r"\bwhat is ([0-9\.\+\-\*\/\(\) \t]+)\??$", re.I),
     lambda m, _: f"Result: {calc(m.group(1))}"),

    # Weather (simple random demo)
    (re.compile(r"\b(weather|temperature)\b", re.I),
     lambda m, _: random.choice([
         "It looks sunny ☀️ outside.",
         "It seems cloudy ☁️ today.",
         "Chilly and windy 🌬️ right now.",
         "Could be rainy 🌧️, carry an umbrella!"
     ])),

    # Jokes
    (re.compile(r"\b(tell me a joke|joke)\b", re.I),
     lambda m, _: random.choice([
         "Why do programmers prefer dark mode? Because light attracts bugs!",
         "I told my computer I needed a break—it said 'No problem, I’ll go to sleep.'",
         "There are 10 types of people: those who understand binary and those who don’t."
     ])),

    # FAQ (mini knowledge base lookup)
    (re.compile(r"\b(capital of \w+|creator|meaning of life)\b", re.I),
     lambda m, _: FACTS.get(m.group(0).lower(), "Hmm, I don’t know that yet 🤔")),

    # Help
    (re.compile(r"\b(help|what can you do)\b", re.I),
     lambda m, _: (
         "I can greet you, tell the time/date, do math, share weather updates, "
         "tell jokes, answer simple facts (like 'capital of France'), and chat a bit. "
         "Type 'exit' to quit."
     )),

    # Small talk reflections (ELIZA-style)
    (re.compile(r"\bI feel (.+)", re.I),
     lambda m, _: f"Why do you feel {m.group(1)}?"),
    (re.compile(r"\bI am (.+)", re.I),
     lambda m, _: f"What makes you {m.group(1)}?"),
    (re.compile(r"\bBecause (.+)", re.I),
     lambda m, _: f"Is that the main reason {m.group(1)}?"),
]

# --------- Fallbacks ---------
FALLBACKS = [
    "I’m not sure about that yet 🤔. Try 'help' to see what I can do.",
    "Hmm, I didn’t catch that. You can ask me about time, date, weather, or say 'joke'.",
    "I’m a rule-based bot—could you rephrase or try 'calc: 2+2'?",
]

EXIT_PAT = re.compile(r"\b(exit|quit|close|bye)\b", re.I)

def respond(user_input: str) -> str:
    for pattern, handler in INTENTS:
        m = pattern.search(user_input)
        if m:
            return handler(m, user_input)
    return random.choice(FALLBACKS)

def main():
    print("RuleBot+ 🤖 at your service! Type 'help' to see abilities. Type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Bye! 👋")
            sys.exit(0)

        if not user:
            continue
        if EXIT_PAT.search(user):
            print("Bot: Goodbye! 👋")
            break

        print("Bot:", respond(user))

if __name__ == "__main__":
    main()
