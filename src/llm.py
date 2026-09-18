import ollama

MODEL_NAME = "llama3" 


def classify_intent(query):
   
    prompt = f"""Classify the question into exactly ONE category. Reply with ONLY one word.

Examples:
Question: "how many tables are there?"
Answer: STATS

Question: "how many images are in the pdf?"
Answer: STATS

Question: "how many pages does the document have?"
Answer: STATS

Question: "explain the documents"
Answer: SUMMARY

Question: "what does image 2 say?"
Answer: SUMMARY

Question: "read the image"
Answer: SUMMARY

Question: "describe image 2"
Answer: SUMMARY

Question: "summarize the pdf"
Answer: SUMMARY

Question: "what is the capital of france?"
Answer: NORMAL

Question: "who wrote this?"
Answer: NORMAL

Question: "what year did this happen?"
Answer: NORMAL

Question: "what is the main topic of the document?"
Answer: NORMAL

Question: "what is the images about?"
Answer: NORMAL

Question: "what is the summary of the sample 2?"
Answer: NORMAL

Now classify this question:
Question: "{query}"
Answer with ONLY one word: STATS, SUMMARY, or NORMAL.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 10}  
    )

    answer = response["message"]["content"].strip().upper()

    if "STATS" in answer:
        return "STATS"
    elif "SUMMARY" in answer:
        return "SUMMARY"
    else:
        return "NORMAL"


def generate_answer(context, question):
    """For normal, specific questions — strict, only uses given context."""
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the given documents."

Context:
{context}

Question: {question}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 300}  # cap answer length -> faster response
    )

    return response["message"]["content"]


def generate_summary(context):
    """For 'summarize the document' type questions — encourages a full,
    detailed, human-style explanation instead of a short cautious answer."""
    prompt = f"""You are explaining this document to someone who has never read it.

Below are excerpts taken from different parts of the document. Using them, write a
clear, detailed, human-style summary — as if a knowledgeable person is explaining it
out loud. Cover the main topics, key points, and overall structure of the document.
Use your own words, don't just copy lines. Be thorough — cover everything important
that appears across the excerpts, not just the first part.

Document excerpts:
{context}

Now write the full, detailed summary:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 500}  # summaries can be a bit longer than normal answers
    )

    return response["message"]["content"]