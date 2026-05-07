import json
from rag import get_retriever, llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def generate_quiz(session_id: str, num_questions: int = 5, lang: str = "en") -> list:
    retriever = get_retriever(session_id)
    docs = retriever.invoke("key concepts and important topics")
    context = "\n\n".join(d.page_content for d in docs)

    lang_instruction = "You MUST write all questions and options in Bengali (Bangla) language." if lang == "bn" else "Write all questions and options in English."

    prompt = PromptTemplate.from_template(
        f"""{lang_instruction}
Based on the following content, generate {{num_questions}} multiple-choice quiz questions.
Return ONLY a JSON array in this format:
[{{"question": "...", "options": ["A)...", "B)...", "C)...", "D)..."], "answer": "A"}}]

Content:
{{context}}
"""
    )

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "num_questions": num_questions}).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
