#!/usr/bin/env python3
"""Debug the decomposition and multi-hop answering flow."""
import sys
sys.path.insert(0, '.')

from src.generator import OllamaGenerator
from src.retriever import BM25Retriever
from src.green_search import Judge

llm = OllamaGenerator(model_name='llama3:8b', temperature=0.0, max_tokens=256, timeout=120)

# Load retriever
retriever = BM25Retriever()
retriever.load_index('data/indexes/faiss.bm25.pkl', './data/processed/passages.json')

question = 'Were Scott Derrickson and Ed Wood of the same nationality?'
ground_truth = 'yes'

print(f'Question: {question}')
print(f'Ground truth: {ground_truth}')
print('=' * 60)

# Step 1: DECOMPOSE
print('\n=== STEP 1: DECOMPOSE ===')
decompose_prompt = f"""Break down this question into 2-3 simpler sub-questions that would help answer it:

Question: {question}

Sub-questions:"""

result = llm.generate(decompose_prompt, [])
sub_qs_raw = result.get('answer', '')
print(f'Raw decomposition output:\n{sub_qs_raw}')

# Parse sub-questions
sub_qs = []
for line in sub_qs_raw.split('\n'):
    line = line.strip()
    cleaned = line.lstrip('0123456789.-) ').strip()
    if cleaned and '?' in cleaned and len(cleaned) > 10:
        sub_qs.append(cleaned)

print(f'\nParsed sub-questions: {sub_qs}')

# Step 2: Answer each sub-question with retrieval
print('\n=== STEP 2: ANSWER SUB-QUESTIONS ===')
sub_answers = {}
for i, sq in enumerate(sub_qs[:2]):
    print(f'\n--- Sub-question {i+1}: {sq} ---')
    
    # Retrieve for this sub-question
    passages, scores = retriever.retrieve(sq, top_k=5)
    print(f'Retrieved {len(passages)} passages:')
    for p in passages[:3]:
        print(f'  - {p.get("title", "?")[:40]}: {p.get("text", "")[:60]}...')
    
    # Generate answer
    context = '\n\n'.join([f"[{j+1}] {p.get('title', 'Unknown')}: {p.get('text', '')[:200]}" for j, p in enumerate(passages)])
    gen_prompt = f"""Answer the question based on the context below.
Be concise - just give the answer.

Context:
{context}

Question: {sq}

Answer:"""
    
    result = llm.generate(gen_prompt, [])
    answer = result.get('answer', '')
    sub_answers[sq] = answer
    print(f'Answer: {answer}')

# Step 3: Final answer using sub-answers
print('\n=== STEP 3: FINAL ANSWER ===')
sub_info = "\n".join([f"- {q}: {a}" for q, a in sub_answers.items()])
final_prompt = f"""Answer the question based on the information below.
Be concise - just give the answer (yes or no).

Relevant information:
{sub_info}

Question: {question}

Answer:"""

print(f'Final prompt:\n{final_prompt}')
result = llm.generate(final_prompt, [])
final_answer = result.get('answer', '')
print(f'\nFinal answer: {final_answer}')

# Judge
judge = Judge(llm=llm, use_llm_judge=False)
is_correct = judge.judge(final_answer, ground_truth)
print(f'Contains "{ground_truth}": {is_correct}')
