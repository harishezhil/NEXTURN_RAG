def cot_prompt(query, chunks):
    """
    WHAT DOES THIS DO??
    -----------
    Generates a structured prompt for a language model to perform Chain-of-Thought (CoT) reasoning 
    based on provided document context.

    Parameters:
    ----------
    query : str
        The user query to be answered by the model.

    chunks : list of dict
        A list of context chunks from source documents. Each chunk should contain:
            - 'content': list of text segments (e.g., paragraphs)
            - 'filename' or 'metadata': optional source information

    Returns:
    -------
    prompt : str
        A formatted prompt containing:
            - Context extracted from the input chunks
            - Reasoning instructions for the model
            - Sample Q&A examples to guide the output style

    Details:
    --------
    - Each chunk’s content is truncated to ensure compatibility with model token limits.
    - The prompt enforces step-by-step, transparent reasoning.
    - Useful in Retrieval-Augmented Generation (RAG) pipelines where factual grounding is necessary.
    """

    
    context = ""
    for chunk in chunks:
        filename = chunk.get("filename", chunk.get("metadata", {}).get("filename", "Unknown"))
        content = "\n\n".join(chunk["content"][:1000] for chunk in chunks)
        context += f"\n\nFrom {filename}:\n{content}"

    return f"""You are a smart assistant. Use clear, step-by-step reasoning to answer based on the provided context.

Before answering, follow these steps:
1. Understand the question and its assumptions.
2. Check if the context supports the question's premise (e.g., names, years, events).
3. If an assumption is incorrect or not found in the context, politely correct it.
4. If the answer isn't present, say so clearly.
5. If valid, break down the answer step by step, especially for time-based questions (year/month), and respond chronologically.

---

Example 1:
Context: Flipkart acquired Myntra in 2014 to expand its fashion e-commerce reach.
Question: Why did Flipkart acquire Myntra?
Answer:
1. Myntra was strong in fashion.
2. Flipkart lacked that domain strength.
3. Acquisition filled the gap.
Final Answer: To strengthen its position in fashion e-commerce.

---

Example 2:
Context: In May 2024, Google invested $350 million in Flipkart.
Question: What happened in May 2024?
Answer:
1. Google made a strategic investment.
2. The investment was worth $350 million.
Final Answer: Google invested $350 million in Flipkart.

---

Example 3:
Context: Flipkart was founded in October 2007 by Sachin Bansal and Binny Bansal.
Question: When did Jac find Flipkart?
Answer:
1. The question assumes someone named Jac founded Flipkart.
2. The context shows it was Sachin and Binny Bansal in 2007.
Final Answer: No one named Jac founded Flipkart. It was founded by Sachin and Binny Bansal.

---

Now, use the same reasoning for the following:

Context:
{context}

Question: {query}
Answer:"""
