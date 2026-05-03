import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model_name

    def generate(self, query, context_chunks):
        # Flatten chunks into one string
        context_text = "\n\n".join([c['content'] for c in context_chunks])
        
        prompt = f"""
        You are a helpful assistant. Use the following context to answer the question.
        If the answer isn't in the context, say you don't know.
        
        CONTEXT:
        {context_text}
        
        QUESTION: 
        {query}
        
        ANSWER:
        """
        
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return chat_completion.choices[0].message.content