from langchain_ollama import OllamaLLM
from app.core.config import settings
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = OllamaLLM(model=settings.llm_model, temperature=0, base_url=settings.ollama_base_url, client_kwargs={"timeout":120})
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        self.validator = OllamaLLM(model=settings.llm_model, temperature=0, base_url=settings.ollama_base_url, client_kwargs={"timeout":120})
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), before_sleep=before_sleep_log(logger, logging.WARNING)) 
    def generate_answer(self, prompt):      
        return self.llm.invoke(prompt)
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), before_sleep=before_sleep_log(logger, logging.WARNING))
    def generate_validation(self, validation_prompt):
        return self.validator.invoke(validation_prompt).strip().upper().rstrip(".")      
      
    def askQuestion(self, question:str, document_id:str | None = None)->dict:
        search_filter = None
        if document_id:
            search_filter = {"document_id":document_id}
        try:    
            results = self.vectorstore.similarity_search_with_score(question, k=8, filter=search_filter)
        except Exception:
            return{"answer":"Document search is currently unavailable. Please try again later",
                   "sources":[]}
            
        print("\nINITIAL RETRIEVED CHUNKS:")
        for chunk, score in results:
            print("Score:", score)
            print("Source:", chunk.metadata.get("source"))
            print("Page:", chunk.metadata.get("page"))
            print("Text:", chunk.page_content[:300])
        
        unique_results = []
        seen_chunks = set()
        for chunk, score in results:
            text = chunk.page_content.strip()
            if text not in seen_chunks:
                seen_chunks.add(text)
                unique_results.append((chunk, score))
        results = unique_results
            
        
        if results:
            passages = [{"id":str(index),
                         "text":chunk.page_content,
                         "meta":chunk.metadata}
                        for index, (chunk, score) in enumerate(results)
                        ] 
            rerank_request = RerankRequest(query=question, passages=passages)
            
            try:
                reranked_results = self.ranker.rerank(rerank_request)
            except Exception:
                return{"answer":"Unable to process search results. Please try again later",
                       "sources":[]}
            print("No of FLASHRANK RESULTS:", len(reranked_results))
            
            results = [(Document(page_content=item["text"],
                                 metadata=item["meta"]),
                                 item["score"])
                       for item in reranked_results]
            
            for i, (chunk, score) in enumerate(results):
                print(f"\nRERANKED CHUNK {i + 1}")
                print("Score:", score)
                print("Page:", chunk.metadata.get("page"))
                print("Text:", chunk.page_content[:300])
            
         # Check whether the closest chunk is relevant enough
        if not results:
            return {"answer":"I could not find that information in the document.",
                    "sources":[]}
            
        sources = []
        unique_sources = set()
        for chunk, score in results:
            document = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "Unknown")
            
            source_key = (document, page)
            if source_key not in unique_sources:
                unique_sources.add(source_key)
                sources.append({"document":document, "page":page})
        
        context = "\n\n".join(chunk.page_content for chunk, score in results)
        context = "\n\n".join(
                            f"Source: {chunk.metadata.get('source', 'Unknown')}\n"
                            f"Page: {chunk.metadata.get('page', 'Unknown')}\n"
                            f"Content: {chunk.page_content}"
                            for chunk, score in results
                            )
        print("RETRIEVED CONTEXT:", context)
        print("QUESTION:", question)
        prompt = f"""
                    You are a careful question-answering assistant for uploaded documents.

                    Answer the user's question using only the retrieved context.

                    Rules:
                    1. Answer all parts of the question that the context supports.
                    2. Never invent facts or make unsupported assumptions.
                    3. Use information from the context, not outside knowledge.
                    4. You may perform simple calculations using facts explicitly
                        stated in the context. Check all arithmetic carefully.
                    5. When calculating age, use the person's date of birth if
                        available. Calculate age based on the requested date or year.
                    6. If the exact date is unknown, state the possible answers
                        or explain the uncertainty.
                    7. Do not infer a person's age from the date a document was
                         executed or from the legal age of majority.
                    8. If different documents or passages give conflicting facts,
                        clearly explain the conflict and identify the relevant dates,
                        events, or sources when available.
                    9. Do not treat an allegation, an age recorded in a document,
                        and an established fact as necessarily equivalent.
                    10. If only part of the question can be answered, provide the
                        supported part and explain what is missing.
                    11. If the context does not contain enough information to answer,
                        say: "I could not find that information in the document."
                    12. Keep the answer clear, concise, and directly relevant.
                    - Distinguish facts explicitly stated in the context from conclusions
                        calculated or inferred from those facts.
                    - Check whether the context contains conflicting or incomplete information.
                    - When the evidence is insufficient to reach one definite conclusion,
                        explain the uncertainty instead of guessing.
                    - Make calculations carefully and state any assumptions needed.
                    - Answer only what the question asks, using relevant evidence from
                        the retrieved context.
                    -Before answering, copy each requested item's details directly from its matching context entry; never infer details from neighboring entries.
                    Context:{context}
                    Question:{question}  
                                        
                    Answer:
                  """
        try:   
            answer = self.generate_answer(prompt)
        except Exception:
            return{"answer": "The AI model is currently unavailable. Please try again later.",
                   "sources": []}
            
        print(f"Answer : {answer}")
        
        fallback = "I could not find that information in the document."
                
        if fallback.lower() in answer.lower():
            return {"answer": "I could not find that information in the document.",
                    "sources": []
                    }
        
        validation_prompt = f"""
                You are a practical answer validator for a document-based RAG chatbot.

                Your job is to check whether the answer is reasonably supported by the
                retrieved context, not whether it uses the exact same words.

                Rules:
                1. Understand the user's question and the meaning of the answer.
                2. Accept paraphrasing, summaries, and reasonable conclusions directly
                    supported by the context.
                3. For questions requiring multiple facts, consider all relevant parts
                       of the context together.
                4. Accept simple calculations when they follow from the provided facts.
                5. Do not reject an answer just because the question is phrased differently
                    from the document.
                6. The context may contain different statements about the same fact.
                Accept an answer that accurately reports each statement and does not
                claim that conflicting statements are consistent.
                7. Return UNSUPPORTED only when the answer contains a material claim
                    that is contradicted by, or cannot reasonably be supported by, the context.
                8.For multiple exhibits or documents, verify each item's details separately.
                    Never mix up numbers, dates, names, or descriptions. If unclear, don't guess.
                8. If the answer is reasonably supported, return SUPPORTED.

                Return exactly one word: SUPPORTED or UNSUPPORTED.

                Context:
                {context}

                Question:
                {question}

                Answer:
                {answer}
                """
        try:  
            validation = self.generate_validation(validation_prompt)
        except Exception:
            return{"answer":"Answer validation is temporarily unavailable. Please try again later",
                   "sources":[]}
            
        print("VALIDATOR PROMPT:", validation_prompt)
        print("GENERATED ANSWER:", repr(answer))
        print("RAW VALIDATION :", repr(validation))
        
        if validation != "SUPPORTED":
            return{"answer": "I could not find that information in the document.",
                    "sources": []} 
        
        print("QUESTION SENT:", repr(question))
        print("Raw LLM ans: ", repr(answer))
        return {"answer":answer, "sources":sources}
        
        