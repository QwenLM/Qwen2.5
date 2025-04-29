Langchain
==========================

.. attention:: 
    To be updated for Qwen3.

This guide helps you build a question-answering application based 
on a local knowledge base using ``Qwen2.5-7B-Instruct`` with ``langchain``.
The goal is to establish a knowledge base Q&A solution.

Basic Usage
-----------

The implementation process of this project includes 
loading files -> reading text -> segmenting text -> vectorizing text -> vectorizing questions 
-> matching the top k most similar text vectors with the question vectors -> 
incorporating the matched text as context along with the question into the prompt -> 
submitting to the Qwen2.5-7B-Instruct to generate an answer.
Below is an example:

.. code:: bash

   pip install langchain==0.0.174
   pip install faiss-gpu

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   from abc import ABC
   from langchain.llms.base import LLM
   from typing import Any, List, Mapping, Optional
   from langchain.callbacks.manager import CallbackManagerForLLMRun
   
   model_name = "Qwen/Qwen2.5-7B-Instruct"

   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype="auto",
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   class Qwen(LLM, ABC):
        max_token: int = 10000
        temperature: float = 0.01
        top_p = 0.9
        history_len: int = 3

        def __init__(self):
            super().__init__()

        @property
        def _llm_type(self) -> str:
            return "Qwen"

        @property
        def _history_len(self) -> int:
            return self.history_len

        def set_history_len(self, history_len: int = 10) -> None:
            self.history_len = history_len

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
        ) -> str:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        
        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return {"max_token": self.max_token,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "history_len": self.history_len}

After loading the Qwen2.5-7B-Instruct model, you should specify the txt file 
for retrieval.

.. code:: python

    import os
    import re
    import torch
    import argparse
    from langchain.vectorstores import FAISS
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from typing import List, Tuple
    import numpy as np
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.prompts.prompt import PromptTemplate
    from langchain.chains import RetrievalQA
    
    class ChineseTextSplitter(CharacterTextSplitter):
        def __init__(self, pdf: bool = False, **kwargs):
            super().__init__(**kwargs)
            self.pdf = pdf

        def split_text(self, text: str) -> List[str]:
            if self.pdf:
                text = re.sub(r"\n{3,}", "\n", text)
                text = re.sub('\s', ' ', text)
                text = text.replace("\n\n", "")
            sent_sep_pattern = re.compile(
                '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))') 
            sent_list = []
            for ele in sent_sep_pattern.split(text):
                if sent_sep_pattern.match(ele) and sent_list:
                    sent_list[-1] += ele
                elif ele:
                    sent_list.append(ele)
            return sent_list


    def load_file(filepath):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(textsplitter)
        write_check_file(filepath, docs)
        return docs


    def write_check_file(filepath, docs):
        folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        fp = os.path.join(folder_path, 'load_file.txt')
        with open(fp, 'a+', encoding='utf-8') as fout:
            fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
            fout.write('\n')
            for i in docs:
                fout.write(str(i))
                fout.write('\n')
            fout.close()

            
    def separate_list(ls: List[int]) -> List[List[int]]:
        lists = []
        ls1 = [ls[0]]
        for i in range(1, len(ls)):
            if ls[i - 1] + 1 == ls[i]:
                ls1.append(ls[i])
            else:
                lists.append(ls1)
                ls1 = [ls[i]]
        lists.append(ls1)
        return lists


    class FAISSWrapper(FAISS):
        chunk_size = 250
        chunk_conent = True
        score_threshold = 0
        
        def similarity_search_with_score_by_vector(
                self, embedding: List[float], k: int = 4
        ) -> List[Tuple[Document, float]]:
            scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
            docs = []
            id_set = set()
            store_len = len(self.index_to_docstore_id)
            for j, i in enumerate(indices[0]):
                if i == -1 or 0 < self.score_threshold < scores[0][j]:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not self.chunk_conent:
                    if not isinstance(doc, Document):
                        raise ValueError(f"Could not find document for id {_id}, got {doc}")
                    doc.metadata["score"] = int(scores[0][j])
                    docs.append(doc)
                    continue
                id_set.add(i)
                docs_len = len(doc.page_content)
                for k in range(1, max(i, store_len - i)):
                    break_flag = False
                    for l in [i + k, i - k]:
                        if 0 <= l < len(self.index_to_docstore_id):
                            _id0 = self.index_to_docstore_id[l]
                            doc0 = self.docstore.search(_id0)
                            if docs_len + len(doc0.page_content) > self.chunk_size:
                                break_flag = True
                                break
                            elif doc0.metadata["source"] == doc.metadata["source"]:
                                docs_len += len(doc0.page_content)
                                id_set.add(l)
                    if break_flag:
                        break
            if not self.chunk_conent:
                return docs
            if len(id_set) == 0 and self.score_threshold > 0:
                return []
            id_list = sorted(list(id_set))
            id_lists = separate_list(id_list)
            for id_seq in id_lists:
                for id in id_seq:
                    if id == id_seq[0]:
                        _id = self.index_to_docstore_id[id]
                        doc = self.docstore.search(_id)
                    else:
                        _id0 = self.index_to_docstore_id[id]
                        doc0 = self.docstore.search(_id0)
                        doc.page_content += " " + doc0.page_content
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
                doc.metadata["score"] = int(doc_score)
                docs.append((doc, doc_score))
            return docs


    if __name__ == '__main__':
        # load docs (pdf file or txt file)
        filepath = 'your file path'
        # Embedding model name
        EMBEDDING_MODEL = 'text2vec'
        PROMPT_TEMPLATE = """Known information:
        {context_str} 
        Based on the above known information, respond to the user's question concisely and professionally. If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided,' and do not include fabricated details in the answer. Please respond in English. The question is {question}"""
        # Embedding running device
        EMBEDDING_DEVICE = "cuda"
        # return top-k text chunk from vector store
        VECTOR_SEARCH_TOP_K = 3
        CHAIN_TYPE = 'stuff'
        embedding_model_dict = {
            "text2vec": "your text2vec model path",
        }
        llm = Qwen()
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],model_kwargs={'device': EMBEDDING_DEVICE})
        
        docs = load_file(filepath)
        
        docsearch = FAISSWrapper.from_documents(docs, embeddings)
        
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context_str", "question"]
        )

        chain_type_kwargs = {"prompt": prompt, "document_variable_name": "context_str"}
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=CHAIN_TYPE, 
            retriever=docsearch.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}), 
            chain_type_kwargs=chain_type_kwargs)
        
        query = "Give me a short introduction to large language models."  
        print(qa.run(query))

Next Step
---------

Now you can chat with Qwen2.5 use your own document. Continue
to read the documentation and try to figure out more advanced usages of
model retrieval!