LlamaIndex
==========

To connect Qwen1.5. with external data, such as documents, web pages, etc., we recommend using `LlamaIndex <https://www.llamaindex.ai/>`__.
This guide helps you quickly implement retrieval-augmented generation (RAG) using LlamaIndex with Qwen1.5.

Preparation
--------------------------------------

To implement RAG, 
we advise you to download Qwen1.5-Chat and install the LlamaIndex-related packages first. 
Qwen1.5-Chat supports conversations in multiple languages, including English and Chinese.
We recommend using the ``bge-base-en-v1.5`` model to retrieve from English documents, and you can also download the ``bge-base-zh-v1.5`` model to retrieve from Chinese documents. 
You can also choose ``bge-large`` or ``bge-small`` as the embedding model depending on your computing resources.
The following is a simple code snippet showing how to do this:

.. code:: bash

   pip install llama-index
   pip install llama-index-llms-huggingface
   pip install llama-index-readers-web

.. code:: python

  from modelscope import snapshot_download
  qwen_dir = snapshot_download('qwen/Qwen1.5-7B-Chat',cache_dir=".)
  embed_dir = snapshot_download('AI-ModelScope/bge-base-en-v1.5',cache_dir=".")

Set Parameters
--------------------------------------

Now we can set up LLM, embedding model, and the related configurations.                               
You can modify the context window size or text chunk size, depending on your computational resources.
Qwen 1.5 model families support a maximum of 32K context window size.

.. code:: python
  
    import torch
    from llama_index.core import Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Set prompt template for generation (optional)
    from llama_index.core import PromptTemplate
  
    def completion_to_prompt(completion):
        return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"
    
    def messages_to_prompt(messages):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|system|>\n{message.content}</s>\n"
            elif message.role == "user":
                prompt += f"<|user|>\n{message.content}</s>\n"
            elif message.role == "assistant":
                prompt += f"<|assistant|>\n{message.content}</s>\n"
    
        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n</s>\n" + prompt
    
        prompt = prompt + "<|assistant|>\n"
    
        return prompt
    
    # Set Qwen1.5 as the language model and set generation config
    Settings.llm = HuggingFaceLLM(
        model_name=qwen_dir,
        tokenizer_name=qwen_dir,
        context_window=30000,
        max_new_tokens=2000,
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="auto",
    )

    # Set embedding model                       
    Settings.embed_model = HuggingFaceEmbedding(
        model_name = embed_dir
    )

    # Set the size of the text chunk for retrieval
    Settings.transformations = [SentenceSplitter(chunk_size=1024)]

Build Index
--------------------------------------

Now we can build index from documents or websites.

The following code snippet demonstrates how to build an index for files (regardless of whether they are in PDF or TXT format) in a local folder named 'document'.                               

.. code:: python
    
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    
    documents = SimpleDirectoryReader("./document").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model, 
        transformations=Settings.transformations
    )

The following code snippet demonstrates how to build an index for the content in a list of websites.                               
                               
.. code:: python
                               
    from llama_index.readers.web import SimpleWebPageReader
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    
    documents = SimpleWebPageReader(html_to_text=True).load_data(
        ["web_address_1","web_address_2",...]
    )
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model, 
        transformations=Settings.transformations
    )

To save and load the index, you can use the following code snippet.                              

.. code:: python

    from llama_index.core import StorageContext, load_index_from_storage

    # save index
    storage_context = StorageContext.from_defaults(persist_dir="save")
    
    # load index
    index = load_index_from_storage(storage_context)
                            
                               
RAG
-------------------

Now you can perform queries, and Qwen1.5 will answer based on the content of the indexed documents.                               
                               
.. code:: python

  query_engine = index.as_query_engine()
  your_query = "<your query here>"                             
  print(query_engine.query(your_query).response)

