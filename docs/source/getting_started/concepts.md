# Key Concepts

## Qwen

Qwen (Chinese: 通义千问; pinyin: _Tongyi Qianwen_) is the large language model and large multimodal model series of the Qwen Team, Alibaba Group. 
Qwen is capable of natural language understanding, text generation, vision understanding, audio understanding, tool use, role play, playing as AI agent, etc. 
Both language models and multimodal models are pre-trained on large-scale multilingual and multimodal data and post-trained on quality data for aligning to human preferences.

There is the proprietary version hosted exclusively at [Alibaba Cloud \[zh\]](https://help.aliyun.com/zh/model-studio/developer-reference/tongyi-qianwen-llm/) and the open-weight version.

The spectrum for the open-weight models spans over
- Qwen: the language models
    - [Qwen](https://github.com/QwenLM/Qwen): 1.8B, 7B, 14B, and 72B models
    - [Qwen1.5](https://github.com/QwenLM/Qwen1.5/tree/v1.5): 0.5B, 1.8B, 4B, 14BA2.7B, 7B, 14B, 32B, 72B, and 110B models
    - [Qwen2](https://github.com/QwenLM/Qwen2): 0.5B, 1.5B, 7B, 57A14B, and 72B models 
- Qwen-VL: the vision-language models
    - [Qwen-VL](https://github.com/QwenLM/Qwen-VL): 7B-based models
- Qwen-Audio: the audio-language models
    - [Qwen-Audio](https://github.com/QwenLM/Qwen-Audio): 7B-based models
    - [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio): 7B-based models
- CodeQwen: the language models for coding
    - [CodeQwen1.5](https://github.com/QwenLM/CodeQwen1.5): 7B models

**In this document, our focus is Qwen, the language models.**

## Causal Language Models

Causal language models, also known as autoregressive language models or decoder-only language models, are a type of machine learning model designed to predict the next token in a sequence based on the preceding tokens. 
In other words, they generate text one token at a time, using the previously generated tokens as context. 
The "causal" aspect refers to the fact that the model only considers the past context (the already generated tokens) when predicting the next token, not any future tokens.

Causal language models are widely used for various natural language processing tasks involving text completion and generation. 
They have been particularly successful in generating coherent and contextually relevant text, making them a cornerstone of modern natural language understanding and generation systems.

**Takeaway: Qwen models are causal language models suitable for text completion.**

:::{dropdown} Learn more about language models

They are three main kinds of models that are commonly referred to as language models in deep learning: 
- Sequence-to-sequence models: T5 and the likes

  Sequence-to-sequence models use both an encoder to capture the entire input sequence and a decoder to generate an output sequence.
  They are widely used for tasks like machine translation, text summarization, etc.

- Bidirectional models or encoder-only models: BERT and the likes
  
  Bidirectional models can access both past and future context in a sequence during training.
  They cannot generate sequential outputs in real-time due to the need for future context.
  They are widely used as embedding models and subsequently used for text classification.
  
- Casual language models or decoder-only models: GPT and the likes

  Causal language models operate unidirectionally in a strictly forward direction, predicting each subsequent word based only on the previous words in the sequence. 
  This unidirectional nature ensures that the model's predictions do not rely on future context, making them suitable for tasks like text completion and generation. 
:::

### Pre-training & Base models

Base language models are foundational models trained on extensive corpora of text to predict the next word in a sequence. 
Their main goal is to capture the statistical patterns and structures of language, enabling them to generate coherent and contextually relevant text. 
These models are versatile and can be adapted to various natural language processing tasks through fine-tuning. 
While adept at producing fluent text, they may require in-context learning or additional training to follow specific instructions or perform complex reasoning tasks effectively.
For Qwen models, the base models are those without "-Instruct" indicators, such as Qwen2-7B and Qwen2-72B.

**Takeaway: Use base models for in-context learning, downstream fine-tuning, etc.**

### Post-training & Instruction-tuned models

Instruction-tuned language models are specialized models designed to understand and execute specific instructions in conversational styles.
These models are fine-tuned to interpret user commands accurately and can perform tasks such as summarization, translation, and question answering with improved accuracy and consistency. 
Unlike base models, which are trained on large corpora of text, instruction-tuned models undergo additional training using datasets that contain examples of instructions and their desired outcomes, often in multiple turns.
This kind of training makes them ideal for applications requiring targeted functionalities while maintaining the ability to generate fluent and coherent text.
For Qwen models, the instruction-tuned models are those with the "-Instruct" suffix, such as Qwen2-7B-Instruct and Qwen2-72B-Instruct. [^instruct-chat]

**Takeaway: Use instruction-tuned models for conducting tasks in conversations, downstream fine-tuning, etc.**

[^instruct-chat]: Previously, they are known as the chat models and with the "-Chat" suffix. Starting from Qwen2, the name is changed to follow the common practice. For Qwen, "-Instruct" and "-Chat" should be regarded as synonymous.


## Tokens & Tokenization

Tokens represent the fundamental units that models process and generate. 
They can represent texts in human languages (regular tokens) or represent specific functionality like keywords in programming languages (control tokens [^special]).
Typically, a tokenizer is used to split text into regular tokens, which can be words, subwords, or characters depending on the specific tokenization scheme employed, and furnish the token sequence with control tokens as needed.
The vocabulary size, or the total number of unique tokens a model recognizes, significantly impacts its performance and versatility. 
Larger language models often use sophisticated tokenization methods to handle the vast diversity of human language while keeping the vocabulary size manageable.
Qwen use a relatively large vocabulary of 151,646 tokens in total.

[^special]: Control tokens can be called special tokens. However, the meaning of special tokens need to be interpreted based on the contexts: special tokens may contain extra regular tokens.

**Takeaway: Tokenization method and vocabulary size is important.**

### Byte-level Byte Pair Encoding

Qwen adopts a subword tokenization method called Byte Pair Encoding (BPE), which attempts to learn the composition of tokens that can represent the text with the fewest tokens. 
For example, the string " tokenization" is decomposed as " token" and "ization" (note that the space is part of the token).
Especially, the tokenization of Qwen ensures that there is no unknown words and all texts can be transformed to token sequences.

There are 151,643 tokens as a result of BPE in the vocabulary of Qwen, which is a large vocabulary efficient for diverse languages.
As a rule of thumb, 1 token is 3~4 characters for English texts and 1.5~1.8 characters for Chinese texts. 

**Takeaway: Qwen processes texts in subwords and there are no unknown words.**

:::{dropdown} Learn more about tokenization in Qwen
Qwen uses byte-level BPE (BBPE) on UTF-8 encoded texts. 
It starts by treating each byte as a token and then iteratively merges the most frequent pairs of tokens occurring the texts into larger tokens until the desired vocabulary size is met. 

In byte-level BPE, minimum 256 tokens are needed to tokenize every piece of text and avoid the out of vocabulary (OOV) problem.
In comparison, character-level BPE needs every Unicode character in its vocabulary to avoid OOV and the Unicode Standard contains 154,998 characters as of Unicode Version 16.0.

One limitation to keep in mind for byte-level BPE is that the individual tokens in the vocabulary may not be seemingly semantically meaningful or even valid UTF-8 byte sequences, and in certain aspects, they should be viewed as a text compression scheme.
:::

### Control Tokens & Chat Template

Control tokens and chat templates both serve as mechanisms to guide the model's behavior and outputs. 

Control tokens are special tokens inserted into the sequence that signifies meta information.
For example, in pre-training, multiple documents may be packed into a single sequence.
For Qwen, the control token "<|endoftext|>" is inserted after each document to signify that the document has ended and a new document will proceed.

Chat templates provide a structured format for conversational interactions, where predefined placeholders or prompts are used to elicit responses from the model that adhere to a desired dialogue flow or context.
Different models may use different kinds of chat template to format the conversations. 
It is crucial to use the designated one to ensure the precise control over the LLM's generation process.

Qwen uses the following format (ChatML[^chatml]), making use of control tokens to format each turn in the conversations
```text
<|im_start|>{{role}}
{{content}}<|im_end|>
```
The user input take the role of `user` and the model generation takes the role of `assistant`. 
Qwen also supports the meta message that instruct the model to perform specific actions or generate text with certain characteristics, such as altering tone, style, or content, which takes the role of `system` and the content defaults to "You are a helpful assistant."

The following is a full example:
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
hello<|im_end|>
<|im_start|>assistant
Hello! How can I assist you today? Let me know if you have any questions or need help with something. I'm here to help!<|im_end|>
<|im_start|>user
Explain large language models like I'm 5.<|im_end|>
<|im_start|>assistant
Sure, I'd be happy to explain large language models in a simple way!
Imagine that you have a friend who is really good at playing with words. This friend has read so many books and talked to so many people that they know how to use words in all sorts of different ways. They can tell stories, answer questions, and even make up new words if they need to.
Now imagine that this friend is actually a computer program, called a "large language model". It's been trained on lots and lots of text, like books, articles, and conversations, so it knows how to use words just like your word-loving friend does. When you ask the model a question or give it a task, it uses all the knowledge it's learned to come up with a response that makes sense.
Just like your friend might sometimes make mistakes or say things in a funny way, the large language model isn't perfect either. But it's still really impressive, because it can understand and generate human language in a way that was once thought impossible for machines!<|im_end|><|endoftext|>
```

There are 3 control tokens in the vocabulary of Qwen, making the vocabulary size totaling 151,646.

**Takeaway: Qwen uses ChatML with 3 control tokens for chat template.**

[^chatml]: For historical reference only, ChatML is first described by the OpenAI Python SDK. The last available version is [this](https://github.com/openai/openai-python/blob/v0.28.1/chatml.md). Please also be aware that that document lists use cases intended for OpenAI models. For Qwen2 models, please only use as in our guide.

## Length Limit

As Qwen models are causal language models, in theory there is only one length limit of the entire sequence.
However, since there is often packing in training and each sequence may contain multiple individual pieces of texts. 
**How long the model can generate or complete ultimately depends on the use case and in that case how long each document (for pre-training) or each turn (for post-training) is in training.**

For Qwen2, the packed sequence length in training is 32,768 tokens.[^yarn]
The maximum document length in pre-training is this length.
The maximum message length for user and assistant is different in post-training.
In general, the assistant message could be up to 2048 tokens and for tasks with less variation like tables to HTML, it could be 6-8K tokens.

[^yarn]: The sequence length can be extended to 131,072 tokens for Qwen2-7B and Qwen2-72B models with YaRN.
         Please refer to the model card on how to enable YaRN in vLLM.

**Takeaway: Qwen2 models can process texts of 32K or 128K tokens but not all of them can be output.**
