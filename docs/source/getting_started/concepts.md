# Key Concepts

## Qwen

Qwen (Chinese: 通义千问; pinyin: _Tongyi Qianwen_) is the large language model and large multimodal model series of the Qwen Team, Alibaba Group. 
Qwen is capable of natural language understanding, text generation, vision understanding, audio understanding, tool use, role play, playing as AI agent, etc. 
Both language models and multimodal models are pre-trained on large-scale multilingual and multimodal data and post-trained on quality data for aligning to human preferences.

There is the proprietary/closed source version and the open-weight version. You can learn more about the proprietary models at Alibaba Cloud Model Studio ([China Site](https://help.aliyun.com/zh/model-studio/getting-started/models#9f8890ce29g5u) \[zh\], [International Site](https://www.alibabacloud.com/en/product/modelstudio)).

In this document, our focus is Qwen, the language models.

## Qwen3

Qwen3 is the newest edition of the Qwen language models, featuring balanced model sizes, enhanced capbilities, hybrid thinking modes, and more language support:
- The mixture-of-experts (MoE) models are reintroduced with the Qwen3-30B-A3B and Qwen3-235B-A22B. The largest dense models are now Qwen3-32B.
- Hybrid thinking mode is designed so that thinking and non-thinking (instruct) can be achieved without changing models, simplifying deployment and making alternating thinking and non-thinking in a single chat possible.
- With 119 languages (and dialects), Qwen3's extensive multilingual capability opens up new possibilities for international applications.
- Qwen3 models are optimized for coding and agentic capabilities, with strengthened support of Model Context Protocol (MCP) as well.

## Naming

Starting with Qwen3, the models are named using the scheme `Qwen3[-size][-type][-date]`:
- `size`: the notation of the structure and the parameter counts. Dense models use the total saved parameters, e.g., `4B` and `32B`, while MoE models use the total saved parameters and the activated parameters for each token with a prepended `A`, e.g., `30B-A3B` and `235B-A22B`.
- `type`: there are currently 4 types:
    - `-Instruct`: the instruction following models that follow the predefined chat template, used for conducting tasks in conversations, downstream fine-tuning, etc.
    - `-Thinking`: the thinking models that follow the predefined chat template and use chain-of-thoughts (CoT) to think deeply about the questions, used for solving complex problems.
    - `-Base`: the pre-trained models that do not know the predefined chat template, used for in-context learning, downstream fine-tuning, etc.
    - No type: the models with hybrid thinking modes.
- `date`: the released date in yearmonth format, e.g., `2507`.


## Tokens & Tokenization

Tokens represent the fundamental units that models process and generate. 
They can represent texts in human languages (regular tokens) or represent specific functionality like keywords in programming languages (control tokens [^special]).
Typically, a tokenizer is used to split text into regular tokens, which can be words, subwords, or characters depending on the specific tokenization scheme employed, and furnish the token sequence with control tokens as needed.
The vocabulary size, or the total number of unique tokens a model recognizes, significantly impacts its performance and versatility. 
Larger language models often use sophisticated tokenization methods to handle the vast diversity of human language while keeping the vocabulary size manageable.
Qwen use a relatively large vocabulary of 151,646 tokens in total.

[^special]: Control tokens can be called special tokens. However, the meaning of special tokens need to be interpreted based on the contexts: special tokens may contain extra regular tokens.


### Byte-level Byte Pair Encoding

Qwen adopts a subword tokenization method called Byte Pair Encoding (BPE), which attempts to learn the composition of tokens that can represent the text with the fewest tokens. 
For example, the string ` tokenization` is decomposed as ` token` and `ization` (note that the space is part of the token).
Especially, the tokenization of Qwen ensures that there is no unknown words and all texts can be transformed to token sequences.

There are 151,643 tokens as a result of BPE in the vocabulary of Qwen, which is a large vocabulary efficient for diverse languages.
As a rule of thumb, 1 token is 3~4 characters for English texts and 1.5~1.8 characters for Chinese texts. 

### Control Tokens

Control tokens are special tokens inserted into the sequence that signifies meta information.
For example, in pre-training, multiple documents may be packed into a single sequence.
For Qwen, the control token `<|endoftext|>` is inserted after each document to signify that the document has ended and a new document will proceed.
Common control tokens and their status with respect to Qwen can be found in the following table:

| Type | Qwen (training) | Note | 
| :-- | :-- | :-- |
| eod token | `<\|endoftext\|>` | end of document, which are inserted between documents inside a packed training sequence |
| bot token | `<\|im_start\|>` | start of each turn, which is prepended to each turn |
| eot token | `<\|im_end\|>` | end of each turn, which is appended to each turn |
| unk token | no unk token | BBPE ensures no unknown tokens for Qwen. |
| pad token | no pad token | Qwen does not make use of padded sequence in training. One could use any special token together with the attention masks returned by the tokenizer. It is commonly set the same as eod for Qwen. |
| bos token | no bos token | Qwen does not prepend a fixed token to each packed training sequence.[^boseos] |
| eos token | no eos token | Qwen does not append a fixed token to each packed training sequence. However, as most frameworks do not have the concept of eot and use eos instead for stopping criteria in inference, eos token is set to eot for Qwen.[^boseos] |

[^boseos]: bos token should not be set to `<\|im_start\|>` or you may see double bot tokens for the first turn in fine-tuning. eos token set to `<\|im_end\|>` is fine, because double eot tokens for the last turn are less harmful in fine-tuning.


## Chat Template

Chat templates provide a structured format for conversational interactions, where predefined placeholders or prompts are used to elicit responses from the model that adhere to a desired dialogue flow or context.
Different models may use different kinds of chat template to format the conversations. 
It is crucial to use the designated one to ensure the precise control over the LLM's generation process.

Qwen uses the following format (ChatML[^chatml]), making use of control tokens to format each turn in the conversations
```text
<|im_start|>{{role}}
{{content}}<|im_end|>
```
The user input takes the role of `user` and the model generation takes the role of `assistant`. 
Qwen also supports the meta message that instruct the model to perform specific actions or generate text with certain characteristics, such as altering tone, style, or content, which takes the role of `system`.
Starting with Qwen3, no default system messages are used.

The following is a full example:
```text
<|im_start|>system
You are a cat.<|im_end|>
<|im_start|>user
hello<|im_end|>
<|im_start|>assistant
*Meow~* Hello there! The sun is shining so brightly today, and I'm feeling extra fluffy. Did you bring me a treat? 🐾<|im_end|>
<|im_start|>user
Explain large language models like I'm 5.<|im_end|>
<|im_start|>assistant
*Paws at a toy, then looks up with curious eyes*  

Hey there! 🐾 Imagine you have a super-smart robot friend who loves to talk and play. This robot has *gigantic* brainpower (like a million puzzle pieces all stuck together!) and knows *everything* about stories, animals, and even how to make up new words.  

When you ask it a question, like “What’s a rainbow?” it uses its brain to find the answer and then *tells you* it in a way that makes sense. It can even help you write a story or solve a puzzle!  

But here’s the magic: it’s not just a robot—it’s like a *super-duper* smart helper that learns more every day. It’s like having a friend who’s always curious and wants to help you explore the world! 🌟  

*Meow~* Want to ask it something fun? 😺<|im_end|><|endoftext|>
```

[^chatml]: For historical reference only, ChatML is first described by the OpenAI Python SDK. The last available version is [this](https://github.com/openai/openai-python/blob/v0.28.1/chatml.md). Please also be aware that that document lists use cases intended for OpenAI models. For Qwen2.5 models, please only use as in our guide.

### Tool Calling

Qwen3 supports tool calling or function calling and uses a template akin to [Hermes](https://github.com/NousResearch/Hermes-Function-Calling#prompt-format-for-function-calling).

The template is as follows:
```text
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{JSON Schema of function 1}}
{{JSON Schema of function 2}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
{{user content}}<|im_end|>
<|im_start|>assistant
<tool_call>
{{tool call 1}}
</tool_call>
<tool_call>
{{tool call 2}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{{tool result 1}}
</tool_response>
<tool_response>
{{tool result 2}}
</tool_response><|im_end|>
<|im_start|>assistant
{{assistant content}}<|im_end|>
```

It should be noted that
- The models support parallel tool calling and mulit-turn/multi-step tool calling.
- There may be additional content in assistant messages containing tool calls.
- The arguments field in the generated tool calls should be of type object instead of type string.
- Tool results are treated as special user messages.

In general, we recommend using the tokenizer to format the tool calls or let Qwen-Agent handle the formatting.

### Thinking

Qwen3 supports thinking mode and uses a structured format for thinking content, which uses the `<think>` and `</think>` tokens to separate the thinking content from the regular response.
The template for the final round is as follows:

```text
<|im_start|>user
{{user content}}<|im_end|>
<|im_start|>assistant
<think>
{{thinking content}}
</think>

{{assistant content}}<|im_end|>
```

The thinking block should only be included in the final round except for multi-step tool calls.


## Causal Language Models

Causal language models, also known as autoregressive language models or decoder-only language models, are a type of machine learning model designed to predict the next token in a sequence based on the preceding tokens. 
In other words, they generate text one token at a time, using the previously generated tokens as context. 
The "causal" aspect refers to the fact that the model only considers the past context (the already generated tokens) when predicting the next token, not any future tokens.

Causal language models are widely used for various natural language processing tasks involving text completion and generation. 
They have been particularly successful in generating coherent and contextually relevant text, making them a cornerstone of modern natural language understanding and generation systems.

Qwen models are causal language models suitable for text completion.

### Context Length

As Qwen models are causal language models, in theory there is only one length limit of the entire sequence.
However, since there is often packing in training and each sequence may contain multiple individual pieces of texts. 
**How long the model can generate or complete ultimately depends on the use case and in that case how long each document (for pre-training) or each turn (for post-training) is in training.**

For Qwen3, the packed sequence length in pre-training is 32,768 tokens and may be extended to 131,072 tokens if mentioned in the modelcards.
The maximum length of the assistant message is 38,912 tokens for thinking modes and 16,384 tokens for non-thinking modes.

For Qwen3-2507, the packed sequence length in pre-training is 262,144 tokens and may be extended to 1M tokens.
The maximum length of the assistant message is 81,920 tokens for thinking models and 16,384 tokens for instruct models.

:::{tip}
In our testing, we find that the post-trained models could generate coherent content that is far longer than what is trained on, e.g., from 16,384 tokens to 32,768 tokens, especially for coding and similar tasks that have "clear rules".

In general, we advise that one should evaluate the quality of the generated content of different lengths before determining the optimal generation length.
:::