---
myst:
  number_code_blocks: ["python3"]
---

# Function Calling

## Preface

Function calling with large language models is a huge and evolving topic.
It is particularly important for AI applications: 
- either for AI-native applications that strive to work around the shortcomings of current AI technology, 
- or for existing applications that seeks the integration of AI technology to improve performance, user interaction and experience, or efficiency.

This guide will not delve into those discussions or which role an LLM should play in an application and the related best practice.
Those views are reflected in the design of AI application frameworks: from LangChain to LlamaIndex to QwenAgent.

Instead, we will talk about how Qwen2.5 can be used to support function calling and how it can be used to achieve your goals, from the inference usage for developing application to the inner workings for hardcore customizations. 
In this guide, 
- We will first demonstrate how to use function calling with Qwen2.5.
- Then, we will introduce the technical details on functional calling with Qwen2.5, which are mainly about the templates.

Before starting, there is one thing we have not yet introduced, that is ...

## What is function calling?

:::{Note}
There is another term "tool use" that may be used to refer to the same concept.
While some may argue that tools are a generalized form of functions, at present, their difference exists only technically as different I/O types of programming interfaces.
:::

Large language models (LLMs) are powerful things.
However, sometimes LLMs by themselves are simply not capable enough.
- On the one hand, LLMs have inherent modeling limitations. 
  For one, they do not know things that are not in their training data, which include those happened after their training ended.
  In addition, they learn things in the way of likelihood, which suggests that they may not be precise enough for tasks with fixed rule sets, e.g., mathematical computation.
- On the other hand, it is not easy to use LLMs as a Plug-and-Play service programmatically with other things.
  LLMs mostly talk in words that are open to interpretation and thus ambiguous, while other software or applications or systems talk in code and through programming interfaces that are pre-defined and fixed and structured.

To this end, function calling establishes a common protocol that specifies how LLMs should interact with the other things.
The procedure is mainly as follows:
1. The application provides a set of functions and the instructions of the functions to an LLM.
2. The LLM choose to or not to, or is forced to use one or many of the functions, in response to user queries.
3. If the LLM chooses to use the functions, it states how the functions should be used based on the function instructions.
4. The chosen functions are used as such by the application and the results are obtained, which are then given to the LLM if further interaction is needed.

They are many ways for LLMs to understand and follow this protocol.
As always, the key is prompt engineering or an internalized template known by the model.
Qwen2.5 were pre-trained with various types of templates that could support function calling, so that users can directly make use of this procedure.


## Inference with Function Calling

:::{note}
Please be aware that the inference usage is subject to change as the frameworks and the Qwen models evolve.
:::

As function calling is essentially implemented using prompt engineering, you could manually construct the model inputs for Qwen2 models.
However, frameworks with function calling support can help you with all that laborious work.

In the following, we will introduce the usage (via dedicated function calling chat template) with
- **Qwen-Agent**,
- **Hugging Face transformers**,
- **Ollama**, and
- **vLLM**.

If you are familiar with the usage of OpenAI API, you could also directly use the OpenAI-compatible API services for Qwen2.5.
However, not all of them support function calling for Qwen2.5.
Currently, supported solutions include the self-hosted service by [Ollama](https://github.com/ollama/ollama/blob/main/docs/openai.md) or [vLLM](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#tool-calling-in-the-chat-completion-api) and the cloud service of [ModelStudio \[zh\]](https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope#97e2b45391x08).

If you are familiar with application frameworks, e.g., LangChain, you can also use function calling abilities in Qwen2.5 via ReAct Prompting.

### The Example Case

Let's also use an example to demonstrate the inference usage.
We assume **Python 3.11** is used as the programming language.

**Scenario**: Suppose we would like to ask the model about the temperature of a location.
Normally, the model would reply that it cannot provide real-time information.
But we have two tools that can be used to obtain the current temperature of and the temperature at a given date of a city respectively, and we would like the model to make use of them.

To set up the example case, you can use the following code:

:::{dropdown} Preparation Code
:name: prepcode

```python
import json

def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }


def get_function_by_name(name):
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]
MESSAGES = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow?"},
]
```
:::

In particular, the tools should be described using JSON Schema and the messages should contain as much available information as possible.
You can find the explanations of the tools and messages below:

:::{dropdown} Example Tools

The tools should be described using the following JSON:
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_current_temperature",
      "description": "Get current temperature at a location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The location to get the temperature for, in the format \"City, State, Country\"."
          },
          "unit": {
            "type": "string",
            "enum": [
              "celsius",
              "fahrenheit"
            ],
            "description": "The unit to return the temperature in. Defaults to \"celsius\"."
          }
        },
        "required": [
          "location"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_temperature_date",
      "description": "Get temperature at a location and date.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The location to get the temperature for, in the format \"City, State, Country\"."
          },
          "date": {
            "type": "string",
            "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."
          },
          "unit": {
            "type": "string",
            "enum": [
              "celsius",
              "fahrenheit"
            ],
            "description": "The unit to return the temperature in. Defaults to \"celsius\"."
          }
        },
        "required": [
          "location",
          "date"
        ]
      }
    }
  }
]
```
For each **tool**, it is a JSON object with two fields:
- `type`: a string specifying the type of the tool, currently only `"function"` is valid
- `function`: an object detailing the instructions to use the function

For each **function**, it is a JSON object with three fields:
- `name`: a string indicating the name of the function
- `description`: a string describing what the function is used for
- `parameters`: [a JSON Schema](https://json-schema.org/learn/getting-started-step-by-step) that specifies the parameters the function accepts. Please refer to the linked documentation for how to compose a JSON Schema. Notable fields include `type`, `required`, and `enum`.

Most frameworks use the tool format and some may use the function format.
Which one to use should be obvious according to the naming.
:::


:::{dropdown} Example Messages

Our query is `What's the temperature in San Francisco now? How about tomorrow?`.
Since the model does not know what the current date is, let alone tomorrow, we should provide the date in the inputs.
Here, we decide to supply that information in the system message after the default system message `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`.
You could append the date to user message in your application code.

```json
[
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow?"}
]
```
:::

### Qwen-Agent

[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) is actually a Python Agent framework for developing AI applications.
Although its intended use cases are higher-level than efficient inference, it does contain the **canonical implementation** of function calling for Qwen2.5.
It provides the function calling ability for Qwen2.5 to an OpenAI-compatible API through templates that is transparent to users.

{#note-official-template}
It's worth noting that since a lot of stuff can be done under the scene with application frameworks, currently the official function calling implementation for Qwen2.5 is very flexible and beyond simple templating, making it hard to adapt it other frameworks that use less capable templating engines.

Before starting, let's make sure the latest library is installed:
```bash
pip install -U qwen-agent
```

For this guide, we are at version v0.0.10.

#### Preparing

Qwen-Agent can wrap an OpenAI-compatible API that does not support function calling.
You can serve such an API with most inference frameworks or obtain one from cloud providers like DashScope or Together.

Assuming there is an OpenAI-compatible API at `http://localhost:8000/v1`, Qwen-Agent provides a shortcut function `get_chat_model` to obtain a model inference class with function calling support: 

```python
from qwen_agent.llm import get_chat_model

llm = get_chat_model({
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "model_server": "http://localhost:8000/v1",
    "api_key": "EMPTY",
})
```

In the above, `model_server` is the `api_base` common used in other OpenAI-compatible API clients.
It is advised to provide the `api_key` (but not via plaintext in the code), even if the API server does not check it, in which case, you can set it to anything.

For model inputs, the common message structure for system, user, and assistant history should be used:

```python
messages = MESSAGES[:]
# [
#    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
#    {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"}
# ]
```

We add the current date to the system message so that the "tomorrow" in the user message is anchored.
It can also be added to the user message if one desires.

At the time, Qwen-Agent works with functions instead of tools.
This requires a small change to our tool descriptions, that is, extracting the function fields:

```python
functions = [tool["function"] for tool in TOOLS]
```

#### Tool Calls and Tool Results

To interact with the model, the `chat` method should be used:

```python
for responses in llm.chat(
    messages=messages,
    functions=functions,
    extra_generate_cfg=dict(parallel_function_calls=True),
):
    pass
messages.extend(responses)
```

In the above code, the `chat` method receives the `messages`, the `functions`, and an `extra_generate_cfg` parameter.
You can put sampling parameters, such as `temperature`, and `top_p`, in the `extra_generate_cfg`.
Here, we add to it a special control `parallel_function_calls` provided by Qwen-Agent.
As its name suggests, it will enable parallel function calls, which means that the model may generate multiple function calls for a single turn as it deems fit.

The `chat` method returns a generator of list, each of which may contain multiple messages.
Since we enable `parallel_function_calls`, we should get two messages in the responses:

```python
[
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_temperature', 'arguments': '{"location": "San Francisco, CA, USA", "unit": "celsius"}'}},
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_temperature_date', 'arguments': '{"location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'}},
]
```

As we can see, Qwen-Agent attempts to parse the model generation in an easier to use structural format.
The details related to function calls are placed in the `function_call` field of the messages:
- `name`: a string representing the function to call
- `arguments`: a JSON-formatted string representing the arguments the function should be called with

Note that Qwen2.5-7B-Instruct is quite capable:
- It has followed the function instructions to add the state and the country to the location.
- It has correctly induced the date of tomorrow and given in the format required by the function.

Then comes the critical part -- checking and applying the function call:
```python3
for message in responses:
    if fn_call := message.get("function_call", None):
        fn_name: str = fn_call['name']
        fn_args: dict = json.loads(fn_call["arguments"])

        fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

        messages.append({
            "role": "function",
            "name": fn_name,
            "content": fn_res,
        })
```

To get tool results: 
- line 1: We should iterate the function calls in the order the model generates them.
- line 2: We can check if a function call is needed as deemed by the model by checking the `function_call` field of the generated messages.
- line 3-4: The related details including the name and the arguments of the function can also be found there, which are `name` and `arguments` respectively.
- line 6: With the details, one should call the function and obtain the results.
  Here, we assume there is a function named [`get_function_by_name`](#prepcode) to help us get the related function by its name.
- line 8-12: With the result obtained, add the function result to the messages as `content` and with `role` as `"function"`.

Now the messages are
```python
[
    {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_temperature', 'arguments': '{"location": "San Francisco, CA, USA", "unit": "celsius"}'}},
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_temperature_date', 'arguments': '{"location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'}},
    {'role': 'function', 'name': 'get_current_temperature', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}'},
    {'role': 'function', 'name': 'get_temperature_date', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'},
]
```

#### Final Response

Finally, run the model again to get the final model results:

```python
for responses in llm.chat(messages=messages, functions=functions):
    pass
messages.extend(responses)
```

The final response should be like

```python
{'role': 'assistant', 'content': 'Currently, the temperature in San Francisco is approximately 26.1°C. Tomorrow, on 2024-10-01, the temperature is forecasted to be around 25.9°C.'}
```

### Hugging Face transformers

Since function calling is based on prompt engineering and templates, `transformers` supports it with its tokenizer utilities, in particular, the `tokenizer.apply_chat_template` method, which hides the sophistication of constructing the model inputs, using the Jinja templating engine.
However, it means that users should handle the model output part on their own, which includes parsing the generated function call message.

The blog piece [_Tool Use, Unified_](https://huggingface.co/blog/unified-tool-use) is very helpful in understanding its design.
Be sure to take a look.

Tool use API is available in transformers since v4.42.0.
Before starting, let's check that:
```bash
pip install "transformers>4.42.0"
```

For this guide, we are at version v4.44.2.

#### Preparing

For Qwen2.5, the chat template in `tokenizer_config.json` has already included support for the Hermes-style tool use. 
We simply need to load the model and the tokenizer:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
)
```

The inputs are the same with those in [the preparation code](#prepcode):

```python
tools = TOOLS
messages = MESSAGES[:]
```

In `transformers`, you can also directly use Python functions as tools with certain constraints[^get_json_schema_note]:

```python
tools = [get_current_temperature, get_temperature_date]
```

[^get_json_schema_note]: `transformers` will use `transformers.utils.get_json_schema` to generate the tool descriptions from Python functions.
    There are some gotchas with `get_json_schema`, and it is advised to check [its doc \[v4.44.2\]](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/utils/chat_template_utils.py#L183-L288) before relying on it. 

    - The function should use Python type hints for parameter types and has a Google-style docstring for function description and parameter descriptions.
    - Supported types are limited, since the types needs to be mapped to JSON Schema.
      In particular, `typing.Literal` is not supported.
      You can instead add `(choices: ...)` at the end of a parameter description, which will be mapped to a `enum` type in JSON Schema.
  
    Please be aware that all the returned results in the examples in the linked docstring are actually the content of the `function` field in the actual returned results.

#### Tool Calls and Tool Results

To construct the input sequence, we should use the `apply_chat_template` method and then let the model continue the texts:

```python
text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
output_text = tokenizer.batch_decode(outputs)[0][len(text):]
```

The output texts should be like
```text
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}
</tool_call>
<tool_call>
{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}
</tool_call><|im_end|>
```

Now we need to do two things: 
1. Parse the generated tool calls to a message and add them to the messages, so that the model knows which tools are used.
2. Obtain the results of the tools and add them to the messages, so that the model knows the results of the tool calls.

In `transformers`, the tool calls should be a field of assistant messages.
Let's use a simple function called `try_parse_tool_calls` to parse the tool calls:

{#parse-function}
```python
import re

def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else: 
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}
```


This function does not cover all possible scenarios and thus is prone to errors.
But it should suffice for the purpose of this guide. 

:::{note}
The template in the `tokenizer_config.json` assumes that the generated content alongside tool calls is in the same message instead of separate assistant messages, e.g.,
```json
{
  "role": "assistant", 
  "content": "To obtain the current temperature, I should call the functions `get_current_temperate`.", 
  "tool_calls": [
    {"type": "function", "function": {"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA", "unit": "celsius"}}}
  ]
}
```
instead of 
```json
[
  {
    "role": "assistant", 
    "content": "To obtain the current temperature, I should call the functions `get_current_temperate`.", 
  },
  {
    "role": "assistant", 
    "content": "", 
    "tool_calls": [
      {"type": "function", "function": {"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA", "unit": "celsius"}}}
    ]
  }
]
```

This is implemented roughly in `try_parse_tool_calls` but keep that in mind if you are writing your own tool call parser.
:::

```python
messages.append(try_parse_tool_calls(output_text))

if tool_calls := messages[-1].get("tool_calls", None):
    for tool_call in tool_calls:
        if fn_call := tool_call.get("function"):
            fn_name: str = fn_call["name"]
            fn_args: dict = fn_call["arguments"]

            fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": fn_res,
            })
```

The messages now should be like
```python
[
    {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'get_current_temperature', 'arguments': {'location': 'San Francisco, CA, USA'}}}, 
        {'type': 'function', 'function': {'name': 'get_temperature_date', 'arguments': {'location': 'San Francisco, CA, USA', 'date': '2024-10-01'}}},
    ]},
    {'role': 'tool', 'name': 'get_current_temperature', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}'},
    {'role': 'tool', 'name': 'get_temperature_date', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'},
]
```

The messages are similar to those of Qwen-Agent, but there are some major differences:
- Tools instead of functions
- Parallel calls are by default
  - Multiple tool calls as a list in a single assistant message, instead of multiple messages.
  - The function arguments are parsed into a dict if it is a valid JSON-formatted string.

#### Final Response

Then it's time for the model to generate the actual response for us based on the tool results. 
Let's query the model again:

```python
text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
output_text = tokenizer.batch_decode(outputs)[0][len(text):]
```

The output_text should be like
```
The current temperature in San Francisco is approximately 26.1°C. Tomorrow, on October 1, 2024, the temperature is expected to be around 25.9°C.<|im_end|>
```

Add the result text as an assistant message and the final messages should be ready for further interaction:
```python
messages.append(try_parse_tool_calls(output_text))
```

### Ollama

Ollama is a set of tools for serving LLMs locally. 
It also relies on its template implementation to support function calling.
Different from transformers, which is written in Python and uses the Jinja template whose syntax is heavily inspired by Django and Python, Ollama, which is mostly written in Go, uses Go's [text/template](https://pkg.go.dev/text/template) packages.
In addition, Ollama implements internally a helper function so that it can automatically parse the generated tool calls in texts to structured messages if the format supported.

You could check the [Tool support](https://ollama.com/blog/tool-support) blog post first.

Tool support has been available in Ollama since v0.3.0.
You can run the following to check the Ollama version:
```bash
ollama -v
```
If lower than expected, follow [the official instructions](https://ollama.com/download) to install the latest version.

In this guide, we will aslo use [ollama-python](https://github.com/ollama/ollama-python), before starting, make sure it is available in your environment:
```bash
pip install ollama
```

For this guide, the `ollama` binary is at v0.3.9 and the `ollama` Python library is at v0.3.2.


#### Preparing

The messages structure used in Ollama is the same with that in `transformers` and the template in [Qwen2.5 Ollama models](https://ollama.com/library/qwen2.5) has supported tool use. 

The inputs are the same with those in [the preparation code](#prepcode):
```python
tools = TOOLS
messages = MESSAGES[:]

model_name = "qwen2.5:7b"
```
Note that you cannot pass Python functions as tools directly and `tools` has to be a `dict`.


#### Tool Calls and Tool Results

We can use the `ollama.chat` method to directly query the underlying API:

```python
import ollama

response = ollama.chat(
    model=model_name,
    messages=messages,
    tools=tools,
)
```

The main fields in the response could be:
```python
{
    'model': 'qwen2.5:7b',
    'message': {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {'function': {'name': 'get_current_temperature', 'arguments': {'location': 'San Francisco, CA, USA'}}},
            {'function': {'name': 'get_temperature_date', 'arguments': {'date': '2024-10-01', 'location': 'San Francisco, CA, USA'}}},
        ],
    },
}
```

Ollama's tool call parser has succeeded in parsing the tool results.
If not, you may refine [the `try_parse_tool_calls` function above](#parse-function).
Then, we can obtain the tool results and add them to the messages.
The following is basically the same with `transformers`:

```python
messages.append(response["message"])

if tool_calls := messages[-1].get("tool_calls", None):
    for tool_call in tool_calls:
        if fn_call := tool_call.get("function"):
            fn_name: str = fn_call["name"]
            fn_args: dict = fn_call["arguments"]

            fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": fn_res,
            })
```

The messages are now like
```python
[
    {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'function': {'name': 'get_current_temperature', 'arguments': {'location': 'San Francisco, CA, USA'}}},
        {'function': {'name': 'get_temperature_date', 'arguments': {'date': '2024-10-01', 'location': 'San Francisco, CA, USA'}}},
    ]},
    {'role': 'tool', 'name': 'get_current_temperature', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}'},
    {'role': 'tool', 'name': 'get_temperature_date', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'},
]
```

#### Final Response

The rest are easy:

```python
response = ollama.chat(
    model=model_name,
    messages=messages,
    tools=tools,
)
messages.append(response["message"])
```

The final message should be like the following:
```python
{'role': 'assistant', 'content': 'The current temperature in San Francisco is approximately 26.1°C. For tomorrow, October 1st, 2024, the forecasted temperature will be around 25.9°C.'}
```

(heading-target)=
### vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving.
It uses the tokenizer from `transformers` to format the input, so we should have no trouble preparing the input.
In addition, vLLm also implements helper functions so that generated tool calls can be parsed automatically if the format is supported.

Tool support has been available in `vllm` since v0.6.0. 
Be sure to install a version that supports tool use.
For more information, check the [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#tool-calling-in-the-chat-completion-api).

For this guide, we are at version v0.6.1.post2.
We will use the OpenAI-Compatible API by `vllm` with the API client from the `openai` Python library.

#### Preparing

For Qwen2.5, the chat template in tokenizer_config.json has already included support for the Hermes-style tool use.
We simply need to start a OpenAI-compatible API with vLLM:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```

The inputs are the same with those in [the preparation code](#prepcode):

```python
tools = TOOLS
messages = MESSAGES[:]
```

Let's also initialize the client:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
```

#### Tool Calls and Tool Results

We can use the create chat completions endpoint to query the model:

```python
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
```

vLLM should be able to parse the tool calls for us, and the main fields in the response (`response.choices[0]`) should be like
```python
Choice(
    finish_reason='tool_calls', 
    index=0, 
    logprobs=None, 
    message=ChatCompletionMessage(
        content=None, 
        role='assistant', 
        function_call=None, 
        tool_calls=[
            ChatCompletionMessageToolCall(
                id='chatcmpl-tool-924d705adb044ff88e0ef3afdd155f15', 
                function=Function(arguments='{"location": "San Francisco, CA, USA"}', name='get_current_temperature'), 
                type='function',
            ), 
            ChatCompletionMessageToolCall(
                id='chatcmpl-tool-7e30313081944b11b6e5ebfd02e8e501', 
                function=Function(arguments='{"location": "San Francisco, CA, USA", "date": "2024-10-01"}', name='get_temperature_date'), 
                type='function',
            ),
        ],
    ), 
    stop_reason=None,
)
```

Note that the function arguments are JSON-formatted strings, which Qwen-Agent follows but `transformers` and Ollama differs.

As before, chances are that there are corner cases where tool calls are generated but they are malformed and cannot be parsed.
For production code, we should try parsing by ourselves.

Then, we can obtain the tool results and add them to the messages as shown below:

```python
messages.append(response.choices[0].message.model_dump())

if tool_calls := messages[-1].get("tool_calls", None):
    for tool_call in tool_calls:
        call_id: str = tool_call["id"]
        if fn_call := tool_call.get("function"):
            fn_name: str = fn_call["name"]
            fn_args: dict = json.loads(fn_call["arguments"])
        
            fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

            messages.append({
                "role": "tool",
                "content": fn_res,
                "tool_call_id": call_id,
            })
```

It should be noted that the OpenAI API uses `tool_call_id` to identify the relation between tool results and tool calls.

The messages are now like
```python
[
    {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'content': None, 'role': 'assistant', 'function_call': None, 'tool_calls': [
        {'id': 'chatcmpl-tool-924d705adb044ff88e0ef3afdd155f15', 'function': {'arguments': '{"location": "San Francisco, CA, USA"}', 'name': 'get_current_temperature'}, 'type': 'function'},
        {'id': 'chatcmpl-tool-7e30313081944b11b6e5ebfd02e8e501', 'function': {'arguments': '{"location": "San Francisco, CA, USA", "date": "2024-10-01"}', 'name': 'get_temperature_date'}, 'type': 'function'},
    ]},
    {'role': 'tool', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}', 'tool_call_id': 'chatcmpl-tool-924d705adb044ff88e0ef3afdd155f15'},
    {'role': 'tool', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}', 'tool_call_id': 'chatcmpl-tool-7e30313081944b11b6e5ebfd02e8e501'},
]
```

#### Final Response

Let's call the endpoint again to seed the tool results and get response:
```python
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)

messages.append(response.choices[0].message.model_dump())
```

The final response (`response.choices[0].message.content`) should be like
```text
The current temperature in San Francisco is approximately 26.1°C. For tomorrow, the forecasted temperature is around 25.9°C.
```

### Discussions

Now, we have introduced how to conduct inference with function calling using Qwen2 in three different frameworks!
Let's make a brief comparison.

| Item | OpenAI API | Hugging Face transformers | Ollama | vLLM | Qwen-Agent |
| :-----  | :---: | :---: | :---: | :---: | :---: | 
| Type | HTTP API | Python Library | HTTP API | HTTP API | Python Library |
| Inference Backend | - | PyTorch | llama.cpp | PyTorch | HTTP API |
| Templating Backend | - | Jinja | Go `text/template` | Jinja | Python |
| Tools/Functions | Tools | Tools | Tools | Tools | Functions |
| Parallel Calls | Default Yes (Configurable) | Yes | Yes | Yes | Default No (Configurable) |
| Call Format | Single assistant message with `tool_calls` | Single assistant message with `tool_calls` | Single assistant message with `tool_calls`  | Single assistant message with `tool_calls` | Multiple assistant messages with `function_call` |
| Call Argument Format | string | object | object | string | string |  
| Call Result Format | Multiple tool messages with `content` | Multiple tool messages with `content` | Multiple tool messages with `content` | Multiple tool messages with `content` | Multiple function messages with `content` |


There are some details not shown in the above table:
- OpenAI API comes with Python, Node.js, Go, and .NET SDKs. It also follows the OpenAPI standard.
- Ollama comes with Python and Node.js SDKs. It has OpenAI-compatible API at a different base url that can be accessed using OpenAI API SDK.
- Qwen-Agent as an application framework can call the tools automatically for you, which is introduced in [the Qwen-Agent guide](./qwen_agent).


In addition, there are more on the model side of function calling, which means you may need to consider more things in production code:
- **Accuracy of function calling**:
  When it comes to evaluate the accuracy of function calling, there are two aspects:
  (a) whether the correct functions (including no functions) are selected and
  (b) whether the correct function arguments are generated.
  It is not always the case that Qwen2.5 will be accurate. 
  Function calling can involve knowledge that is deep and domain-specific.
  Sometimes, it doesn't fully understand the function and select the wrong one by mistake.
  Sometimes, it can fall into a loop and require calling the same function again and again. 
  Sometimes, it will fabricate required function arguments instead of asking the user for input.
  To improve the function calling accuracy, it is advised to first try prompt engineering:
  does a more detailed function description help?
  can we provide instructions and examples to the model in the system message?
  If not, finetuning on your own data could also improve performance.
- **Protocol consistency**:
  Even with the proper function calling template, the protocol may break.
  The model may generate extra texts to tool calls, e.g., explanations.
  The generated tool call may be invalid JSON-formatted string but a representation of a Python dict
  The generated tool call may be valid JSON but not conforms to the provided JSON Schema.
  For those kinds of issues, while some of them could be addressed with prompt engineering, some are caused by the nature of LLMs and can be hard to resolve in a general manner by LLMs themselves.
  While we strive to improve Qwen2.5 in this regard, edge cases are unlikely to be eliminated completely.


## Function Calling Templates

The template design for function calling often includes the following aspects:
- How to describe the functions to the model, so that the model understands what they are and how to use them.
- How to prompt the model, so that it knows that functions can be used and in what format to generate the function calls.
- How to tell a function call generation from others in generated text, so that we can extract the calls from the generated texts and actually make the calls.
- How to incorporate the function results to the text, so that the model can tell them from its own generation and make connection among the calls and the results.

For experienced prompt engineers, it should be possible to make any LLM support function calling, using in-context learning techniques and with representative examples, though with varied accuracy and stability depending on how "zero-shot" the task at hand is.

### Starting from ReAct Prompting

For example, ReAct Prompting can be used to implement function calling with an extra element of planning: 
- **Thought**: the overt reasoning path, analyzing the functions and the user query and saying it out "loud"
- **Action**: the function to use and the arguments with which the function should be called
- **Observation**: the results of the function

In fact, Qwen2 is verse in the following variant of ReAct Prompting (similar to LangChain ReAct) to make the intermediate texts more structured:

```
Answer the following questions as best you can. You have access to the following tools:

{function_name}: Call this tool to interact with the {function_name_human_readable} API. What is the {function_name_human_readable} API useful for? {function_desciption} Parameters: {function_parameter_descriptions} {argument_formatting_instructions}

{function_name}: Call this tool to interact with the {function_name_human_readable} API. What is the {function_name_human_readable} API useful for? {function_desciption} Parameters: {function_parameter_descriptions} {argument_formatting_instructions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{function_name},{function_name}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
Thought: {some_text}
Action: {function_name}
Action Input: {function_arguments}
Observation: {function_results}
Final Answer: {response}
```

As you can see, there is no apparent user/assistant conversation structure in the template.
The model will simply continue the texts.
One should write the code to actively detect which step the model is at and in particular to add the observations in the process, until the Final Answer is generated.

However, as most programming interfaces accept the message structure, there should be some kind of adapter between the two.
[The ReAct Chat Agent](https://github.com/QwenLM/Qwen-Agent/blob/v0.0.10/qwen_agent/agents/react_chat.py) in Qwen-Agent facilitates this kind of conversion.

### Qwen2 Function Calling Template

As a step forward, the official Qwen2 function calling template is in the vein of the ReAct Prompting format but focuses more on
- differentiating the keywords like `Question`, `Thought`, `Action`, etc., from generation,
- simplifying the process,
- supporting better multi-turn conversation, and
- adding controls for specialized usage.


An equivalent example would be
```
<|im_start|>system
You are a helpful assistant.

## Tools

You have access to the following tools:

### {function_name_human_readable}

{function_name}: {function_description} Parameters: {function_parameter_descriptions} {argument_formatting_instructions}

### {function_name_human_readable}

{function_name}: {function_description} Parameters: {function_parameter_descriptions} {argument_formatting_instructions}

## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

✿FUNCTION✿: The tool to use, should be one of [{function_name},{function_name}]
✿ARGS✿: The input of the tool
✿RESULT✿: Tool results
✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
✿FUNCTION✿: {function_name}
✿ARGS✿: {function_arguments}
✿RESULT✿: {function_result}
✿RETURN✿:{response}<|im_end|>
```

Let's first list the obvious differences:
- Keywords (`✿FUNCTION✿`, `✿ARGS✿`, etc.) seem rare in ordinary text and more semantically related to function calling, but not special tokens yet.
- Thought is omitted. This could affect accuracy for some use cases.
- Use the system-user-assistant format for multi-turn conversations. Function calling prompting is moved to the system message.

How about adding controls for specialized usage?
The template actually has the following variants:
- Language: the above is for non-Chinese language; there is another template in Chinese.
- Parallel Calls: the above is for non-parallel calls; there is another template for parallel calls.

In the canonical implementation in Qwen-Agent, those switches are implemented in Python, according to the configuration and current input.

The actual text with parallel calls should be like the following:

```text
<|im_start|>system
You are a helpful assistant.

Current Date: 2024-08-31

## Tools

You have access to the following tools:

### get_current_temperature

get_current_temperature: Get current temperature at a location. Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location"]} Format the arguments as a JSON object.

### get_temperature_date

get_temperature_date: Get temperature at a location and date. Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location", "date"]} Format the arguments as a JSON object.

## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

✿FUNCTION✿: The tool to use, should be one of [get_current_temperature,get_temperature_date]
✿ARGS✿: The input of the tool
✿RESULT✿: Tool results
✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)<|im_end|>
<|im_start|>user
What's the temperature in San Francisco now? How about tomorrow?<|im_end|>
<|im_start|>assistant
✿FUNCTION✿: get_current_temperature
✿ARGS✿: {"location": "San Francisco, CA, USA", "unit": "celsius"}
✿FUNCTION✿: get_temperature_date
✿ARGS✿: {"date": "2024-09-01", "location": "San Francisco, CA, USA", "unit": "celsius"}
✿RESULT✿: {"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}
✿RESULT✿: {"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-09-01", "unit": "celsius"}
✿RETURN✿: The current temperature in San Francisco is 26.1°C. The temperature for tomorrow in San Francisco is expected to be 25.9°C.<|im_end|>
```


This template is hard to adapt it for other frameworks that use less capable templating engines.
But it is doable at least partially for Jinja, which is Python-oriented after all.
We didn't use it because using the template in `transformers` leads to more changes to the inference usage, which are not very common for beginners.

For the interested, you can find the Jinja template and key points on usage below:

:::{dropdown} Qwen2 Function Calling Jinja Template

```jinja
{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = "You are a helpful assistant." %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- if parallel_tool_calls is undefined %}
    {%- set parallel_tool_calls = false %}
{%- endif %}
{%- if language is undefined or language != "zh" %}
    {%- set language = "en" %}
{%- endif %}

{{- "<|im_start|>system\n" + system_message|trim }}
{%- if tools is defined %}
    {{- "\n\n# 工具\n\n## 你拥有如下工具：\n\n" if language == "zh" else "\n\n## Tools\n\nYou have access to the following tools:\n\n" }}
    {%- set functions = tools|map(attribute="function")|list %}
    {%- set function_names = functions|map(attribute="name")|join(",") %}
    {%- for function in functions %}
        {{- "### " + function.name + "\n\n" + function.name + ": " + function.description + (" 输入参数：" if language == "zh" else " Parameters: ") + function.parameters|tojson + (" 此工具的输入应为JSON对象。\n\n" if language == "zh" else " Format the arguments as a JSON object.\n\n") }}
    {%- endfor %}
    {%- if parallel_tool_calls and language == "zh" %}
        {{- "## 你可以在回复中插入以下命令以并行调用N个工具：\n\n✿FUNCTION✿: 工具1的名称，必须是[" + function_names + "]之一\n✿ARGS✿: 工具1的输入\n✿FUNCTION✿: 工具2的名称\n✿ARGS✿: 工具2的输入\n...\n✿FUNCTION✿: 工具N的名称\n✿ARGS✿: 工具N的输入\n✿RESULT✿: 工具1的结果\n✿RESULT✿: 工具2的结果\n...\n✿RESULT✿: 工具N的结果\n✿RETURN✿: 根据工具结 果进行回复，需将图片用![](url)渲染出来" }}
    {%- elif parallel_tool_calls %}
        {{- "## Insert the following command in your reply when you need to call N tools in parallel:\n\n✿FUNCTION✿: The name of tool 1, should be one of [" + function_names + "]\n✿ARGS✿: The input of tool 1\n✿FUNCTION✿: The name of tool 2\n✿ARGS✿: The input of tool 2\n...\n✿FUNCTION✿: The name of tool N\n✿ARGS✿: The input of tool N\n✿RESULT✿: The result of tool 1\n✿RESULT✿: The result of tool 2\n...\n✿RESULT✿: The result of tool N\n✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)" }}
    {%- elif language == "zh" %}
        {{- "## 你可以在回复中插入零次、一次或多次以下命令以调用工具：\n\n✿FUNCTION✿: 工具名称，必须是[" + function_names + "]之一。\n✿ARGS✿: 工具输入\n✿RESULT✿: 工具结果\n✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来" }}
    {%- else %}
        {{- "## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:\n\n✿FUNCTION✿: The tool to use, should be one of [" + function_names + "]\n✿ARGS✿: The input of the tool\n✿RESULT✿: Tool results\n✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)" }}
    {%- endif %}
{%- endif %}
{{- "<|im_end|>" }}

{%- for message in loop_messages %}
    {%- if message.role == "user" %}
        {{- "\n<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>" }}
        {%- if loop.last and add_generation_prompt %}
            {{- "\n<|im_start|>assistant\n" }}
        {%- endif %}
    {%- elif message.role == "tool" %}
        {{- "✿RESULT✿: " + message.content + "\n" }}
        {%- if loop.last and add_generation_prompt %}
            {{- "✿RETURN✿:" }}
        {%- endif %}
    {%- elif message.role == "assistant" and message.tool_calls is defined %}
        {%- if loop.previtem.role == "user" %}
            {{- "\n<|im_start|>assistant\n" }}
        {%- endif %}
        {%- for function in message.tool_calls|map(attribute="function") %}
            {{- "✿FUNCTION✿: " + function.name + "\n✿ARGS✿: " + function.arguments|tojson + "\n" }}
        {%- endfor %}
    {%- elif message.role == "assistant" %}
        {%- if loop.previtem.role == "user" %}
            {{- "\n<|im_start|>assistant\n" }}
        {%- elif loop.previtem.role == "tool" %}
            {{- "✿RETURN✿:" }}
        {%- endif %}
        {{- message.content }}
        {%- if loop.nextitem is undefined or loop.nextitem.role == "user" %}
            {{- "<|im_end|>" }}
        {%- endif %}
    {%- else %}
        {{- "\n<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>" }}
    {%- endif %}
{%- endfor %}
```

To use this template in `transformers`:

- Switches can be enabled by passing them to the `apply_chat_template` method, e.g., `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, parallel_tool_call=True, language="zh", tokenize=False)`. By default, it is for English non-parallel function calling.

- Since the generation needs to be stopped at `✿RESULT✿` or else the model will generate fabricated tool results, we should add it to `stop_strings` in `generation_config`:
    ```python
    model.generation_config.stop_strings = ["✿RESULT✿:", "✿RETURN✿:"]
    ```

- As a result of using `stop_strings`, you need to pass the tokenizer to `model.generate` as `model.generate(**inputs, tokenizer=tokenizer, max_new_tokens=512)`.

- `response`, i.e., the model generation based on the tool calls and tool results, may contain a leading space. You should not strip it for the model. It is resulted from the tokenization and the template design.

- The `try_parse_tool_calls` function should also be modified accordingly.
:::


### Qwen2.5 Function Calling Templates

For `transformers` and Ollama, we have also used templates that are easier to implement with Jinja or Go.
They are variants of [the Nous Research's Hermes function calling template](https://github.com/NousResearch/Hermes-Function-Calling#prompt-format-for-function-calling).
The Jinja template and the Go template should produce basically the same results.
They final text should look like the following:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Current Date: 2024-09-30

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location"]}}}
{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location", "date"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's the temperature in San Francisco now? How about tomorrow?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}
</tool_call>
<tool_call>
{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}
</tool_response>
<tool_response>
{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}
</tool_response><|im_end|>
<|im_start|>assistant
The current temperature in San Francisco is approximately 26.1°C. Tomorrow, on October 1, 2024, the temperature is expected to be around 25.9°C.<|im_end|>
```

While the text may seem different from the previous one, the basic prompting structure is still the same.
There are just more structural tags and more JSON-formatted strings.

---

There is one thing we haven't talked about: how should functions be described to the LLMs.
In short, you could describe them as you would normally describe them in an API documentation, as long as you can effectively parse, validate, and execute the tool calls generated by the models.
The format with JSON Schema appears a valid and common choice.


## Finally

In whichever way you choose to use function calling with Qwen2.5, keep in mind that the limitation and the perks of prompt engineering applies:
- It is not guaranteed that the model generation will always follow the protocol even with proper prompting or templates.
  Especially, for the templates that are more complex and relies more on the model itself to think and stay on track than the ones that are simpler and relies on the template and the use of control or special tokens.
  The latter one, of course, requires some kind of training.
  In production code, be prepared that if it breaks, countermeasures or rectifications are in place.
- If in certain scenarios, the generation is not up to expectation, you can refine the template to add more instructions or constraints.
  While the templates mentioned here are general enough, they may not be the best or the most specific or the most concise for your use cases.
  The ultimate solution is fine-tuning using your own data.

Have fun prompting!