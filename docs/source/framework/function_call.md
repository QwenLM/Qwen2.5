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

We will talk about how Qwen3 can be used to support function calling and how it can be used to achieve your goals, from the inference usage for developing application to the inner workings for hardcore customizations. 
In this guide, 
- We will first demonstrate how to use function calling with Qwen3.
- Then, we will introduce the technical details on functional calling with Qwen3, which are mainly about the templates.

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

There are many ways for LLMs to understand and follow this protocol.
As always, the key is prompt engineering or an internalized template known by the model.
We recommend using Hermes-style tool use for Qwen3 to maximize function calling performance.

## Inference with Function Calling

As function calling is essentially implemented using prompt engineering, you could manually construct the model inputs for Qwen3 models.
However, frameworks with function calling support can help you with all that laborious work.

In the following, we will introduce the usage (via dedicated function calling chat template) with
- **Qwen-Agent**,
- **vLLM**.

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
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30."},
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

Our query is `What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30.`.

```json
[ 
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30."}
]
```
:::

### Qwen-Agent

[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) is actually a Python Agent framework for developing AI applications.
Although its intended use cases are higher-level than efficient inference, it does contain the **canonical implementation** of function calling for Qwen3.
It provides the function calling ability for Qwen3 to an OpenAI-compatible API through templates that is transparent to users.

It is worth noting that for reasoning models like Qwen3, it is *not recommended* to use tool call template based on stopwords, such as ReAct, because the model may output stopwords in the thought section, potentially leading to unexpected behavior in tool calls.

Before starting, let's make sure the latest library is installed:
```bash
pip install -U qwen-agent
```

#### Preparing

Qwen-Agent can wrap an OpenAI-compatible API that does not support function calling.
You can serve such an API with most inference frameworks or obtain one from cloud providers like DashScope or Together.

Assuming there is an OpenAI-compatible API at `http://localhost:8000/v1`, Qwen-Agent provides a shortcut function `get_chat_model` to obtain a model inference class with function calling support: 

```python
from qwen_agent.llm import get_chat_model

llm = get_chat_model({
    "model": "Qwen/Qwen3-8B",
    "model_server": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "generate_cfg": {
      "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False}  # default to True
      }
    }
})
```

In the above, `model_server` is the `api_base` common used in other OpenAI-compatible API clients.
It is advised to provide the `api_key` (but not via plaintext in the code), even if the API server does not check it, in which case, you can set it to anything.
You can pass model parameters to the model by `generate_cfg`. Here we demonstrate how to control the think and no_think modes of Qwen3. 
Different APIs may have different control methods.

For model inputs, the common message structure for system, user, and assistant history should be used:

```python
messages = MESSAGES[:]
```

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
):
    pass
messages.extend(responses)
```

The `chat` method returns a generator of list, each of which may contain multiple messages.


- The results of `no_think` mode:
```python
[
    {"role": "assistant", "content": "", "function_call": {"name": "get_current_temperature", "arguments": "{\"location\": \"San Francisco, California, United States\", \"unit\": \"celsius\"}"}},
    {"role": "assistant", "content": "", "function_call": {"name": "get_temperature_date", "arguments": "{\"location\": \"San Francisco, California, United States\", \"date\": \"2024-10-01\", \"unit\": \"celsius\"}"}},
]
```

- The results of `think` mode:
```python
[
    {"role": "assistant", "content": "", "reasoning_content": "Okay, the user is asking for the current temperature in San Francisco and the temperature for tomorrow. Let me check the available tools.\n\nFirst, there's the get_current_temperature function. It requires the location and optionally the unit. Since the user didn't specify the unit, I'll default to celsius. The location should be \"San Francisco, State, Country\". Wait, the example format is \"City, State, Country\", but San Francisco is a city in California, USA. So the location parameter would be \"San Francisco, California, United States\".\n\nThen, for tomorrow's temperature, the user mentioned the current date is 2024-09-30, so tomorrow would be 2024-10-01. The get_temperature_date function requires location, date, and unit. Again, using the same location and default unit. I need to format the date as \"Year-Month-Day\", which is 2024-10-01.\n\nWait, the current date given is 2024-09-30. If today is September 30, then tomorrow is October 1st. So the date parameter for the second function call should be \"2024-10-01\".\n\nI should make two separate function calls: one for the current temperature and another for tomorrow's date. Let me structure the JSON for both tool calls accordingly."},
    {"role": "assistant", "content": "", "function_call": {"name": "get_current_temperature", "arguments": "{\"location\": \"San Francisco, California, United States\", \"unit\": \"celsius\"}"}},
    {"role": "assistant", "content": "", "function_call": {"name": "get_temperature_date", "arguments": "{\"location\": \"San Francisco, California, United States\", \"date\": \"2024-10-01\", \"unit\": \"celsius\"}"}},
]
```

As we can see, Qwen-Agent attempts to parse the model generation in an easier to use structural format.
The details related to function calls are placed in the `function_call` field of the messages:
- `name`: a string representing the function to call
- `arguments`: a JSON-formatted string representing the arguments the function should be called with

In the thinking mode, it will first generate a thought and then generate the tool call(s).

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

Now the messages are:

- `no_think` mode:
```python
[
    {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30."},
    {"role": "assistant", "content": "", "function_call": {"name": "get_current_temperature", "arguments": "{\"location\": \"San Francisco, California, United States\", \"unit\": \"celsius\"}"}},
    {"role": "assistant", "content": "", "function_call": {"name": "get_temperature_date", "arguments": "{\"location\": \"San Francisco, California, United States\", \"date\": \"2024-10-01\", \"unit\": \"celsius\"}"}},
    {"role": "function", "name": "get_current_temperature", "content": '{"temperature": 26.1, "location": "San Francisco, California, United States", "unit": "celsius"}'},
    {"role": "function", "name": "get_temperature_date", "content": '{"temperature": 25.9, "location": "San Francisco, California, United States", "date": "2024-10-01", "unit": "celsius"}'},
]
```

- `think` mode:
```python
[
    {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30."},
    {"role": "assistant", "content": "", "reasoning_content": "Okay, the user is asking for the current temperature in San Francisco and the temperature for tomorrow. Let me check the available tools.\n\nFirst, there's the get_current_temperature function. It requires the location and optionally the unit. Since the user didn't specify the unit, I'll default to celsius. The location should be \"San Francisco, State, Country\". Wait, the example format is \"City, State, Country\", but San Francisco is a city in California, USA. So the location parameter would be \"San Francisco, California, United States\".\n\nThen, for tomorrow's temperature, the user mentioned the current date is 2024-09-30, so tomorrow would be 2024-10-01. The get_temperature_date function requires location, date, and unit. Again, using the same location and default unit. I need to format the date as \"Year-Month-Day\", which is 2024-10-01.\n\nWait, the current date given is 2024-09-30. If today is September 30, then tomorrow is October 1st. So the date parameter for the second function call should be \"2024-10-01\".\n\nI should make two separate function calls: one for the current temperature and another for tomorrow's date. Let me structure the JSON for both tool calls accordingly."},
    {"role": "assistant", "content": "", "function_call": {"name": "get_current_temperature", "arguments": "{\"location\": \"San Francisco, California, United States\", \"unit\": \"celsius\"}"}},
    {"role": "assistant", "content": "", "function_call": {"name": "get_temperature_date", "arguments": "{\"location\": \"San Francisco, California, United States\", \"date\": \"2024-10-01\", \"unit\": \"celsius\"}"}},
    {"role": "function", "name": "get_current_temperature", "content": '{"temperature": 26.1, "location": "San Francisco, California, United States", "unit": "celsius"}'},
    {"role": "function", "name": "get_temperature_date", "content": '{"temperature": 25.9, "location": "San Francisco, California, United States", "date": "2024-10-01", "unit": "celsius"}'},
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

- `no_think` mode:
```python
[
    {"role": "assistant", "content": "The current temperature in San Francisco, CA, USA is **26.1°C**.  \n\nFor tomorrow (2024-10-01), the temperature is projected to be **25.9°C**.  \n\nThere is a slight decrease in temperature expected from today to tomorrow."}
]
```

- `think` mode:
```python
[
    {"role": "assistant", "content": "", "reasoning_content": "Okay, the user asked for the current temperature in San Francisco and tomorrow's temperature. I called the get_current_temperature function for now and get_temperature_date for tomorrow. The responses came back with 26.1°C today and 25.9°C tomorrow. Let me present this info clearly.\n\nFirst, confirm the location to make sure there's no confusion. The current temp is 26.1°C, so I'll state that. Then, tomorrow's date is 2024-10-01, which is October 1st, so I'll mention the date in a user-friendly way. The temp drops slightly to 25.9°C. I should note the unit is Celsius as per the default. Keep the answer concise but informative. Maybe add a brief note about the slight decrease. Make sure the dates are correctly formatted and the temperatures are accurate based on the data provided."}, 
    {"role": "assistant", "content": "The current temperature in San Francisco, CA, USA is **26.1°C**.  \n\nFor tomorrow (2024-10-01), the temperature is projected to be **25.9°C**.  \n\nThere is a slight decrease in temperature expected from today to tomorrow."}
]
```


(heading-target)=
### vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving.
It uses the tokenizer from `transformers` to format the input, so we should have no trouble preparing the input.
In addition, vLLm also implements helper functions so that generated tool calls can be parsed automatically if the format is supported.

- `vllm` >= v0.8.5. 

For more information, check the [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#tool-calling-in-the-chat-completion-api).

We will use the OpenAI-Compatible API by `vllm` with the API client from the `openai` Python library.

#### Preparing

For Qwen3, the chat template in tokenizer_config.json has already included support for the Hermes-style tool use.
We simply need to start a OpenAI-compatible API with vLLM:

```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1
```

The inputs are the same with those in [the preparation code](#prepcode):

```python
tools = TOOLS
messages = MESSAGES
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

model_name = "Qwen/Qwen3-8B"
```

#### Tool Calls and Tool Results

We can use the create chat completions endpoint to query the model. 
Here is an example of the `no_think` mode:

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
        "chat_template_kwargs": {"enable_thinking": False}  # default to True
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

Note that the function arguments are JSON-formatted strings, which Qwen-Agent follows.

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
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30."},
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


## Finally

In whichever way you choose to use function calling with Qwen3, keep in mind that the limitation and the perks of prompt engineering applies:
- It is not guaranteed that the model generation will always follow the protocol even with proper prompting or templates.
  Especially, for the templates that are more complex and relies more on the model itself to think and stay on track than the ones that are simpler and relies on the template and the use of control or special tokens.
  The latter one, of course, requires some kind of training.
  In production code, be prepared that if it breaks, countermeasures or rectifications are in place.
- If in certain scenarios, the generation is not up to expectation, you can refine the template to add more instructions or constraints.
  While the templates mentioned here are general enough, they may not be the best or the most specific or the most concise for your use cases.
  The ultimate solution is fine-tuning using your own data.

Have fun prompting!
