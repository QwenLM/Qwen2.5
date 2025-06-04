# Thinking budget

This example demonstrates the inference process with thinking budgets using Qwen3 series models. The process involves two steps: 
1. the model generates reasoning content within the specified thinking budget
2. append the reasoning content to the conversation context and call the model again to get the final response

## Environment Setup

- `transformers >= 4.51.0`
- `openai >= 1.65.0`    

## Basic Usage

First, you should start a Qwen3 model in thinking mode. You can refer to [Quickstart](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md) for more details.

Then, you can use the following code to call the model with thinking budgets.

```python
from typing import Any, Dict, List

import openai
from transformers import AutoTokenizer


class ThinkingBudgetClient:
    def __init__(self, base_url: str, api_key: str, tokenizer_name_or_path: str):
        self.base_url = base_url
        self.api_key = api_key
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        thinking_budget: int = 512,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        assert max_tokens > thinking_budget, f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {thinking_budget=}"

        # 1. first call chat completion to get reasoning content
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=thinking_budget,
            **kwargs
        )
        
        content = response.choices[0].message.content
        reasoning_content = response.choices[0].message.reasoning_content.strip("\n")
        if content is None:
            # reasoning content is too long
            reasoning_content = (
                f"{reasoning_content}"
                "\n\nConsidering the limited time by the user, "
                "I have to give the solution based on the thinking directly now."
            )
        reasoning_tokens_len = len(self.tokenizer.encode(reasoning_content, add_special_tokens=False))
        remaining_tokens = max_tokens - reasoning_tokens_len
        assert remaining_tokens > 0, f"remaining tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the thinking_budget."

        # 2. append reasoning content to messages and call completion
        messages.append({"role": "assistant", "content": f"<think>\n{reasoning_content}\n</think>\n\n"})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True
        )
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=remaining_tokens,
            **kwargs
        )

        response_data = {
            "reasoning_content": reasoning_content,
            "content": response.choices[0].text,
            "finish_reason": response.choices[0].finish_reason,
        }
        return response_data

tokenizer_name_or_path = "Qwen/Qwen3-8B"
client = ThinkingBudgetClient(
    base_url="http://localhost:30000/v1", # Qwen3 deployed in thinking mode
    api_key="EMPTY",
    tokenizer_name_or_path=tokenizer_name_or_path
)

result = client.chat_completion(
    model="Qwen3-8B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a funny story about a cat."}
    ],
    thinking_budget=512,
    max_tokens=1024,
)
print(result["content"])
```