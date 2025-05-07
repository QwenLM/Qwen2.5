Qwen-Agent
==========

`Qwen-Agent <https://github.com/QwenLM/Qwen-Agent>`__ is a framework for
developing LLM applications based on the instruction following, tool
usage, planning, and memory capabilities of Qwen.

This is a simple tutorial on using Qwen-Agent to quickly experience the
agentic capabilities of Qwen3. For more detailed information, please
refer to `Qwen-Agent <https://github.com/QwenLM/Qwen-Agent>`__
repository.

Installation
------------

-  Install the stable version from PyPI:

.. code:: bash

   pip install -U "qwen-agent[gui,rag,code_interpreter,mcp]"
   # Or use `pip install -U qwen-agent` for the minimal requirements.
   # The optional requirements, specified in double brackets, are:
   #   [gui] for Gradio-based GUI support;
   #   [rag] for RAG support;
   #   [code_interpreter] for Code Interpreter support;
   #   [mcp] for MCP support.

Developing Your Own Agent
-------------------------

Qwen3 excels in tool calling capabilities. Qwen-Agent encapsulates
tool-calling templates and tool-calling parsers internally, greatly
reducing coding complexity.

To define the available tools, you can use the MCP configuration file,
use the integrated tool of Qwen-Agent, or integrate other tools by
yourself.

.. code:: python

   import os
   from qwen_agent.agents import Assistant

   # Define LLM
   llm_cfg = {
       # Use a custom endpoint compatible with OpenAI API by vLLM/SGLang:
       'model': 'Qwen/Qwen3-32B',
       'model_server': 'http://localhost:8000/v1',  # api_base
       'api_key': 'EMPTY',

       # 'generate_cfg': {
       #     # When using vLLM/SGLang OAI API, pass the parameter of whether to enable thinking mode in this way
       #     'extra_body': {
       #         'chat_template_kwargs': {'enable_thinking': False}
       #     },
       #
       #     # Add: When the content is `<think>this is the thought</think>this is the answer`
       #     # Do not add: When the response has been separated by reasoning_content and content
       #     # This parameter will affect the parsing strategy of tool call
       #     # 'thought_in_content': True,
       # },
   }
   # llm_cfg = {
   #         # Use the model service provided by DashScope:
   #         'model': 'qwen3-235b-a22b',
   #         'model_type': 'qwen_dashscope',
   #
   #         # 'generate_cfg': {
   #         #     # When using the Dash Scope API, pass the parameter of whether to enable thinking mode in this way
   #         #     'enable_thinking': False,
   #         # },
   # }
   # llm_cfg = {
   #     # Use the OpenAI-compatible model service provided by DashScope:
   #     'model': 'qwen3-235b-a22b',
   #     'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
   #     'api_key': os.getenv('DASHSCOPE_API_KEY'),
   #
   #     # 'generate_cfg': {
   #     #     # When using Dash Scope OAI API, pass the parameter of whether to enable thinking mode in this way
   #     #     'extra_body': {
   #     #         'enable_thinking': False
   #     #     },
   #     # },
   # }

   # Define Tools
   tools = [
       {'mcpServers': {  # You can specify the MCP configuration file
               'time': {
                   'command': 'uvx',
                   'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
               },
               "fetch": {
                   "command": "uvx",
                   "args": ["mcp-server-fetch"]
               }
           }
       },
     'code_interpreter',  # Built-in tools
   ]

   # Define Agent
   bot = Assistant(llm=llm_cfg, function_list=tools)

   # Streaming generation
   messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]
   for responses in bot.run(messages=messages):
       pass
   print(responses)

For more detailed examples and MCP cookbooks, please refer to
`Qwen-Agent <https://github.com/QwenLM/Qwen-Agent>`__ repository.
