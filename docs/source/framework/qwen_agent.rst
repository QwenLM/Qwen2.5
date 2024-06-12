Qwen-Agent
==========

`Qwen-Agent <https://github.com/QwenLM/Qwen-Agent>`__ is a framework for
developing LLM applications based on the instruction following, tool
usage, planning, and memory capabilities of Qwen. It also comes with
example applications such as Browser Assistant, Code Interpreter, and
Custom Assistant.

Installation
------------

.. code:: bash

   git clone https://github.com/QwenLM/Qwen-Agent.git
   cd Qwen-Agent
   pip install -e ./

Developing Your Own Agent
-------------------------

Qwen-Agent provides atomic components such as LLMs and prompts, as well
as high-level components such as Agents. The example below uses the
Assistant component as an illustration, demonstrating how to add custom
tools and quickly develop an agent that uses tools.

.. code:: py

   import json
   import os

   import json5
   import urllib.parse
   from qwen_agent.agents import Assistant
   from qwen_agent.tools.base import BaseTool, register_tool

   llm_cfg = {
       # Use the model service provided by DashScope:
       'model': 'qwen-max',
       'model_server': 'dashscope',
       # 'api_key': 'YOUR_DASHSCOPE_API_KEY',
       # It will use the `DASHSCOPE_API_KEY' environment variable if 'api_key' is not set here.

       # Use your own model service compatible with OpenAI API:
       # 'model': 'Qwen/Qwen2-72B-Instruct',
       # 'model_server': 'http://localhost:8000/v1',  # api_base
       # 'api_key': 'EMPTY',

       # (Optional) LLM hyperparameters for generation:
       'generate_cfg': {
           'top_p': 0.8
       }
   }
   system = 'According to the user\'s request, you first draw a picture and then automatically run code to download the picture ' + \
             'and select an image operation from the given document to process the image'

   # Add a custom tool named my_image_gen：
   @register_tool('my_image_gen')
   class MyImageGen(BaseTool):
       description = 'AI painting (image generation) service, input text description, and return the image URL drawn based on text information.'
       parameters = [{
           'name': 'prompt',
           'type': 'string',
           'description': 'Detailed description of the desired image content, in English',
           'required': True
       }]

       def call(self, params: str, **kwargs) -> str:
           prompt = json5.loads(params)['prompt']
           prompt = urllib.parse.quote(prompt)
           return json.dumps(
               {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
               ensure_ascii=False)


   tools = ['my_image_gen', 'code_interpreter']  # code_interpreter is a built-in tool in Qwen-Agent
   bot = Assistant(llm=llm_cfg,
                   system_message=system,
                   function_list=tools,
                   files=[os.path.abspath('doc.pdf')])

   messages = []
   while True:
       query = input('user question: ')
       messages.append({'role': 'user', 'content': query})
       response = []
       for response in bot.run(messages=messages):
           print('bot response:', response)
       messages.extend(response)

The framework also provides more atomic components for developers to
combine. For additional showcases, please refer to
`examples <https://github.com/QwenLM/Qwen-Agent/tree/main/examples>`__.
