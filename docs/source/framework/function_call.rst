Function Calling
================

We offer a wrapper for function calling over the dashscope API and the
OpenAI API in `Qwen-Agent <https://github.com/QwenLM/Qwen-Agent>`__.

Use Case
--------

.. code:: py

   import json
   import os
   from qwen_agent.llm import get_chat_model


   # Example dummy function hard coded to return the same weather
   # In production, this could be your backend API or an external API
   def get_current_weather(location, unit='fahrenheit'):
       """Get the current weather in a given location"""
       if 'tokyo' in location.lower():
           return json.dumps({
               'location': 'Tokyo',
               'temperature': '10',
               'unit': 'celsius'
           })
       elif 'san francisco' in location.lower():
           return json.dumps({
               'location': 'San Francisco',
               'temperature': '72',
               'unit': 'fahrenheit'
           })
       elif 'paris' in location.lower():
           return json.dumps({
               'location': 'Paris',
               'temperature': '22',
               'unit': 'celsius'
           })
       else:
           return json.dumps({'location': location, 'temperature': 'unknown'})


   def test():
       llm = get_chat_model({
           # Use the model service provided by DashScope:
           'model': 'qwen-max',
           'model_server': 'dashscope',
           'api_key': os.getenv('DASHSCOPE_API_KEY'),

           # Use the model service provided by Together.AI:
           # 'model': 'Qwen/Qwen1.5-14B-Chat',
           # 'model_server': 'https://api.together.xyz',  # api_base
           # 'api_key': os.getenv('TOGETHER_API_KEY'),

           # Use your own model service compatible with OpenAI API:
           # 'model': 'Qwen/Qwen1.5-72B-Chat',
           # 'model_server': 'http://localhost:8000/v1',  # api_base
           # 'api_key': 'EMPTY',
       })

       # Step 1: send the conversation and available functions to the model
       messages = [{
           'role': 'user',
           'content': "What's the weather like in San Francisco?"
       }]
       functions = [{
           'name': 'get_current_weather',
           'description': 'Get the current weather in a given location',
           'parameters': {
               'type': 'object',
               'properties': {
                   'location': {
                       'type': 'string',
                       'description':
                       'The city and state, e.g. San Francisco, CA',
                   },
                   'unit': {
                       'type': 'string',
                       'enum': ['celsius', 'fahrenheit']
                   },
               },
               'required': ['location'],
           },
       }]

       print('# Assistant Response 1:')
       responses = []
       for responses in llm.chat(messages=messages,
                                 functions=functions,
                                 stream=True):
           print(responses)

       messages.extend(responses)  # extend conversation with assistant's reply

       # Step 2: check if the model wanted to call a function
       last_response = messages[-1]
       if last_response.get('function_call', None):

           # Step 3: call the function
           # Note: the JSON response may not always be valid; be sure to handle errors
           available_functions = {
               'get_current_weather': get_current_weather,
           }  # only one function in this example, but you can have multiple
           function_name = last_response['function_call']['name']
           function_to_call = available_functions[function_name]
           function_args = json.loads(last_response['function_call']['arguments'])
           function_response = function_to_call(
               location=function_args.get('location'),
               unit=function_args.get('unit'),
           )
           print('# Function Response:')
           print(function_response)

           # Step 4: send the info for each function call and function response to the model
           messages.append({
               'role': 'function',
               'name': function_name,
               'content': function_response,
           })  # extend conversation with function response

           print('# Assistant Response 2:')
           for responses in llm.chat(
                   messages=messages,
                   functions=functions,
                   stream=True,
           ):  # get a new response from the model where it can see the function response
               print(responses)


   if __name__ == '__main__':
       test()
