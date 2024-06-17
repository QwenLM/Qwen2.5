# Requirement:
#   pip install "openai<1.0"
# Usage:
#   python openai_api_issue.py
# Visit http://localhost:8000/docs for documents.

import base64
import copy
import json
import time
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pprint import pprint
from typing import Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer
from transformers.generation import GenerationConfig
from qwen2chat import Qwen2ForChatLM


class BasicAuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.required_credentials = base64.b64encode(
            f'{username}:{password}'.encode()).decode()

    async def dispatch(self, request: Request, call_next):
        authorization: str = request.headers.get('Authorization')
        if authorization:
            try:
                schema, credentials = authorization.split()
                if credentials == self.required_credentials:
                    return await call_next(request)
            except ValueError:
                pass

        headers = {'WWW-Authenticate': 'Basic'}
        return Response(status_code=401, headers=headers)


def _gc(forced: bool = False):
    global args
    if args.disable_gc and not forced:
        return

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(forced=True)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class ModelCard(BaseModel):
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'owner'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function']
    content: Optional[str]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Union[ChatMessage]
    finish_reason: Literal['stop', 'length', 'function_call']


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal['chat.completion', 'chat.completion.chunk']
    choices: List[Union[ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get('/v1/models', response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id='gpt-3.5-turbo')
    return ModelList(data=[model_card])


# To work around that unpleasant leading-\n tokenization issue!
def add_extra_stop_words(stop_words: list):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip('\n')
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def trim_stop_words(response: str, stop_words: list):
    # Remove the first occurrence of the stop word and everything after it in the response
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


TOOL_DESC_WITH_PARAMETERS = (
    '{name_for_model}: Call this tool to interact with the {name_for_human} API.'
    ' What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}'
)
TOOL_DESC_NO_PARAMETERS = (
    '{name_for_model}: Call this tool to interact with the {name_for_human} API.'
    ' What is the {name_for_human} API useful for? {description_for_model}'
)

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action, if no parameters are provided, marking this as empty.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

REACT_INSTRUCTION_NO_PARAMETERS = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def parse_messages(messages, functions):
    if all(m.role != 'user' for m in messages):
        raise HTTPException(
            status_code=400,
            detail='Invalid request: Expecting at least one user message.',
        )

    messages = copy.deepcopy(messages)
    if messages[0].role == 'system':
        system = messages.pop(0).content.lstrip('\n').rstrip()
    else:
        system = 'You are a helpful assistant.'  # 拿出系统消息，如果没有的话用默认的

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:  # functions have 2 styles, one is qwen style, the other is openai style. here porcess the two style simultaneously.
            name = func_info.get('name', '')
            name_m = func_info.get('name_for_model', name)
            name_h = func_info.get('name_for_human', name)
            desc = func_info.get('description', '')
            desc_m = func_info.get('description_for_model', desc)
            if "parameters" in func_info:
                tool = TOOL_DESC_WITH_PARAMETERS.format(  # qwen style
                    name_for_model=name_m,
                    name_for_human=name_h,
                    # Hint: You can add the following format requirements in description:
                    #   "Format the arguments as a JSON object."
                    #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                    description_for_model=desc_m,
                    parameters=json.dumps(func_info['parameters'],
                                          ensure_ascii=False),
                )
            else:
                tool = TOOL_DESC_NO_PARAMETERS.format(  # qwen style
                    name_for_model=name_m,
                    name_for_human=name_h,
                    description_for_model=desc_m,
                )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = '\n\n'.join(tools_text)
        tools_name_text = ', '.join(tools_name_text)
        instruction = (REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        ).lstrip('\n').rstrip())
    else:
        instruction = ''

    messages_with_fncall = messages
    messages = []
    for m_idx, m in enumerate(messages_with_fncall):
        role, content, func_call = m.role, m.content, m.function_call
        content = content or ''
        content = content.lstrip('\n').rstrip()
        if role == 'function':
            if (len(messages) == 0) or (messages[-1].role != 'assistant'):
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting role assistant before role function.',
                )
            messages[-1].content += f'\nObservation: {content}'
            if m_idx == len(messages_with_fncall) - 1:
                # add a prefix for text completion
                messages[-1].content += '\nThought:'
        elif role == 'assistant':
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting role user before role assistant.',
                )
            if func_call is None:
                if functions:
                    content = f'Thought: I now know the final answer.\nFinal Answer: {content}'
            else:
                f_name, f_args = func_call['name'], func_call['arguments']
                if not content.startswith('Thought:'):
                    content = f'Thought: {content}'
                content = f'{content}\nAction: {f_name}\nAction Input: {f_args}'
            if messages[-1].role == 'user':
                messages.append(
                    ChatMessage(role='assistant',
                                content=content.lstrip('\n').rstrip()))
            else:
                messages[-1].content += '\n' + content
        elif role == 'user':
            messages.append(
                ChatMessage(role='user',
                            content=content.lstrip('\n').rstrip()))
        else:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid request: Incorrect role {role}.')

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == 'user':
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail='Invalid request')

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == 'user' and messages[i + 1].role == 'assistant':
            usr_msg = messages[i].content.lstrip('\n').rstrip()
            bot_msg = messages[i + 1].content.lstrip('\n').rstrip()
            if instruction and (i == len(messages) - 2):
                usr_msg = f'{instruction}\n\nQuestion: {usr_msg}'
                instruction = ''
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Expecting exactly one user (or function) role before every assistant role.',
            )
    if instruction:
        assert query is not _TEXT_COMPLETION_CMD  # if false, will show an error
        query = f'{instruction}\n\nQuestion: {query}'
    return query, history, system


def parse_response(response):
    """
    Parsing into Openai's response format:
    ChatCompletionResponseChoice(index=0, message=ChatMessage(role='assistant', content='我需要查询波士顿的当前天气情况。',
    function_call={'name': 'get_current_weather', 'arguments': '{"location": "波士顿"}'}), finish_reason='function_call')
    """
    func_name, func_args = '', ''
    i = response.find('\nAction:')
    j = response.find('\nAction Input:')
    k = response.find('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + '\nObservation:'  # Add it back.
        k = response.find('\nObservation:')
        func_name = response[i + len('\nAction:'):j].strip()
        func_args = response[j + len('\nAction Input:'):k].strip()

    if func_name:
        response = response[:i]
        t = response.find('Thought: ')
        if t >= 0:
            response = response[t + len('Thought: '):]
        response = response.strip()
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role='assistant',
                content=response,
                function_call={
                    'name': func_name,
                    'arguments': func_args
                },
            ),
            finish_reason='function_call',
        )
        return choice_data

    z = response.rfind('\nFinal Answer: ')
    if z >= 0:
        response = response[z + len('\nFinal Answer: '):]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=response),
        finish_reason='stop',
    )
    return choice_data


# completion mode, not chat mode
def text_complete_last_message(history, stop_words_ids, gen_kwargs, system):
    im_start = '<|im_start|>'
    im_end = '<|im_end|>'
    prompt = f'{im_start}system\n{system}{im_end}'
    for i, (query, response) in enumerate(history):
        query = query.lstrip('\n').rstrip()
        response = response.lstrip('\n').rstrip()
        prompt += f'\n{im_start}user\n{query}{im_end}'
        prompt += f'\n{im_start}assistant\n{response}{im_end}'
    prompt = prompt[:-len(im_end)]

    _stop_words_ids = [tokenizer.encode(im_end)]
    if stop_words_ids:
        for s in stop_words_ids:
            _stop_words_ids.append(s)
    stop_words_ids = _stop_words_ids

    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = model.generate(input_ids,
                            stop_words_ids=stop_words_ids,
                            **gen_kwargs).tolist()[0]
    output = tokenizer.decode(output, errors='ignore')
    assert output.startswith(prompt)
    output = output[len(prompt):]
    output = trim_stop_words(output, ['<|endoftext|>', im_end])
    print(f'<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>')
    return output


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    gen_kwargs = {}
    if request.top_k is not None:
        gen_kwargs['top_k'] = request.top_k
    if request.temperature is not None:
        if request.temperature < 0.01:
            gen_kwargs['top_k'] = 1  # greedy decoding
        else:
            # Not recommended. Please tune top_p instead.
            gen_kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        gen_kwargs['top_p'] = request.top_p
    if request.max_length is not None:
        gen_kwargs['max_new_tokens'] = request.max_length
    else:
        gen_kwargs['max_new_tokens'] = 512

    stop_words = add_extra_stop_words(request.stop)
    if request.functions:
        stop_words = stop_words or []
        if 'Observation:' not in stop_words:
            stop_words.append('Observation:')

    query, history, system = parse_messages(request.messages,
                                            request.functions)

    if request.stream:
        if request.functions:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Function calling is not yet implemented for stream mode.',
            )
        generate = predict(query,
                           history,
                           request.model,
                           stop_words,
                           gen_kwargs,
                           system=system)
        return EventSourceResponse(generate, media_type='text/event-stream')

    stop_words_ids = [tokenizer.encode(s)
                      for s in stop_words] if stop_words else None
    if query is _TEXT_COMPLETION_CMD:
        response = text_complete_last_message(history,
                                              stop_words_ids=stop_words_ids,
                                              gen_kwargs=gen_kwargs,
                                              system=system)
    else:
        response, _ = model.chat(
            tokenizer,
            query,
            history=history,
            system=system,
            stop_words_ids=stop_words_ids,
            **gen_kwargs,
        )
        print('<chat>')
        pprint(history, indent=2)
        print(f'{query}\n<!-- *** -->\n{response}\n</chat>')
    _gc()

    response = trim_stop_words(response, stop_words)
    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response),
            finish_reason='stop',
        )
    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


def _dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:  # pydantic<2.0.0
        return data.json(*args, **kwargs)  # noqa


async def predict(
        query: str,
        history: List[List[str]],
        model_id: str,
        stop_words: List[str],
        gen_kwargs: Dict,
        system: str,
):
    global model, tokenizer
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role='assistant'), finish_reason=None)
    chunk = ChatCompletionResponse(model=model_id,
                                   choices=[choice_data],
                                   object='chat.completion.chunk')
    yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    current_length = 0
    stop_words_ids = [tokenizer.encode(s)
                      for s in stop_words] if stop_words else None

    delay_token_num = max([len(x) for x in stop_words]) if stop_words_ids else 0
    response_generator = model.chat_stream(tokenizer,
                                           query,
                                           history=history,
                                           stop_words_ids=stop_words_ids,
                                           system=system,
                                           **gen_kwargs)
    for _new_response in response_generator:
        if len(_new_response) <= delay_token_num:
            continue
        new_response = _new_response[:-delay_token_num] if delay_token_num else _new_response

        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
        chunk = ChatCompletionResponse(model=model_id,
                                       choices=[choice_data],
                                       object='chat.completion.chunk')
        yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    if current_length != len(_new_response):
        # Determine whether to print the delay tokens
        delayed_text = _new_response[current_length:]
        new_text = trim_stop_words(delayed_text, stop_words)
        if len(new_text) > 0:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
            chunk = ChatCompletionResponse(model=model_id,
                                           choices=[choice_data],
                                           object='chat.completion.chunk')
            yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                     delta=DeltaMessage(),
                                                     finish_reason='stop')
    chunk = ChatCompletionResponse(model=model_id,
                                   choices=[choice_data],
                                   object='chat.completion.chunk')
    yield '{}'.format(_dump_json(chunk, exclude_unset=True))
    yield '[DONE]'

    _gc()


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint-path',
        type=str,
        default='Qwen/Qwen2-7B-Instruct',
        help='Checkpoint name or path, default to %(default)r'
    )
    parser.add_argument('--device',
                        help='number of device of cuda, e.g cuda:0',
                        type=str,
                        default='cuda:2'
                        )
    parser.add_argument('--api-auth', help='API authentication credentials')
    parser.add_argument('--cpu-only',
                        action='store_true',
                        help='Run demo with CPU only')
    parser.add_argument('--server-port',
                        type=int,
                        default=6071,
                        help='Demo server port.')
    parser.add_argument(
        '--server-name',
        type=str,
        default='0.0.0.0',
        help=
        'Demo server name. Default: 127.0.0.1, which is only visible from the local computer.'
        ' If you want other computers to access your server, use 0.0.0.0 instead.',
    )
    parser.add_argument(
        '--disable-gc',
        action='store_true',
        help='Disable GC after each response generated.',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()

    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.checkpoint_path,
        resume_download=True
    )

    if args.api_auth:
        app.add_middleware(BasicAuthMiddleware,
                           username=args.api_auth.split(':')[0],
                           password=args.api_auth.split(':')[1])

    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = args.device

    model = Qwen2ForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        resume_download=True,
        torch_dtype="auto",
    )
    model.eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        resume_download=True,
    )
    model.__class__ = Qwen2ForChatLM

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
