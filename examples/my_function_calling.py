import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
import requests

# 现在的情况是让agent调用ragTool来处理去噪后的文本,做法是让agent拿到rag接口返回的json格式中"符号"key对应的value
# 感觉会存在问题，有时候返回值铁定不符合规则
# prompt需要大改

# 步骤1（可选）：添加一个名为 `my_image_gen` 的自定义工具。
class ragTool(BaseTool):
    # 注意这里
    name = 'rag_tool'
    # `description` 用于告诉智能体该工具的功能。
    description = '标识库检索服务。输入一段描述，返回基于该描述检索到的表示符号中文名。'
    # `parameters` 告诉智能体该工具有哪些输入参数。
    parameters = [
        {
            'name': 'sign_description',
            'type': 'string',
            'description': '关于标识描述的list',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        # `params` 是由LLM智能体生成的参数。
        param = json5.loads(params)['sign_description']
        # param = urllib.parse.quote(param)
        # 要访问的URL
        url_set = "http://192.168.1.111/v1/conversation/set"
        url_chat = "http://192.168.1.111/v1/conversation/completion"
        params_set = {
            "dialog_id": "9da207c4a26a11efb6b60242ac120006",
        }
        # 发送请求
        response = requests.post(url_set, json=params_set)
        # 检查响应状态码是否为200（表示成功）
        result_json = response.json()
        cover_id = result_json["data"]["id"]

        params_chat = {
            "conversation_id": cover_id,
            "messages": [
                {
                    "content": "Hi! I'm your assistant, what can I do for you?",
                    "role": "assistant"
                },
                {
                    "role": "user",
                    "content": str(param)
                }
            ]
        }
        response2 = requests.post(url_chat, json=params_chat)
        result_json = response2.json()
        answer = result_json["data"]
        return json5.dumps(
            {'chat_answer': answer},
            ensure_ascii=False
        )


# # 步骤2：配置您所使用的LLM。
# llm_cfg = {
#         # 使用与OpenAI API兼容的模型服务，例如vLLM或Ollama：
#         'model': 'qwen-max',
#         'api_key':'sk-b14c9a9bfbcd4200b4f439db48b44841',
#         'model_server': 'dashscope'
# }
#
#
# # 步骤3：创建一个智能体。这里我们以 `Assistant` 智能体为例，它能够使用工具并读取文件。
# system_instruction = '''你是一个乐于助人的AI助手。
# 在收到用户的请求后，你应该：
# - 将用户的请求作为参数调用ragTool
# - 返回结果
# 你只需要把ragTool的返回结果中的"fuhao"对应value输出即可。'''
# tools = [ragTool()]  # `code_interpreter` 是框架自带的工具，用于执行代码。
#
# bot = Assistant(llm=llm_cfg,
#                 system_message=system_instruction,
#                 function_list=tools)
#
#
# # 步骤4：作为聊天机器人运行智能体。
# messages = []  # 这里储存聊天历史。
# # 例如，输入请求 "绘制一只狗并将其旋转90度"。
# query = input('用户请求: ')
# # 将用户请求添加到聊天历史。
# messages.append({'role': 'user', 'content': query})
# response = []
# completed_answer = None
# for response in bot.run(messages=messages):
#     # 流式输出。
#     # print('机器人回应:')
#     # pprint.pprint(response, indent=2)
#     completed_answer = response
#
# print(completed_answer[2]["content"])