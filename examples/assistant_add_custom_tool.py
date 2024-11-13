"""An image generation agent implemented by assistant"""

import json
import os
import pprint
import urllib.parse

import json5

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


# Add a custom tool named my_image_gen：
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    description = 'AI绘画（图像生成）服务，输入文本描述，并返回基于文本信息绘制的图像URL。'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '所需图像内容的详细描述，用中文',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False,
        )


def init_agent_service():
    llm_cfg = {'model': 'qwen-max',
               'api_key':'sk-b14c9a9bfbcd4200b4f439db48b44841',
               'model_server': 'dashscope'}
    system = '''你是一个乐于助人的助手。
收到用户的请求后，你应该：
- 首先绘制一张图像并获取图像URL，
- 然后运行代码`request.get(image_url)`来下载图像，
- 最后从给定的文档中选择一个图像操作来处理图像。
请使用`plt.show()`显示图像。'''

    tools = [
        'my_image_gen',
        'code_interpreter',
    ]  # code_interpreter is a built-in tool in Qwen-Agent
    bot = Assistant(
        llm=llm_cfg,
        name='AI painting',
        description='AI painting service',
        system_message=system,
        function_list=tools,
        files=[os.path.join(ROOT_RESOURCE, 'doc.pdf')],
    )

    return bot


def test(query="你好"):
    # Define the agent
    messages = []  # 这存储聊天记录。
    bot = init_agent_service()
    messages.append({'role': 'user', 'content': query})
    response = []
    for response in bot.run(messages=messages):
        # 流式输出。
        print('机器人响应：')
        pprint.pprint(response, indent=2)
    # 将机器人响应添加到聊天记录中。
    messages.extend(response)



def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        query = input('user question: ')
        messages.append({'role': 'user', 'content': query})
        response = []
        for response in bot.run(messages=messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    # Define the agent
    bot = init_agent_service()
    chatbot_config = {
        'prompt.suggestions': [
            '画一只猫的图片',
            '画一只可爱的小腊肠狗',
            '画一幅风景画，有湖有山有树',
        ]
    }
    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()


if __name__ == '__main__':
    test()
    # app_tui()
    # app_gui()
