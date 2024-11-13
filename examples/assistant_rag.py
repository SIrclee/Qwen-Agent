from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI


def test():
    llm_cfg = {
        'model': 'qwen-max',
        'api_key': 'sk-b14c9a9bfbcd4200b4f439db48b44841',
        'model_server': 'dashscope'}
    bot = Assistant(llm=llm_cfg)
    messages = [{'role': 'user', 'content': [{'text': '如果想翻转图像应该怎么做？'}, {'file': 'E:/bishe_code/Qwen-Agent/examples/resource/doc.pdf'}]}]
    for rsp in bot.run(messages):
        print(rsp)


def app_gui():
    # Define the agent
    bot = Assistant(llm={'model': 'qwen2-72b-instruct'},
                    name='Assistant',
                    description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。')
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '介绍图一'
            },
            {
                'text': '第二章第一句话是什么？'
            },
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # test()
    app_gui()
