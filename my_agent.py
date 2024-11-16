from qwen_agent.agents import Assistant
from examples.my_function_calling import ragTool
#TODO
# Agent1：用来分析用户的输入，做规划、Plan
# Agent2：调用工具（Rag发成OpenAI格式接口），调用Rag服务获得结果/也可以简单点把所有标志都搞在一个文件中（搞几个意思意思就行）
# Agent3：反思检索结果的正确性，根据结果可以选择不同的route，执行对应的操作（对应CRAG的操作）

# LLM配置
llm_cfg = {
            'model': 'qwen-max',
            'api_key':'sk-b14c9a9bfbcd4200b4f439db48b44841',
            'model_server': 'dashscope'}

###
### 第一个agent提示词
###
first_instruction = '''
# 背景
给你对公共标识牌进行OCR识别后的结果，你需要从去除这段文字因为OCR所产生的噪声，如果有描述有明显识别错误，进行改正

用户输入如下：
$input$

按如下格式要求回复：
描述：[去噪后的公共标识描述]
'''

###
### 第二个agent提示词
###
second_instruction = '''你是一个乐于助人的AI助手。
在收到用户的请求后，你应该：
- 将用户的请求作为参数调用ragTool
- 返回结果
你只需要把ragTool的返回结果中的"fuhao"对应value输出即可。
用户请求如下：
$input$
'''

while True:
    query = input('用户请求: ')

    first_msg = []
    first_msg.append({'role': 'user', 'content': query})
    first_query = first_instruction.replace('$input$', query)

    first_agent = Assistant(
        llm=llm_cfg,
        system_message=first_query,
        name='Assistant',
        description='意图识别'
    )

    *_, first_response = first_agent.run(first_msg)
    print(first_response)

    print("\n\n-------------------------------------- agent分割线 ---------------------------------------------\n")

    ### 第二个agent
    messages = []
    second_query = second_instruction.replace('$input$', first_response[0]['content'])
    messages.append({'role': 'user', 'content': second_query})
    print(second_query)
    tools = [ragTool()]
    second_agent = Assistant(
        llm=llm_cfg,
        system_message=second_query,
        name='Assistant',
        function_list=tools,
        description='响应回复'
    )

    *_, second_response = first_agent.run(messages)
    print(second_response)
    messages.extend(second_response)
    print("\n\n")