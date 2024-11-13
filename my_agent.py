from qwen_agent.agents import Assistant

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
你是一个综合的工具，用于从如下方面处理用户的输入内容：
1. 识别真实意图
2. 提取人物及其关系
3. 以如上两点为维度，拆分子内容块
4. 对整个输入做情感分析

用户输入如下：
$input$

按如下格式要求回复：
意图：[用户的意图想法]
人物关系：[人物1]-[关系]、[人物2]-[关系]
子内容1：[拆分后的第一个子内容]
.....
子内容n：[拆分后的第n个子内容]
情感分析：[情感状态]
'''

###
### 第二个agent提示词
###
second_instruction = '''
围绕如下已知的前提条件做问询，从而了解患者的实际情况及实际问题，已知前提条件如下：
$input$

按如下要求回复：
1. 以问询的方式，与用户进行沟通
2. 必须紧扣已知前提条件进行回复
'''

while True:
    query = input('用户请求: ')

    first_msg = []
    first_msg.append({'role': 'user', 'content': query})
    first_query = first_instruction.replace('$input$', query)

    ###
    ### 集合了NLP任务的AI
    ### 1. 文本摘要
    ### 2. 关系抽取
    ### 3. 实体识别
    ### 4. 情感分析
    ### 5....
    first_agent = Assistant(
        llm=llm_cfg,
        system_message=first_query,
        name='Assistant',
        description='意图识别'
    )

    *_, first_response = first_agent.run(first_msg)
    print(first_response)

    print("\n\n-------------------------------------- agent分割线 ---------------------------------------------\n")

    ###
    ### 第二个agent
    ###
    messages = []
    second_query = second_instruction.replace('$input$', first_response[0]['content'])
    messages.append({'role': 'user', 'content': second_query})
    print(second_query)

    second_agent = Assistant(
        llm=llm_cfg,
        system_message=second_query,
        name='Assistant',
        description='响应回复'
    )

    *_, second_response = first_agent.run(messages)
    print(second_response)
    messages.extend(second_response)
    print("\n\n")