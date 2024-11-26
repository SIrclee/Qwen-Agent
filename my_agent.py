from qwen_agent.agents import Assistant
from examples.my_function_calling import ragTool
#
# Agent1：用来分析用户的输入，Hyde。对输入进行联想，例如公厕->[卫生间]，[公共厕所]，[洗手间]
# Agent2：调用工具（Rag发成OpenAI格式接口），调用Rag服务获得结果/也可以简单点把所有标志都搞在一个文件中（搞几个意思意思就行）
# Agent3：反思检索结果的正确性，从结果列表中选择出最符合初始描述的一个。根据结果可以选择不同的route，执行对应的操作（对应CRAG的操作）
#
# LLM配置
llm_cfg = {
            'model': 'qwen-max',
            'api_key':'sk-b14c9a9bfbcd4200b4f439db48b44841',
            'model_server': 'dashscope'}

###
### 第一个agent提示词
###
first_instruction = '''
# 角色
你是一个标准化公共标识领域的专家，你对中国的各种公共标识的中文名以及用途非常了解。
# 背景
在中国，一个公共标识虽有唯一的标准中文名，但在实际运用中，只要遵循通俗易懂的原则，其他类似表述也可视为正确用法。
比如 “卫生间” 这一标准标识，在现实场景里，写成 “公共厕所”“公厕”“洗手间” 等情况也很常见。
另外，还存在依据实际场景对公共标识灵活运用的情况。例如，“仓库重地，严禁烟火” 可对应到基本的 “严禁烟火” 标识；“南出站口” 可对应到 “出口” 标识；“车站行李寄存处” 可对应 “行李寄存” 标识。
# 任务步骤
- 用户输入是OCR产生的结果，可能有噪声与错误，请你优化该描述。
- 罗列出几个与该描述最为契合的标识标准中文名。
- 去噪后的用户输入也要返回

严格按照如下格式回复：
["去噪后的用户输入","标识中文名1","标识中文名2"...]

用户输入如下：
$input$
'''

# TODO
# 将Agent1返回的列表，add原始去噪的描述（prompt还是业务逻辑来做一下）
###
### 第二个agent提示词
###
second_instruction = '''你是一个乐于助人的AI助手。
在收到用户的请求后，你应该：
- 用户的请求是一个列表格式的字符串，你需要保留列表的格式
- 将用户的请求作为参数调用ragTool
- 返回结果

注意！！你必须只输出调用工具后的结果，一定不要有任何其他输出。
按如下格式要求回复：
result：结果1

用户请求如下：
$input$
'''

while True:
    query = input('用户请求: ')
    # query = "检票口前方100米"
    # 去噪/语义提取Agent
    first_msg = []
    first_msg.append({'role': 'user', 'content': query})
    first_query = first_instruction.replace('$input$', query)

    first_agent = Assistant(
        llm=llm_cfg,
        system_message=first_query,
        name='DenoiseAgent',
        description='描述去噪'
    )

    *_, first_response = first_agent.run(first_msg)
    print("First Response:",first_response)

    print("\n\n-------------------------------------- agent分割线 ---------------------------------------------\n")

    # 跨模态予语义检索Agent
    messages = []
    second_query = second_instruction.replace('$input$', first_response[0]['content'])
    messages.append({'role': 'user', 'content': second_query})
    print("Second query:",second_query)
    tools = [ragTool()]
    second_agent = Assistant(
        llm=llm_cfg,
        system_message=second_query,
        name='CrossModalAgent',
        function_list=tools,
        description='知识库检索'
    )

    *_, second_response = second_agent.run(messages)
    print("Secont Response:",second_response[-1]["content"])
    print("\n\n")