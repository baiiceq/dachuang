import os
import requests
import json
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain.tools import StructuredTool
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

# =============================================================================
# 全局对话记忆（实现多轮对话功能，要求最终链仅有一个输入变量）
# =============================================================================
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
# =============================================================================
# 1. 初始化思维链（Chain-of-Thought, COT）
# =============================================================================
def init_cot_chain():
    """
    初始化用于生成思维链的链条，分步思考问题。

    Returns:
        LLMChain: 已配置的思维链链条。
    """
    cot_template = '''
你是一个逻辑推理专家，负责将问题进行逐步思考并生成推理过程。
请一步一步思考题目: {question}，生成详细的推理步骤(COT)
要求：
- 分步骤解释核心步骤 
- 尽可能详细分析问题各种可能情况
- 将问题分解成一个个小任务，体现思维链特点
- 对于不清楚的知识或概念不要随便假设
我不需要问题答案，只需要你思考问题的过程。
    '''
    prompt = PromptTemplate(input_variables=["question"], template=cot_template)

    llm_cot = Tongyi(
        model="qwen-max",
        temperature=0.8,
        openai_api_key='sk-e2492dea19b945059a9a05abb4d0fc0b',
        openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )
    chain_cot = LLMChain(llm=llm_cot, prompt=prompt)
    return chain_cot


# =============================================================================
# 2. 初始化本地知识库向量数据库
# =============================================================================
def init_vector_db():
    """
    初始化本地 C 语言习题知识库的向量数据库。

    Returns:
        Chroma: 构建好的向量数据库对象。
    """
    os.environ["DASHSCOPE_API_KEY"] = "your-dashscope-api-key"  # 请替换为实际密钥
    persist_directory = 'data/'  # 持久化目录

    loader = TextLoader("c_language_exercises.txt", encoding="utf-8")
    documents = loader.load()

    # 自定义文本切分器：按行拆分，每行视为一个习题
    class ExerciseTextSplitter(RecursiveCharacterTextSplitter):
        def split_text(self, text):
            exercises = text.split('\n')
            return [exercise.strip() for exercise in exercises if exercise.strip()]

    text_splitter = ExerciseTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embedding = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key="sk-e2492dea19b945059a9a05abb4d0fc0b"
    )
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding)
    return vectordb


# =============================================================================
# 3. 初始化基于本地知识库的检索问答链
# =============================================================================
def init_retrieval_qa_chain(vectordb):
    """
    初始化基于本地知识库的问答链，用于根据问题推荐相关习题。

    Args:
        vectordb (Chroma): 向量数据库对象。

    Returns:
        RetrievalQA: 配置好的检索问答链。
    """
    llm = Tongyi(
        model="qwen-max",
        temperature=0.5,
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    prompt_template = """
你是一个C语言习题生成大师，为学生生成符合特点的习题。
用户问题：
{question}
请先分析题目涉及的知识点和难度，再根据以下已有习题推荐相关题目：{context}
要求：
1. 优先选出题库中与用户问题关联度最高的习题作为推荐；
2. 若题库中无高关联习题，则仿照题库风格生成5道相关习题；
3. 最终输出仅包含推荐习题，并自动排序。
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context"])
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
        output_key='习题'
    )
    return qa_chain


# =============================================================================
# 4. 初始化网络搜索工具及相关链条
# =============================================================================
def init_search_tool(llm):
    """
    初始化网络搜索工具、摘要链及 Agent，用于执行网络搜索和生成摘要。

    Args:
        llm: 用于搜索和摘要生成的语言模型实例。

    Returns:
        tuple: (search_tool, summary_chain, agent, agent_prompt)
    """

    def bocha_websearch_tool(query: str, count: int = 20) -> str:
        """
        通过 Bocha API 执行网络搜索。

        Args:
            query (str): 搜索关键词。
            count (int): 返回结果数量（默认20）。

        Returns:
            str: 格式化后的搜索结果摘要。
        """
        url = 'https://api.bochaai.com/v1/web-search'
        headers = {
            'Authorization': f'Bearer sk-782dce9336b549299c48f4607376d0f2',  # 替换为实际 API 密钥
            'Content-Type': 'application/json'
        }
        data = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": count
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            if json_data.get("code") != 200 or not json_data.get("data"):
                return "搜索服务暂时不可用"
            webpages = json_data["data"]["webPages"]["value"]
            return "\n\n".join(
                f"标题：{p['name']}\n链接：{p['url']}\n摘要：{p['summary']}"
                for p in webpages[:5]
            )
        except Exception as e:
            return f"搜索失败：{str(e)}"

    search_tool = StructuredTool.from_function(
        func=bocha_websearch_tool,
        name="web_search",
        description="""
用于执行网络搜索的工具。
输入应为包含以下键的 JSON 对象：
- query: 搜索关键词（例如 "人字路口 真话 假话"）
- count: 可选，结果数量（默认20）
        """
    )

    summary_prompt = PromptTemplate(
        input_variables=["question", "web_results"],
        template="""你是一个擅长总结网络信息的信息分析师，请结合下列搜索结果生成准确、清晰的摘要。
问题：{question}
搜索结果：
{web_results}
请提炼关键信息供参考。"""
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    agent_prompt_template = """你是一个智能搜索助手，请按以下步骤处理问题：
1. 分析问题并提取3-5个核心关键词；
2. 使用 web_search 工具进行搜索；
3. 综合搜索结果生成结构化报告。
当前问题：
{question}
请以 JSON 格式返回如下字段：
{{
  "keywords": ["关键词1", "关键词2"],
  "tool_input": "要搜索的关键词组合"
}}"""
    agent_prompt = PromptTemplate(input_variables=['question'], template=agent_prompt_template)

    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    return search_tool, summary_chain, agent, agent_prompt


# =============================================================================
# 5. 封装网络搜索：调用搜索工具并生成摘要
# =============================================================================
def get_web_search_summary(input_text, summary_chain, agent, agent_prompt):
    """
    根据用户问题调用网络搜索工具，并生成搜索摘要。

    Args:
        input_text (str): 用户问题。
        summary_chain (LLMChain): 用于生成摘要的链条。
        agent: 网络搜索的 Agent。
        agent_prompt (PromptTemplate): Agent 的提示模板。

    Returns:
        str: 生成的网络搜索摘要文本。
    """
    formatted_input = agent_prompt.format(question=input_text)
    search_response = agent.invoke({"input": formatted_input})
    summary_result = summary_chain.invoke({
        'question': input_text,
        'web_results': search_response['output']
    })
    return summary_result['text']


# =============================================================================
# 6. 定义最终答案生成提示模板（只接收一个输入变量 combined_input）
# =============================================================================
RESULT_PROMPT = '''
{chat_history}
你是一位优秀的C语言教授，擅长从学生角度讲解编程问题。请阅读下面整合后的内容，并生成详细解答：

{combined_input}

回答要求:
1. 详细讲解问题解决过程，条理清晰；
2. 针对复杂问题提供示例和代码说明；
3. 分析学生可能遇到的难点及易错点；
4. 根据问题特点提出相关拓展问题。
'''


# =============================================================================
# 7. 处理函数 process_result（支持多轮对话和可选网络搜索）
# =============================================================================
def process_result(input_text, qwen, chain_cot, summary_chain, agent, agent_prompt, use_web_search=True):
    """
    处理用户问题，整合思维链和（可选）网络搜索摘要后生成最终答案。

    Args:
        input_text (str): 用户提问。
        qwen: 用于生成最终答案的通义千问模型实例。
        chain_cot (LLMChain): 生成思维链的链条。
        summary_chain (LLMChain): 生成搜索摘要的链条。
        agent: 网络搜索的 Agent。
        agent_prompt (PromptTemplate): Agent 的提示模板。
        use_web_search (bool): 是否调用网络搜索（默认 True）。

    Returns:
        tuple: (最终回答, 网络搜索摘要, 思维链文本)
    """
    # 调用思维链生成思维过程
    cot_result = chain_cot.invoke({"question": input_text})
    thought_text = cot_result["text"] if "text" in cot_result else cot_result

    # 根据标志判断是否调用网络搜索生成摘要
    if use_web_search:
        web_summary = get_web_search_summary(input_text, summary_chain, agent, agent_prompt)
    else:
        web_summary = ""

    # 将问题、思维链和网络摘要合并成单一输入变量
    combined_input = f"### 问题:\n{input_text}\n\n### 思维链 / 推理:\n{thought_text}\n\n### 网络信息:\n{web_summary}"

    # 创建最终答案生成链，采用单一输入变量，并注入对话历史
    final_prompt = PromptTemplate(
        input_variables=['chat_history', 'combined_input'],
        template=RESULT_PROMPT
    )
    chain_result = LLMChain(llm=qwen, prompt=final_prompt, memory=conversation_memory)
    response = chain_result.invoke({
        'chat_history': conversation_memory.buffer,
        'combined_input': combined_input
    })
    return response['text'], web_summary, thought_text


# =============================================================================
# 8. 处理函数 thinking_input（基于 DeepSeek 深度推理，支持可选网络搜索及多轮对话）
# =============================================================================
def thinking_input(input_text, qwen, summary_chain, agent, agent_prompt, use_web_search=True):
    """
    通过 DeepSeek 深度推理，并结合（可选）网络搜索摘要和对话记忆生成答案。

    Args:
        input_text (str): 用户提问。
        qwen: 用于生成最终答案的通义千问模型实例。
        summary_chain (LLMChain): 生成搜索摘要的链条。
        agent: 网络搜索的 Agent。
        agent_prompt (PromptTemplate): Agent 的提示模板。
        use_web_search (bool): 是否调用网络搜索（默认 True）。

    Returns:
        tuple: (最终回答, 网络搜索摘要, DeepSeek 推理文本)
    """
    # 调用 DeepSeek 获取深度推理结果
    client = OpenAI(api_key="sk-251abe6e76f64f79a2321611c6f67bc6", base_url="https://api.deepseek.com")
    messages = [{"role": "user", "content": input_text}]
    deepseek_response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
    )
    reasoning_content = deepseek_response.choices[0].message.reasoning_content

    # 根据标志判断是否调用网络搜索生成摘要
    if use_web_search:
        formatted_input = agent_prompt.format(question=input_text)
        search_response = agent.invoke({"input": formatted_input})
        web_summary_result = summary_chain.invoke({
            'question': input_text,
            'web_results': search_response['output']
        })
        web_summary = web_summary_result['text']
    else:
        web_summary = ""

    # 合并 DeepSeek 推理结果、问题和网络摘要为单一输入变量
    combined_input = f"### 问题:\n{input_text}\n\n### 深度推理:\n{reasoning_content}\n\n### 网络信息:\n{web_summary}"

    # 创建最终答案生成链，注入对话历史
    final_prompt = PromptTemplate(
        input_variables=['chat_history', 'combined_input'],
        template=RESULT_PROMPT
    )
    chain_result = LLMChain(llm=qwen, prompt=final_prompt, memory=conversation_memory)
    response = chain_result.invoke({
        'chat_history': conversation_memory.buffer,
        'combined_input': combined_input
    })
    return response['text'], web_summary, reasoning_content


# =============================================================================
# 9. 调用生成思维导图的函数（调用外部 API）（可以选择，使用方法是将ai给出的答案文本，放入已经构建好的扣子工作流当中，实现生成回答的思维导图）
# =============================================================================
def create_mind_map(input_text: str):
    """
    调用外部 API 生成思维导图。

    Args:
        input_text (str): 用于生成思维导图的输入文本。

    Returns:
        dict: API 返回的思维导图数据。格式为{“output”：“url”}
    """
    API_URL = 'https://api.coze.cn/v1/workflow/run'
    API_KEY = "pat_lFeh61WZGRkTegrzgDe7VmMhpOZFElKT5tJwlMsdRiQJywMXUplHMLe6W65E9KEL"
    WORKFLOW_ID = "7482050366412095539"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "workflow_id": WORKFLOW_ID,
        'parameters': {'input_2': input_text},
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()['data']


# =============================================================================
# 主程序入口（示例用法）
# =============================================================================
if __name__ == "__main__":
    # 1. 初始化各个组件
    chain_cot = init_cot_chain()
    vectordb = init_vector_db()
    qa_chain = init_retrieval_qa_chain(vectordb)

    qwen = Tongyi(
        model="qwen-max",
        temperature=0.5,
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )

    search_tool, summary_chain, agent, agent_prompt = init_search_tool(qwen)

    # 示例1：调用 process_result，设 use_web_search 为 True（调用网络搜索）
    question = "动态规划的一般解法"
    final_response, web_summary, thought_text = process_result(
        question, qwen, chain_cot, summary_chain, agent, agent_prompt, use_web_search=True
    )
    print("【process_result 最终回答】")
    print(final_response)
    print("【网络搜索摘要】")
    print(web_summary)
    print("【思维链内容】")
    print(thought_text)

    # 示例2：调用 thinking_input，设 use_web_search 为 False（不调用网络搜索）
    final_response_thinking, web_summary_thinking, reasoning_content = thinking_input(
        question, qwen, summary_chain, agent, agent_prompt, use_web_search=False
    )
    print("【thinking_input 最终回答】")
    print(final_response_thinking)
    print("【网络搜索摘要】")
    print(web_summary_thinking)
    print("【DeepSeek 推理内容】")
    print(reasoning_content)
