import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain.agents import create_agent  # 只导入 create_agent，不再需要 AgentExecutor

# ==================== 配置 ====================
# 请替换为你的 DeepSeek API Key
DEEPSEEK_API_KEY = " " 

# PDF 文件路径
PDF_PATH = "product_manual.pdf"

# ==================== 初始化 LLM ====================
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0
)

# ==================== 加载并处理 PDF ====================
def load_and_split_pdf(pdf_path):
    """加载 PDF 并分块"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ 文档已加载，共 {len(documents)} 页，分成 {len(chunks)} 个文本块")
    return chunks

# ==================== 构建向量数据库 ====================
def build_vectorstore(chunks):
    """构建 FAISS 向量数据库"""
    print("正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name="D:\软件下载安装\python\代码文件\models/Xorbits/bge-small-zh-v1.5")
    print("正在构建向量索引...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✓ 向量数据库构建完成")
    return vectorstore

# ==================== 定义工具 ====================
@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。
    输入应该是一个数学表达式，例如 "3*5+2" 或 "1599-1299"。
    返回计算结果。
    """
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。不需要输入参数。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==================== 创建 Agent ====================
def create_agent_with_retriever(vectorstore):
    """创建 Agent（使用新版 create_agent API）"""
    # 创建检索器工具
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_docs",
        description="搜索产品手册中的相关信息。当用户询问产品功能、价格、规格等问题时使用这个工具。"
    )
    
    tools = [retriever_tool, calculate, get_current_time]
    
    # 创建 Agent（直接使用 create_agent，不需要 AgentExecutor）
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""你是一个智能客服助手，专门回答产品相关问题。

使用规则：
1. 当用户询问产品功能、价格、规格等问题时，使用 search_docs 工具
2. 当用户需要计算（如差价、总和）时，使用 calculate 工具
3. 当用户询问当前日期和时间时，使用 get_current_time 工具
4. 必须用中文回答用户问题
5. 回答要友好、专业、准确

请根据用户的问题，选择合适的工具来提供帮助。"""
    )
    
    print("✓ Agent 已就绪")
    return agent

# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("智能客服系统启动中...")
    print("=" * 50)
    
    # 检查 PDF 文件是否存在
    if not os.path.exists(PDF_PATH):
        print(f"错误：PDF 文件 '{PDF_PATH}' 不存在！")
        print("请确保 product_manual.pdf 文件在当前目录下")
        return
    
    # 加载并处理 PDF
    try:
        chunks = load_and_split_pdf(PDF_PATH)
    except Exception as e:
        print(f"加载 PDF 失败：{e}")
        return
    
    # 构建向量数据库
    try:
        vectorstore = build_vectorstore(chunks)
    except Exception as e:
        print(f"构建向量数据库失败：{e}")
        return
    
    # 创建 Agent
    agent = create_agent_with_retriever(vectorstore)
    
    print("\n" + "=" * 50)
    print("智能客服已就绪！输入问题开始对话，输入 'q' 退出")
    print("示例问题：")
    print("  - X5 Pro 的价格是多少？")
    print("  - 标准版和尊享版的差价是多少？")
    print("  - 现在是什么时间？")
    print("=" * 50 + "\n")
    
    # 交互式问答
    while True:
        user_input = input("\n用户：")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("再见！")
            break
        if not user_input.strip():
            continue
        
        try:
            # 新版 API：使用 messages 格式
            response = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            
            # 提取回复内容
            if "messages" in response and response["messages"]:
                # 获取最后一条消息（助手的回复）
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    answer = last_message.content
                elif isinstance(last_message, dict):
                    answer = last_message.get('content', str(last_message))
                else:
                    answer = str(last_message)
            else:
                # 如果响应格式不同，尝试直接获取输出
                answer = response.get('output', str(response))
            
            print(f"\n助手：{answer}")
            
        except Exception as e:
            print(f"\n错误：{e}")
            print("提示：如果持续出错，请检查 API Key 和网络连接")

if __name__ == "__main__":
    main()
