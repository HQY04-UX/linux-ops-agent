import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

# 加载密钥
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# ----------------------工具定义----------------------
@tool
def linux_command_explainer(command: str) -> str:
    """
    用来解释Linux命令、说明参数含义、适用场景和风险
    参数 command：要解释的linux命令，例如 df -h、top
    """
    prompt = f"""
你是云计算运维助手，解释这条Linux命令：{command}
输出固定四点：
1. 命令作用
2. 参数含义
3. 使用场景
4. 风险提示（是否会修改、删除数据）
简洁，适合课堂演示。
"""
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0
    )
    res = llm.invoke(prompt)
    return res.content

@tool
def generate_linux_command(requirement: str) -> str:
    """
    根据需求生成Linux运维命令
    参数 requirement：运维需求，比如查看内存、查找大文件
    """
    prompt = f"""
根据需求生成合适Linux命令：{requirement}
输出：命令 + 简短说明，禁止输出rm等高危删除命令
"""
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0
    )
    res = llm.invoke(prompt)
    return res.content

# 初始化大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
    temperature=0
)
tools = [linux_command_explainer, generate_linux_command]
agent = create_react_agent(llm, tools)

# ----------------------Streamlit网页界面----------------------
st.set_page_config(page_title="云计算Linux运维智能体", page_icon="🐧")
st.title("🐧 Linux运维智能体（Agent工具调用演示）")
st.markdown("智能体自动判断：解释命令 / 生成运维命令")
st.divider()

user_input = st.text_input("请输入你的问题", placeholder="例子1：解释 df -h    例子2：帮我找出占用内存最高的进程")

if user_input:
    with st.spinner("智能体思考并选择工具中..."):
        response = agent.invoke({"messages": [("user", user_input)]})
        final_answer = response["messages"][-1].content
    st.success("返回结果：")
    st.write(final_answer)

st.divider()
st.info("项目说明：本Agent具备工具调用能力，区分普通对话和运维工具调用；仅文字演示，不会真实操作服务器。")
