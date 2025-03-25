# app.py
import streamlit as st
from chat_agent import ConversationManager
from langchain.schema import HumanMessage, AIMessage


def main():
    # 页面基础设置
    st.set_page_config(
        page_title="智能教学助手",
        page_icon="📚",
        layout="centered"
    )

    # 初始化session状态（核心记忆隔离存储结构）
    ###################################################
    # session_state结构说明：
    # - current_user: 当前登录用户名（字符串）
    # - user_sessions: 用户会话存储器（字典结构）
    #   - 键：用户名
    #   - 值：用户会话数据（字典）
    #     - sessions: 会话集合（字典）
    #       - 键：会话ID（如"session_1"）
    #       - 值：会话数据（字典）
    #         - manager: ConversationManager实例
    #         - history: 对话历史列表
    #     - session_count: 该用户创建的会话总数
    # - active_session: 当前活动会话ID
    ###################################################
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'user_sessions' not in st.session_state:
        st.session_state.user_sessions = {}
    if 'active_session' not in st.session_state:
        st.session_state.active_session = None

    st.title("智能教学助手")

    # 用户认证模块（侧边栏）
    with st.sidebar:
        st.header("用户管理")
        username = st.text_input("用户名", key="username_input")

        # 登录/切换用户按钮
        if st.button("登录/切换用户"):
            if username:
                # 创建新用户数据结构（如果不存在）
                if username not in st.session_state.user_sessions:
                    st.session_state.user_sessions[username] = {
                        "sessions": {},  # 存储该用户的所有会话
                        "session_count": 0  # 该用户的会话计数器
                    }
                st.session_state.current_user = username
                st.success(f"欢迎 {username}!")

        # 已登录用户显示会话管理界面
        if st.session_state.current_user:
            st.markdown("---")

            # 新建对话按钮
            if st.button("新建对话"):
                user_sessions = st.session_state.user_sessions[st.session_state.current_user]
                # 生成新会话ID（session_1, session_2...）
                new_session_id = f"session_{user_sessions['session_count'] + 1}"
                # 为新会话创建独立存储空间
                user_sessions['sessions'][new_session_id] = {
                    "manager": ConversationManager(),  # 独立的对话管理器实例
                    "history": []  # 独立的对话历史记录
                }
                user_sessions['session_count'] += 1
                st.session_state.active_session = new_session_id
                st.rerun()  # 刷新页面加载新会话

            # 会话选择下拉框
            sessions = st.session_state.user_sessions.get(
                st.session_state.current_user, {}
            ).get("sessions", {})
            session_options = list(sessions.keys())

            # 自动选择最新会话（如果当前没有活动会话）
            selected_session = st.selectbox(
                "选择对话",
                session_options,
                index=session_options.index(st.session_state.active_session)
                if st.session_state.active_session else 0
            )

            # 切换会话时的处理
            if selected_session != st.session_state.active_session:
                st.session_state.active_session = selected_session
                st.rerun()  # 刷新页面加载选中会话

    # 主聊天界面
    if st.session_state.current_user and st.session_state.active_session:
        # 获取当前会话数据（关键的记忆隔离点）
        current_session = st.session_state.user_sessions[
            st.session_state.current_user]["sessions"][
            st.session_state.active_session]
        manager = current_session["manager"]

        # 显示对话历史（从独立manager实例读取）
        st.header("当前对话")
        for msg in manager.conversation_memory.buffer:
            # 根据消息类型渲染不同样式
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

        # 用户输入处理
        input_text = st.chat_input("请输入您的问题...")
        if input_text:
            with st.spinner("思考中..."):
                # 获取功能开关状态
                use_web = st.toggle("启用网络搜索", value=True)

                # 根据模式调用不同处理方法
                if st.session_state.get("deep_mode"):
                    response, web_summary, reasoning = manager.thinking_input(input_text, use_web)
                else:
                    response, web_summary, reasoning = manager.process_conversation(input_text, use_web)

                # 显示主回复
                with st.chat_message("assistant"):
                    st.markdown(response)

                # 显示调试信息
                with st.expander("查看推理过程"):
                    st.subheader("思维链")
                    st.markdown(reasoning)
                    st.subheader("网络摘要")
                    st.markdown(web_summary)

                # 思维导图生成（使用当前会话的独立数据）
                if st.button("生成思维导图"):
                    mind_map = manager.create_mind_map(input_text)
                    st.markdown(f"[查看思维导图]({mind_map['output']})")

        # 控制面板（侧边栏）
        with st.sidebar:
            st.markdown("---")
            st.subheader("设置")
            # 深度模式开关（存储在session_state）
            st.session_state.deep_mode = st.toggle("深度推理模式")
            # 清空当前对话按钮
            if st.button("清空当前对话"):
                manager.clear_memory()  # 仅清除当前会话的记忆
                st.rerun()

    elif st.session_state.current_user:
        st.warning("请先创建一个新对话或选择已有对话。")
    else:
        st.info("请输入用户名并登录以开始使用。")


if __name__ == "__main__":
    main()