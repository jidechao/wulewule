from pathlib import Path
import os
import requests
from typing import List, Dict, Any, Optional, Iterator
from PIL import Image
import re
import torch

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.core.postprocessor import LLMRerank
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from openai import OpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import streamlit as st


class PromptEngineerAgent:
    """专门用于优化提示词的代理"""
    def __init__(self, llm):
        self.llm = llm
        
    def optimize_image_prompt(self, user_input: str) -> str:
        """
        将用户的图像需求转换为优化的stable-diffusion提示词
        """
        prompt_template = f"""
        请将以下用户的图像需求转换为stable-diffusion所需的文生图提示词。
        
        用户需求: {user_input}
        
        请生成一个优化的英文提示词，格式要求：
        1. 使用详细的描述性语言
        2. 包含具体的艺术风格
        3. 说明构图和视角
        4. 描述光影和氛围
        5. 添加相关的艺术家参考或风格类型
        
        提示词:
        """
        
        response = self.llm.complete(prompt_template)
        return str(response)
    
    def optimize_voice_prompt(self, user_input: str) -> Dict[str, str]:
        """
        优化语音合成的参数
        """
        prompt_template = f"""
        请分析以下文本，并提供优化的语音合成参数。
        
        文本: {user_input}
        
        请考虑：
        1. 最适合的语言
        2. 说话的语速
        3. 语气特点
        4. 情感色彩
        
        以JSON格式返回参数：
        """
        
        response = self.llm.complete(prompt_template)
        try:
            params = eval(str(response))
            return params
        except:
            return {"lang": "zh", "speed": 1.0}


class MultiModalAssistant:
    def __init__(self, data_source_dir, llm, api_key):
        """
        初始化助手，设置必要的API密钥和加载文档
        """
        
        # 初始化LLM
        self.llm = llm
        self.__api_key = api_key
        # 初始化Prompt Engineer Agent
        self.prompt_engineer = PromptEngineerAgent(self.llm)
        
        # 加载文档并创建索引
        documents = SimpleDirectoryReader(data_source_dir, recursive=False, required_exts=[".txt"]).load_data()
        self.index = VectorStoreIndex.from_documents(
            documents
        )
        
        # 创建rag 用于回答知识问题
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)

        # 创建rag+reranker用于回答知识问题
        # self.query_engine = self.index.as_query_engine(similarity_top_k=3,
        #                                         node_postprocessors=[
        #                                         LLMRerank(
        #                                             choice_batch_size=5,
        #                                             top_n=2,
        #                                         )],
        #                                         response_mode="tree_summarize",)        
        # 设置工具
        tools = [
            FunctionTool.from_defaults(
                fn=self.rag_query,
                name="rag_tool",
                description="无法直接回答时，查询和《黑神话：悟空》有关知识的工具"
            ),
            FunctionTool.from_defaults(
                fn=self.text_to_speech,
                name="tts_tool",
                description="将文本转换为语音的工具"
            ),
            FunctionTool.from_defaults(
                fn=self.generate_image,
                name="image_tool",
                description="生成图像的工具"
            )
        ]
        
        # 初始化Agent
        self.agent = ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=True,
            max_function_calls=5,
        )

        ## 画图的url
        self.image_url = None
        self.audio_save_file = "audio.mp3"
        self.audio_text = None
        
    def rag_query(self, query: str) -> str:
        """
        使用RAG系统查询知识库
        """
        response = self.query_engine.query(query)
        return str(response)
    
    def text_to_speech(self, text: str) -> str:
        """
        将文本转换为语音
        """
        if not self.audio_text is None:
            print(f"文本已转为语音:  {self.audio_text}")
            return 
        try:
            client = OpenAI( api_key = self.__api_key, base_url="https://api.siliconflow.cn/v1")

            with client.audio.speech.with_streaming_response.create(
            model="fishaudio/fish-speech-1.5", # 目前仅支持 fishaudio 系列模型
            voice="fishaudio/fish-speech-1.5:benjamin", # 系统预置音色
            # 用户输入信息  "孙悟空身穿金色战甲，手持金箍棒，眼神锐利"
            input=f"{text}",
            response_format="mp3" # 支持 mp3, wav, pcm, opus 格式
            ) as response:
                response.stream_to_file(self.audio_save_file)
            
            if response.status_code == 200:
                self.audio_text = text
                print(f"文本已转为语音: {self.audio_save_file}")
                # return f"文本转语音已完成。"
            else:
                print("文本转语音失败，状态码：", response.status_code)
        except Exception as e:
            return f"文本转语音时出错: {str(e)}"

    def generate_image(self, prompt: str) -> str:
        """
        使用API生成图像
        """
        if not self.image_url is None:
            print(f"图像已生成:  {self.image_url}")
            return 
        try:
            # 使用Prompt Engineer优化提示词
            optimized_prompt = self.prompt_engineer.optimize_image_prompt(prompt)
            print(f"优化后的图像提示词: {optimized_prompt}")

            ## create an image of superman in a tense, action-packed scene, with explosive energy and bold dynamic composition, in the style of Ross Tran
            url = "https://api.siliconflow.cn/v1/images/generations"
            payload = {
                "model": "stabilityai/stable-diffusion-3-5-large",
                "prompt": f"{optimized_prompt}",
                "negative_prompt": "<string>",
                "image_size": "1024x1024",
                "batch_size": 1,
                "seed": 4999999999,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "prompt_enhancement": False
            }

            headers = {
                "Authorization": f"Bearer {self.__api_key}",
                "Content-Type": "application/json"
            }
            response = requests.request("POST", url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                self.image_url = data['data'][0]['url']
                print(f"图像已生成: {self.image_url}")
                # return f"图像已生成。"
                # return f"图像已生成已完成。继续下一个任务"
            else:
                print("生成图像失败，状态码：", response.status_code)

        except Exception as e:
            return f"生成图像时出错: {str(e)}"
    
    def chat(self, user_input: str) -> dict:
        """
        处理用户输入并返回适当的响应
        """
        # 创建提示来帮助agent理解如何处理不同类型的请求
        prompt = f"""
        用户输入: {user_input}
        
        请根据以下规则处理这个请求:
        1. 如果是知识相关的问题，使用rag_tool查询知识库
        2. 如果用户要求语音输出，使用tts_tool转换文本
        3. 如果用户要求生成图像，使用image_tool生成

        根据需求请选择合适的工具并执行操作，可能需要多个工具。
        """
        self.image_url = None
        self.audio_text = None
        response = self.agent.chat(prompt)
        response_dict = {"response": str(response), "image_url": self.image_url, "audio_text": self.audio_text }
        return response_dict


if __name__ == "__main__":
    ## load wulewule agent 
    wulewule_assistant = load_wulewule_agent()

    ## streamlit setting
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    # 在侧边栏中创建一个标题和一个链接
    with st.sidebar:
        st.markdown("## 悟了悟了💡")
        logo_path = "assets/sd_wulewule.webp"
        if os.path.exists(logo_path):
            image = Image.open(logo_path)
            st.image(image, caption='wulewule')
        "[InternLM](https://github.com/InternLM)"
        "[悟了悟了](https://github.com/xzyun2011/wulewule.git)"

        # 创建一个标题
    st.title("悟了悟了：黑神话悟空AI助手🐒")

    # 遍历session_state中的所有消息，并显示在聊天界面上
    for msg in st.session_state.messages:
        st.chat_message("user").write(msg["user"])
        assistant_res = msg["assistant"]
        if isinstance(assistant_res, str):
            st.chat_message("assistant").write(assistant_res)
        elif isinstance(assistant_res, dict):
            image_url = assistant_res["image_url"]
            audio_text = assistant_res["audio_text"]
            st.chat_message("assistant").write(assistant_res["response"])
            if image_url:
                # 使用st.image展示URL图像，并设置使用列宽
                st.image( image_url, width=256 )
            if audio_text:
                # 使用st.audio函数播放音频
                st.audio("audio.mp3")
                st.write(f"语音内容为: {audio_text}")


    # Get user input #你觉得悟空长啥样，按你的想法画一个
    if prompt := st.chat_input("请输入你的问题，换行使用Shfit+Enter。"):
        # Display user input
        st.chat_message("user").write(prompt)
        ## 初始化完整的回答字符串
        full_answer = ""
        with st.chat_message('robot'):
            message_placeholder = st.empty()
            response_dict = wulewule_assistant.chat(prompt)
            image_url = response_dict["image_url"]
            audio_text = response_dict["audio_text"]
            for cur_response in response_dict["response"]:
                full_answer += cur_response
                # Display robot response in chat message container
                message_placeholder.markdown(full_answer + '▌')
            message_placeholder.markdown(full_answer)
        # 将问答结果添加到 session_state 的消息历史中
        st.session_state.messages.append({"user": prompt, "assistant": response_dict})
        if image_url:
            # 使用st.image展示URL图像，并设置使用列宽
            st.image( image_url, width=256 )

        if audio_text:
            # 使用st.audio函数播放音频
            st.audio("audio.mp3")
            st.write(f"语音内容为: {audio_text}")
        torch.cuda.empty_cache()
