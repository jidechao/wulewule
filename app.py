import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import streamlit as st
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(__file__))
from download_models import download_model


@st.cache_resource
def load_simple_rag(config, used_lmdeploy=False):
    ## load config
    data_source_dir = config["data_source_dir"]
    db_persist_directory = config["db_persist_directory"]
    llm_model = config["llm_model"]
    embeddings_model = config["embeddings_model"]
    reranker_model = config["reranker_model"]
    llm_system_prompt = config["llm_system_prompt"]
    rag_prompt_template = config["rag_prompt_template"]
    from rag.simple_rag import  WuleRAG

    if not used_lmdeploy:
        from rag.simple_rag import InternLM, WuleRAG
        base_mode = InternLM(model_path=llm_model, llm_system_prompt=llm_system_prompt)
    else:
        from deploy.lmdeploy_model import LmdeployLM, GenerationConfig
        cache_max_entry_count = config.get("cache_max_entry_count", 0.2)
        base_mode = LmdeployLM(model_path=llm_model, llm_system_prompt=llm_system_prompt, cache_max_entry_count=cache_max_entry_count)
    
    ## loda final rag model
    wulewule_rag = WuleRAG(data_source_dir, db_persist_directory, base_mode, embeddings_model, reranker_model, rag_prompt_template)
    return wulewule_rag


@st.cache_resource
def load_wulewule_agent(config):
    from agent.wulewule_agent import MultiModalAssistant, Settings
    use_remote = config["use_remote"]
    SiliconFlow_api = config["SiliconFlow_api"]
    data_source_dir = config["data_source_dir"]
    if len(SiliconFlow_api)<51 and os.environ.get('SiliconFlow_api', ""):
        SiliconFlow_api = os.environ.get('SiliconFlow_api')

    print(f"======= loading llm =======")
    if use_remote:
        from llama_index.llms.siliconflow import SiliconFlow
        from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
        api_base_url =  "https://api.siliconflow.cn/v1/chat/completions"
        # model = "Qwen/Qwen2.5-72B-Instruct"
        # model = "deepseek-ai/DeepSeek-V2.5"
        remote_llm = config["remote_llm"]
        remote_embeddings_model = config["remote_embeddings_model"]
        llm = SiliconFlow( model=remote_llm, base_url=api_base_url, api_key=SiliconFlow_api,  max_tokens=4096)
        embed_model = SiliconFlowEmbedding(  model=remote_embeddings_model, api_key=SiliconFlow_api)
    else:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface import HuggingFaceLLM
        local_llm = config["llm_model"]
        local_embeddings_model = config["agent_embeddings_model"]
        llm = HuggingFaceLLM(
            model_name=local_llm,
            tokenizer_name=local_llm,
            model_kwargs={"trust_remote_code":True},
            tokenizer_kwargs={"trust_remote_code":True},
            # context_window=4096,
            # max_new_tokens=4096,
        )
        embed_model = HuggingFaceEmbedding(
            model_name=local_embeddings_model
        )
    # settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    wulewule_assistant = MultiModalAssistant(data_source_dir, llm, SiliconFlow_api)
    print(f"======= finished loading ! =======")
    return wulewule_assistant


GlobalHydra.instance().clear()
@hydra.main(version_base=None, config_path="./configs", config_name="model_cfg")
def main(cfg):
    # omegaconf.dictcfg.DictConfig 转换为普通字典
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    ## download model from modelscope
    if not config_dict["use_remote"] and not os.path.exists(config_dict["llm_model"]):
        download_model(llm_model_path =config_dict["llm_model"])
    
    ## agent mode, used llama-index, rturn off lmdeloy and chroma rag
    if cfg.agent_mode:
        ## load wulewule agent 
        wulewule_assistant = load_wulewule_agent(config_dict) 
        cfg.use_rag = False
        cfg.use_lmdepoly = False

    if cfg.use_rag:
        ## load rag model
        wulewule_model = load_simple_rag(config_dict, used_lmdeploy=cfg.use_lmdepoly)
    elif ( cfg.use_lmdepoly):
        ## load lmdeploy model
        from deploy.lmdeploy_model import load_turbomind_model, GenerationConfig
        wulewule_model = load_turbomind_model(config_dict["llm_model"], config_dict["llm_system_prompt"], config_dict["cache_max_entry_count"])

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
        elif cfg.agent_mode and isinstance(assistant_res, dict):
            image_url = assistant_res["image_url"]
            audio_text = assistant_res["audio_text"]
            st.chat_message("assistant").write(assistant_res["response"])
            if image_url:
                # 使用st.image展示URL图像，并设置使用列宽
                st.image( image_url, width=256 )
            if audio_text:
                # 使用st.audio函数播放音频
                st.audio("audio.mp3")
                st.write(f"语音内容为: \n\n{audio_text}")

    # Get user input
    if prompt := st.chat_input("请输入你的问题，换行使用Shfit+Enter。"):
        # Display user input
        st.chat_message("user").write(prompt)
        ## 初始化完整的回答字符串
        full_answer = ""
        if cfg.agent_mode:
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
                st.write(f"语音内容为: \n\n{audio_text}")

        # 流式显示, used streaming result
        else:
            if cfg.stream_response:
                # rag
                with st.chat_message('robot'):
                    message_placeholder = st.empty()
                    if cfg.use_rag:
                        for cur_response in wulewule_model.query_stream(prompt):
                            full_answer += cur_response
                            # Display robot response in chat message container
                            message_placeholder.markdown(full_answer + '▌')
                    elif cfg.use_lmdepoly:
                        # gen_config = GenerationConfig(top_p=0.8,
                        #             top_k=40,
                        #             temperature=0.8,
                        #             max_new_tokens=2048,
                        #             repetition_penalty=1.05)
                        messages = [{'role': 'user', 'content': f'{prompt}'}]
                        for response in wulewule_model.stream_infer(messages):
                            full_answer += response.text
                            # Display robot response in chat message container
                            message_placeholder.markdown(full_answer + '▌')

                    message_placeholder.markdown(full_answer)
            # 一次性显示结果
            else:
                if cfg.use_lmdepoly:
                        messages = [{'role': 'user', 'content': f'{prompt}'}]
                        full_answer = wulewule_model(messages).text
                elif cfg.use_rag:         
                    full_answer = wulewule_model.query(prompt)
                # 显示回答
                st.chat_message("assistant").write(full_answer)

            # 将问答结果添加到 session_state 的消息历史中
            st.session_state.messages.append({"user": prompt, "assistant": full_answer})


if __name__ == "__main__":
    main()