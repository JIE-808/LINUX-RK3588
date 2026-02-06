import ollama
from ollama import Client
import time

def continuous_chat(model_name="deepseek-r1:1.5b", host="http://127.0.0.1:11434"):
    """
    DeepSeek多轮流式对话核心逻辑[7](@ref)
    
    参数：
    model_name -- 模型标识符 (默认：deepseek-r1:7b)
    host -- Ollama服务地址 (默认本地11434端口)
    """
    client = Client(host=host)
    messages = []  # 对话历史存储器[8](@ref)
    
    try:
        while True:
            # 用户输入环节
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("对话已终止")
                break
                
            # 添加用户消息到历史
            messages.append({"role": "user", "content": user_input})
            
            # 流式请求初始化[5](@ref)
            print("\nDeepSeek: ", end="", flush=True)
            full_response = ""
            stream = client.chat(
                model=model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.3,  # 严谨性设置[5](@ref)
                    "num_ctx": 4096      # 上下文窗口
                }
            )
            
            # 流式响应处理
            start_time = time.time()
            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    print(content, end="", flush=True)  # 实时流式输出
                    full_response += content
                time.sleep(0.01)  # 控制输出节奏
            
            # 记录响应数据
            end_time = time.time()
            messages.append({"role": "assistant", "content": full_response})
            
            # 显示性能指标
            print(f"\n\n[本次响应耗时：{end_time-start_time:.2f}s]")
            print(f"[当前上下文长度：{len(messages)}轮]")
            
    except KeyboardInterrupt:
        print("\n检测到中断信号，正在安全退出...")
    except Exception as e:
        print(f"\n发生异常：{str(e)}")
    finally:
        print("\n对话历史已保存，可导出messages变量进行后续分析")

if __name__ == "__main__":
    # 示例调用（可替换为其他deepseek模型）确保你的ollama中部署了该大模型才可以调用
    continuous_chat(model_name="deepseek-r1:1.5b")
