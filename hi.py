from openai import OpenAI

client = OpenAI(
    
    api_key="sk-BgIgfOlprX76xLOK496b3e9fB1C94a67A25dD34611F102Cf",
    
    base_url="https://chat.cloudapi.vip/v1"
)

# 测试API
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("API 连接成功:", response)
except Exception as e:
    print("API 连接错误:", e)