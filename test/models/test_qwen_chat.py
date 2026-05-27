import os
from ours.models import QwenChat


def test_qwen_chat():
    # 创建模型实例（默认加载 qwen2.5-vl-32b-instruct）
    model = QwenChat(model_id="qwen3-vl-32b-instruct")



    # 构造输入消息（图片 + 文本）
    messages = [
        {   "role": "system",
            "content": {
                "Instruction:Only output the final answer option (e.g., A, B, C, or D).Do not provide explanations."
            },
            "role": "user",
            "content": {
                "text": "Please choose one option you like from the images above. Give me your answer and tell me why.",
                "image_path": "merged.png"  # 换成你本地图片路径
            }
        }
    ]

    # 生成回答
    response = model.chat(
        messages,
        temperature=0.7,
        max_new_tokens=256
    )

    print("===== 模型输出 =====")
    print("模型ID:", response.model_id)
    print("内容:", response.content)
    print("结束原因:", response.finish_reason)
    print("====================")


if __name__ == "__main__":
    test_qwen_chat()
