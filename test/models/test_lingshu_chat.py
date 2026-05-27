import os

from ours.models.lingshu import LingshuChat



def test_lingshu_chat():

    model = LingshuChat(model_id="lingshu-32b")
    # 构造输入消息（图片 + 文本）
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical imaging assistant. "
                "Please provide clinically accurate, concise descriptions, "
                "and do NOT hallucinate findings that are not obvious."
            )
        },
        {
            "role": "user",
            "content": {
                "text": "请帮我描述这张胸片中的主要异常。",
                "image_path": "./test/testimage.png"  # 换成你本地图片路径
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
    print("内容:", response)
    print("====================")


if __name__ == "__main__":
    test_lingshu_chat()