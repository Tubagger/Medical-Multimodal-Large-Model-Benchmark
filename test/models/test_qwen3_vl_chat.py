import os

# ✅ 改这里：导入你刚写的 Qwen3vlChat
from ours.models.qwen3_vl import Qwen3vlChat


def test_qwen3vl_chat():

    # ✅ 模型名要和你 registry 里一致
    model = Qwen3vlChat(model_id="qwen3-vl-32b-instruct")

    # =========================
    # 构造输入（图 + 文）
    # =========================
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical imaging assistant. "
                "Provide accurate and concise descriptions. "
                "Do not hallucinate findings."
            )
        },
        {
            "role": "user",
            "content": {
                "text": "请帮我描述这张胸片中的主要异常。",
                "image_path": "./test/testimage.png"   # ⚠️ 确保路径存在
            }
        }
    ]

    # =========================
    # 推理
    # =========================
    response = model.chat(
        messages,
        temperature=0.0,          # ✅ VL建议0更稳定
        max_new_tokens=256        # ⚠️ 不要超过256
    )

    print("===== 模型输出 =====")
    print(response)
    print("====================")


if __name__ == "__main__":
    test_qwen3vl_chat()