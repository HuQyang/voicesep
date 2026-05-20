import os
import json
import argparse
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

SYSTEM_PROMPT = """
你是一个专业的会议记录纠错助手。
下面是一段会议语音识别（ASR）生成的带有时间戳和说话人标签的逐字稿。
由于语音识别的局限性，文本中可能包含以下类型的错误：
1. 同音字/近音字错误（例如：“预值”应为“阈值”，“账单”应为“张丹”等）
2. 标点符号错误或断句不准
3. 口语化词汇（“嗯”、“啊”、“那个”）导致的语义不顺

请你在【绝对不改变原意】且【不凭空捏造新信息】的前提下：
1. 修正同音字和专有名词错误。
2. 稍微平滑语句，使其符合书面阅读习惯，但保留原说话人的语气。
3. 返回格式必须是一段纯文本，保留原始的格式 `[开始时间-结束时间] 说话人:\n正文`。

不要输出任何额外的解释或对话，直接输出修正后的完整记录。
"""

class LLMCorrector:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o"):
        """
        初始化大模型纠错器。
        如果你使用的是国内的大模型（如 DeepSeek, Qwen, Moonshot等），
        只需传入对应的 base_url 和 api_key，并将 model 设置为对应名称。
        """
        if OpenAI is None:
            raise ImportError("请先安装 openai 库: pip install openai")
            
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        
        if not api_key:
            raise ValueError("请提供 API KEY，或设置环境变量 LLM_API_KEY。")
            
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def correct_transcript(self, transcript_text: str) -> str:
        """
        调用大模型对整段或分块的文本进行纠错。
        如果文本非常长，建议在外部按段落进行切割，此处演示直接整体请求。
        """
        print(f"正在调用大模型 ({self.model}) 进行智能纠错...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请纠正以下会议记录：\n\n{transcript_text}"}
                ],
                temperature=0.1, # 保持低温度以减少幻觉
            )
            corrected_text = response.choices[0].message.content.strip()
            return corrected_text
        except Exception as e:
            print(f"大模型纠错失败: {e}")
            return transcript_text


def main():
    parser = argparse.ArgumentParser(description="使用大语言模型(LLM)修正ASR逐字稿")
    parser.add_argument("input_txt", help="原始ASR转写输出文本文件路径 (例如: test_audio.pp.txt)")
    parser.add_argument("--api-key", default="", help="LLM API Key")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="LLM Base URL (用于适配国内模型，如 https://api.deepseek.com)")
    parser.add_argument("--model", default="gpt-4o", help="模型名称 (例如: deepseek-chat)")
    parser.add_argument("--output", default="", help="保存修正后结果的文件路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_txt):
        print(f"找不到输入文件: {args.input_txt}")
        return
        
    with open(args.input_txt, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    corrector = LLMCorrector(api_key=args.api_key, base_url=args.base_url, model=args.model)
    corrected_text = corrector.correct_transcript(raw_text)
    
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input_txt)
        output_path = f"{base}_llm_corrected{ext}"
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)
        
    print(f"\n[成功] 纠错完成！")
    print(f"结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
