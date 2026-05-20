import sys
import os
from api import MeetingDiarizationPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    # Initialize the pipeline
    pipeline = MeetingDiarizationPipeline()
    
    # Process the audio
    results = pipeline.process_audio(audio_file)
    
    if not results:
        print("No results generated.")
        return

    # Format the transcript and print
    formatted_transcript = pipeline.format_transcript(results)
    print(formatted_transcript)

    # Save to .txt file
    base_name = os.path.splitext(audio_file)[0]
    txt_output_path = f"{base_name}.txt"
    pipeline.save_transcript(formatted_transcript, txt_output_path)
    
    # Check for Ground Truth PDF
    pdf_path = f"{base_name}.pdf"
    if os.path.exists(pdf_path):
        print(f"\nFound Ground Truth PDF: {pdf_path}")
        print("Evaluating accuracy...")
        try:
            eval_metrics = pipeline.evaluate_accuracy(results, pdf_path)
            if eval_metrics:
                print("\n========== 评估结果 ==========")
                print(f"CER (字符错误率): {eval_metrics['cer'] * 100:.2f}%")
                print(f"正确率 (1 - CER): {eval_metrics['accuracy'] * 100:.2f}%")
                print(f"识别字数: {eval_metrics['asr_len']} | 参考字数: {eval_metrics['gt_len']}")
        except ImportError as e:
            print(f"\n[提示] 缺少评估所需的依赖包: {e}")
            print("如果在新环境运行，请确保执行: pip install pypdf jiwer")
    else:
        print(f"\n[提示] 未找到对应的参考文档 ({pdf_path})，跳过正确率评估。")

if __name__ == "__main__":
    main()
