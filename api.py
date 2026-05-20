import os
import re
from funasr import AutoModel

try:
    import pypdf
    import jiwer
except ImportError:
    pypdf = None
    jiwer = None

class MeetingDiarizationPipeline:
    def __init__(self):
        print("Initializing MeetingDiarizationPipeline...")
        self.model = AutoModel(
            model="paraformer-zh",
            model_revision="v2.0.4",
            vad_model="fsmn-vad",
            vad_model_revision="v2.0.4",
            punc_model="ct-punc",
            punc_model_revision="v2.0.4",
            spk_model="cam++",
            spk_model_revision="v2.0.2",
        )
        print("Pipeline initialized successfully.")

    def process_audio(self, audio_path: str) -> list:
        """
        Process the given audio file and return diarized transcripts.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Processing audio: {audio_path}")
        res = self.model.generate(input=audio_path, batch_size_s=300)
        
        results = []
        if not res:
            return results
            
        for item in res:
            text = item.get("text", "")
            sentence_info = item.get("sentence_info", [])
            
            if not sentence_info:
                results.append({"speaker": "Unknown", "text": text, "start": 0, "end": 0})
                continue
                
            for sentence in sentence_info:
                start = sentence.get("start", 0) / 1000.0
                end = sentence.get("end", 0) / 1000.0
                spk = sentence.get("spk", "Unknown")
                s_text = sentence.get("text", "")
                
                results.append({
                    "speaker": spk,
                    "text": s_text,
                    "start": start,
                    "end": end
                })
                
        return results

    def format_transcript(self, results: list) -> str:
        """
        Format the raw results into a readable transcript.
        Consecutive sentences from the same speaker are merged into one block.
        """
        transcript = "========== 会议记录 ==========\n\n"
        if not results:
            return transcript

        # Merge consecutive same-speaker sentences
        merged = []
        current = {
            "speaker": results[0]["speaker"],
            "start": results[0]["start"],
            "end": results[0]["end"],
            "text": results[0]["text"],
        }

        for item in results[1:]:
            if item["speaker"] == current["speaker"]:
                # Same speaker: extend the time range and append text
                current["end"] = item["end"]
                current["text"] += item["text"]
            else:
                # Different speaker: save current block, start a new one
                merged.append(current)
                current = {
                    "speaker": item["speaker"],
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                }
        merged.append(current)

        # Format merged blocks
        for block in merged:
            s = block["start"]
            e = block["end"]
            ts = f"[{int(s // 60):02d}:{int(s % 60):02d}-{int(e // 60):02d}:{int(e % 60):02d}]"
            transcript += f"{ts} Speaker {block['speaker']}:\n{block['text']}\n\n"

        return transcript

    def save_transcript(self, transcript_text: str, output_path: str):
        """
        Save the transcript to a text file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"Transcript saved to {output_path}")

    def evaluate_accuracy(self, asr_results: list, pdf_path: str) -> dict:
        """
        Evaluate the accuracy (CER) of the ASR results against a Ground Truth PDF.
        """
        if not pypdf or not jiwer:
            raise ImportError("Please install pypdf and jiwer to use evaluation features: pip install pypdf jiwer")
            
        print(f"Extracting Ground Truth from {pdf_path}...")
        try:
            reader = pypdf.PdfReader(pdf_path)
            gt_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None
            
        # Extract pure text from ASR results
        asr_text = "".join([item["text"] for item in asr_results])
        
        # Normalize function: remove spaces and all punctuations
        def normalize_zh_text(text):
            # remove all non-alphanumeric/non-chinese characters
            text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
            return text

        gt_norm = normalize_zh_text(gt_text)
        asr_norm = normalize_zh_text(asr_text)
        
        if not gt_norm:
            print("Warning: Ground Truth PDF is empty after text extraction.")
            return None
            
        # For jiwer, it calculates word error rate by default. 
        # For Chinese character error rate, we can split text into individual characters (separated by space)
        gt_chars = " ".join(list(gt_norm))
        asr_chars = " ".join(list(asr_norm))
        
        cer = jiwer.cer(gt_chars, asr_chars)
        accuracy = max(0, 1.0 - cer)
        
        return {
            "cer": cer,
            "accuracy": accuracy,
            "gt_len": len(gt_norm),
            "asr_len": len(asr_norm)
        }


if __name__ == "__main__":
    import os
    import sys
    
    pipeline = MeetingDiarizationPipeline()
    audio_file = "test_audio.wav"
    if os.path.exists(audio_file):
        res = pipeline.process_audio(audio_file)
        formatted = pipeline.format_transcript(res)
        print("--- Original Transcript ---")
        print(formatted)
        
        # Optional: LLM Correction
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            try:
                from llm_correction import LLMCorrector
                print("\n--- Applying LLM Correction ---")
                base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
                model = os.getenv("LLM_MODEL", "gpt-4o")
                corrector = LLMCorrector(api_key=api_key, base_url=base_url, model=model)
                corrected_text = corrector.correct_transcript(formatted)
                print("\n--- Corrected Transcript ---")
                print(corrected_text)
                pipeline.save_transcript(corrected_text, "test_audio_corrected.txt")
            except ImportError:
                print("LLM corrector not available. Please install openai: pip install openai")
        else:
            print("\nHint: Set LLM_API_KEY environment variable to enable automatic LLM post-correction for better verbatim accuracy.")
            pipeline.save_transcript(formatted, "test_audio_raw.txt")
