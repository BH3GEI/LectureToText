import torch
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration, logging
import os
import numpy as np
from tqdm import tqdm
import warnings
import srt
import datetime

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def process_audio_in_chunks(file_path, chunk_length_ms=60000):  
    audio = AudioSegment.from_file(file_path, format="mp4")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to(device)
    print("Model loaded successfully.")

    full_transcript = ""
    srt_subtitles = []
    total_chunks = len(audio) // chunk_length_ms + (1 if len(audio) % chunk_length_ms else 0)
    
    print(f"Total audio length: {len(audio) / 1000:.2f} seconds")
    print(f"Processing {total_chunks} chunks...")

    for i, chunk in enumerate(tqdm(audio[::chunk_length_ms], total=total_chunks, desc="Processing chunks")):
        chunk_name = f"temp_chunk_{i}.wav"
        chunk = chunk.set_frame_rate(16000)  # 重新采样到 16000 Hz
        chunk.export(chunk_name, format="wav")
        
        audio_data = AudioSegment.from_wav(chunk_name)
        samples = np.array(audio_data.get_array_of_samples())
        
        if audio_data.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)
        
        samples = samples / np.max(np.abs(samples))
        
        with torch.no_grad():
            input_features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            predicted_ids = model.generate(
                input_features, 
                language="zh",  
                task="transcribe",
                no_repeat_ngram_size=3,  
                num_beams=5,  
                max_length=448  
            )
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        full_transcript += transcription[0] + " "
        
        start_time = datetime.timedelta(milliseconds=i * chunk_length_ms)
        end_time = start_time + datetime.timedelta(milliseconds=chunk_length_ms)
        srt_subtitles.append(srt.Subtitle(index=i+1, start=start_time, end=end_time, content=transcription[0]))
        
        os.remove(chunk_name)  

    print("\nTranscription complete!")
    return full_transcript, srt_subtitles

def save_transcript(transcript, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

def save_srt(subtitles, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subtitles))

if __name__ == "__main__":
    print("Starting audio transcription...")
    input_file = "Zoom Meeting 2024-10-14 18-02-03.mp4"
    result, subtitles = process_audio_in_chunks(input_file)
    
    save_transcript(result, "transcript.txt")
    print("Full transcript saved to transcript.txt")
    
    save_srt(subtitles, "subtitles.srt")
    print("SRT subtitles saved to subtitles.srt")
    
    print("\nFinal Transcript:")
    print(result[:1000] + "...") 
