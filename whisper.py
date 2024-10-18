import torch
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration, logging
import os
import numpy as np
from tqdm import tqdm
import warnings
import srt
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import pyaudio
import queue
import logging
from transformers import logging as transformers_logging
import sounddevice as sd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

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
        if self.stop_flag:
            print("Transcription stopped by user.")
            break

        print(f"Processing chunk {i+1}/{total_chunks} on {device}") 

        chunk_name = f"temp_chunk_{i}.wav"
        chunk = chunk.set_frame_rate(16000)  
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

class WhisperGUI:
    def __init__(self, master):
        self.master = master
        master.title("Whisper Transcription")
        master.geometry("500x600")  

        self.label = tk.Label(master, text="Choose transcription mode:")
        self.label.pack(pady=10)

        self.mode_frame = tk.Frame(master)
        self.mode_frame.pack(pady=10)

        self.realtime_button = tk.Button(self.mode_frame, text="Realtime Transcription", command=self.setup_realtime_mode)
        self.realtime_button.pack(side=tk.LEFT, padx=5)

        self.file_button = tk.Button(self.mode_frame, text="File Transcription", command=self.setup_file_mode)
        self.file_button.pack(side=tk.LEFT, padx=5)

        self.content_frame = tk.Frame(master)
        self.content_frame.pack(pady=10, expand=True, fill=tk.BOTH)

        self.file_path = None
        self.output_dir = None
        self.stop_flag = False
        self.transcription_thread = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.transcription_thread = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.language_var = tk.StringVar(value="English")
        self.languages = {
            "English": "en", "Chinese": "zh", "Spanish": "es", "French": "fr",
            "German": "de", "Japanese": "ja", "Korean": "ko", "Russian": "ru"
        }

        self.model_var = tk.StringVar(value="openai/whisper-base")
        self.models = ["openai/whisper-tiny", "openai/whisper-base", 
                       "openai/whisper-small", "openai/whisper-medium", 
                       "openai/whisper-large"]

        self.input_devices = sd.query_devices()
        self.input_device_ids = [d['index'] for d in self.input_devices if d['max_input_channels'] > 0]
        self.input_device_names = [f"{d['name']} (ID: {d['index']})" for d in self.input_devices if d['max_input_channels'] > 0]
        self.input_device_var = tk.StringVar(value=self.input_device_names[0] if self.input_device_names else "No input devices found")

        self.input_device_id = None

        master.protocol("WM_DELETE_WINDOW", self.destroy)

    def setup_realtime_mode(self):
        self.clear_content_frame()
        
        self.language_label = tk.Label(self.content_frame, text="Select language:")
        self.language_label.pack(pady=5)
        self.language_dropdown = ttk.Combobox(self.content_frame, textvariable=self.language_var, 
                                              values=list(self.languages.keys()))
        self.language_dropdown.pack(pady=5)

        self.model_label = tk.Label(self.content_frame, text="Select Whisper model:")
        self.model_label.pack(pady=5)
        self.model_dropdown = ttk.Combobox(self.content_frame, textvariable=self.model_var, 
                                           values=self.models)
        self.model_dropdown.pack(pady=5)

        self.mic_label = tk.Label(self.content_frame, text="Select microphone:")
        self.mic_label.pack(pady=5)
        self.mic_dropdown = ttk.Combobox(self.content_frame, textvariable=self.input_device_var, 
                                         values=self.input_device_names)
        self.mic_dropdown.pack(pady=5)
        self.mic_dropdown.bind("<<ComboboxSelected>>", self.update_input_device_id)

        self.realtime_text = tk.Text(self.content_frame, height=10, width=50)
        self.realtime_text.pack(pady=10)

        self.toggle_button = tk.Button(self.content_frame, text="Start Realtime Transcription", command=self.toggle_realtime_transcription)
        self.toggle_button.pack(pady=5)

    def setup_file_mode(self):
        self.clear_content_frame()
        
        self.select_button = tk.Button(self.content_frame, text="Select File", command=self.select_file)
        self.select_button.pack(pady=5)

        self.language_label = tk.Label(self.content_frame, text="Select language:")
        self.language_label.pack(pady=5)
        self.language_dropdown = ttk.Combobox(self.content_frame, textvariable=self.language_var, 
                                              values=list(self.languages.keys()))
        self.language_dropdown.pack(pady=5)

        self.model_label = tk.Label(self.content_frame, text="Select Whisper model:")
        self.model_label.pack(pady=5)
        self.model_dropdown = ttk.Combobox(self.content_frame, textvariable=self.model_var, 
                                           values=self.models)
        self.model_dropdown.pack(pady=5)

        self.output_button = tk.Button(self.content_frame, text="Select Output Directory", command=self.select_output_dir)
        self.output_button.pack(pady=5)

        self.transcribe_button = tk.Button(self.content_frame, text="Transcribe", command=self.start_transcription, state=tk.DISABLED)
        self.transcribe_button.pack(pady=5)

        self.stop_button = tk.Button(self.content_frame, text="Stop", command=self.stop_transcription, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.progress_label = tk.Label(self.content_frame, text="")
        self.progress_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(self.content_frame, length=300, mode='determinate')
        self.progress_bar.pack(pady=5)

    def clear_content_frame(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.mp4 *.wav")])
        if self.file_path:
            self.transcribe_button['state'] = tk.NORMAL
            self.label.config(text=f"Selected file: {self.file_path}")

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            self.output_button.config(text=f"Output: {self.output_dir}")

    def start_transcription(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a file first.")
            return
        if not self.output_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        self.transcribe_button['state'] = tk.DISABLED
        self.stop_button['state'] = tk.NORMAL
        self.progress_label.config(text="Transcription in progress...")
        self.stop_flag = False

        self.transcription_thread = threading.Thread(target=self.run_transcription)
        self.transcription_thread.start()

    def stop_transcription(self):
        self.stop_flag = True
        self.stop_button['state'] = tk.DISABLED
        self.progress_label.config(text="Stopping transcription...")

    def run_transcription(self):
        try:
            result, subtitles = self.process_audio_in_chunks(self.file_path)
            if not self.stop_flag:
                save_transcript(result, os.path.join(self.output_dir, "transcript.txt"))
                save_srt(subtitles, os.path.join(self.output_dir, "subtitles.srt"))
                self.master.after(0, self.show_completion)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.master.after(0, self.reset_ui)

    def process_audio_in_chunks(self, file_path, chunk_length_ms=60000):
        audio = AudioSegment.from_file(file_path, format="mp4")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Loading Whisper model...")
        processor = WhisperProcessor.from_pretrained(self.model_var.get())
        model = WhisperForConditionalGeneration.from_pretrained(self.model_var.get()).to(device)
        print("Model loaded successfully.")

        full_transcript = ""
        srt_subtitles = []
        total_chunks = len(audio) // chunk_length_ms + (1 if len(audio) % chunk_length_ms else 0)
        
        print(f"Total audio length: {len(audio) / 1000:.2f} seconds")
        print(f"Processing {total_chunks} chunks...")

        for i, chunk in enumerate(tqdm(audio[::chunk_length_ms], total=total_chunks, desc="Processing chunks")):
            if self.stop_flag:
                print("Transcription stopped by user.")
                break

            # Update progress bar
            progress = (i + 1) / total_chunks * 100
            self.master.after(0, lambda p=progress: self.update_progress(p))

            print(f"Processing chunk {i+1}/{total_chunks} on {device}")  

            chunk_name = f"temp_chunk_{i}.wav"
            chunk = chunk.set_frame_rate(16000)  
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
                    language=self.languages[self.language_var.get()],  
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

    def show_completion(self):
        messagebox.showinfo("Transcription Complete", f"Transcription has been saved to {self.output_dir}")

    def update_progress(self, value):
        self.progress_bar['value'] = value
        self.progress_label.config(text=f"Progress: {value:.1f}%")

    def reset_ui(self):
        self.transcribe_button['state'] = tk.NORMAL
        self.stop_button['state'] = tk.DISABLED
        self.progress_label.config(text="")
        self.progress_bar['value'] = 0

    def toggle_realtime_transcription(self):
        if not self.is_recording:
            self.start_realtime_transcription()
        else:
            self.stop_realtime_transcription()

    def start_realtime_transcription(self):
        self.logger.info("Starting realtime transcription")
        self.is_recording = True
        self.toggle_button.config(text="Stop Realtime Transcription")
        self.realtime_text.delete('1.0', tk.END)  
        
        while not self.audio_queue.empty():
            self.audio_queue.get()

        self.recording_thread = threading.Thread(target=self.record_audio)
        self.transcription_thread = threading.Thread(target=self.transcribe_audio)
        
        self.recording_thread.start()
        self.transcription_thread.start()
        
        self.logger.info("Realtime transcription threads started")

    def stop_realtime_transcription(self):
        self.logger.info("Stopping realtime transcription")
        self.is_recording = False
        self.toggle_button.config(text="Start Realtime Transcription")
        
        if self.recording_thread:
            self.recording_thread.join()
        if self.transcription_thread:
            self.transcription_thread.join()
        
        self.logger.info("Realtime transcription threads stopped")

    def record_audio(self):
        self.logger.info("Starting audio recording")
        chunk = 1024
        format = pyaudio.paFloat32
        channels = 1
        rate = 16000

        p = pyaudio.PyAudio()

        try:
            stream = p.open(format=format,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=self.input_device_id,
                            frames_per_buffer=chunk)

            while self.is_recording:
                data = stream.read(chunk)
                self.audio_queue.put(data)

        except Exception as e:
            self.logger.error(f"Error in audio recording: {e}")

        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()
            self.logger.info("Audio recording stopped")

    def transcribe_audio(self):
        self.logger.info("Starting audio transcription")
        buffer = np.array([], dtype=np.float32)

        while self.is_recording or (not self.audio_queue.empty() and self.is_recording):
            while not self.audio_queue.empty() and self.is_recording:
                data = self.audio_queue.get()
                buffer = np.append(buffer, np.frombuffer(data, dtype=np.float32))

            if len(buffer) >= 16000 and self.is_recording:  
                self.logger.info("Processing 1 second of audio")
                audio_chunk = buffer[:16000]
                buffer = buffer[16000:]

                try:
                    with torch.no_grad():
                        input_features = self.processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
                        predicted_ids = self.model.generate(
                            input_features, 
                            language=self.languages[self.language_var.get()],  
                            task="transcribe",
                            max_length=448
                        )
                        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                    if transcription.strip():  
                        self.logger.info(f"Transcription: {transcription}")
                        self.master.after(0, lambda t=transcription: self.update_realtime_text(t))
                except Exception as e:
                    self.logger.error(f"Error in transcription: {e}")

        self.logger.info("Audio transcription stopped")

    def update_realtime_text(self, text):
        self.logger.info(f"Updating realtime text: {text}")
        self.realtime_text.insert(tk.END, text + " ")
        self.realtime_text.see(tk.END)

    def destroy(self):
        if self.is_recording:
            self.stop_realtime_transcription()

        if hasattr(self, 'recording_thread') and self.recording_thread is not None:
            self.recording_thread.join(timeout=1)
        if hasattr(self, 'transcription_thread') and self.transcription_thread is not None:
            self.transcription_thread.join(timeout=1)

        self.master.destroy()

    def update_input_device_id(self, event):
        selected_device = self.input_device_var.get()
        device_id = int(selected_device.split("(ID: ")[1].split(")")[0])
        self.input_device_id = device_id
        print(f"Selected input device ID: {self.input_device_id}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = WhisperGUI(root)
    root.mainloop()
