import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import whisper
import numpy as np
import os
from TTS.api import TTS
import soundfile as sf

# Add ffmpeg to PATH
os.environ["PATH"] += os.pathsep + r"./bin"
# Load your fine-tuned model and tokenizer
new_model = "Trained Model/Trainer/fine_tuned_gpt2_epoch_1_newdata"  # Use forward slashes for path compatibility

model = AutoModelForCausalLM.from_pretrained(new_model).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(new_model)

# Load Whisper model
whisper_model = whisper.load_model("small.en")

# Initialize Coqui TTS
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

# Set the model to evaluation mode
model.eval()

# Global variables for previous prompt and response
last_prompt = None
last_response = None

# Global chat history
chat_history = []

# Repeat phrases for TTS
repeat_phrases = ["repeat it", "say that again", "state it again", "can you repeat that", "tell me again"]

# Define response generation function
def generate_response(message):
    global last_prompt, last_response
    if not message:
        gr.Warning("Message is empty!")
        return "Please provide a message."

    model_input = tokenizer(message, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    if generated_text.startswith(message):
        generated_text = generated_text[len(message):].strip()

    last_prompt = message
    last_response = generated_text
    return generated_text

def transcribe_audio(audio):
    if audio is None:
        gr.Warning("No audio input detected!")
        return "No audio detected."

    audio_path = "temp.wav"
    sf.write(audio_path, audio[1], audio[0], format='WAV')
    result = whisper_model.transcribe(audio_path, language="en")
    os.remove(audio_path)
    return result["text"]

def speak(text):
    if not text:
        gr.Warning("No text to convert to speech!")
        return None

    audio = tts.tts(text)
    audio_np = np.array(audio)
    output_filename = "response.wav"
    sf.write(output_filename, audio_np, 22050, format='WAV')
    return output_filename

def stt_and_tts_with_text(audio):
    global last_prompt, last_response
    
    # Check if audio input is None (reset condition)
    if audio is None:
        return None, "No audio input detected!"

    transcription = transcribe_audio(audio)

    if any(phrase in transcription.lower() for phrase in repeat_phrases):
        response = last_response if last_response is not None else "No previous response available."
    else:
        response = generate_response(transcription)

    if not response:
        gr.Error("Response generation failed.")
        return None, "Response generation failed!"

    audio_output_file = speak(response)
    return audio_output_file, response

def reset_interface():
    return None, None, ""

# Chatbot Functions
def respond(message):
    global chat_history
    if not message:
        gr.Warning("Please enter a message.")
        return "", chat_history

    bot_message = generate_response(message)
    chat_history.append((message, bot_message))
    return "", chat_history

def retry_last_message():
    global chat_history
    if chat_history:
        last_message = chat_history[-1][0]
        return respond(last_message)
    gr.Warning("No previous message to retry.")
    return "", chat_history

def clear_chat():
    global chat_history
    chat_history = []  # Reset chat history
    return "", []  # Reset chatbot display

# Create the Gradio Interface with Tabs
with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.yellow, secondary_hue=gr.themes.colors.gray)) as interface:
    with gr.Tabs():
        # Tab 1 - Chatbot Interface
        with gr.Tab("💭 Chatbot 💭"):
            tiger_gif_path = "Code/TigerNav.png"
            tiger_gif_display = gr.Image(tiger_gif_path, label="Tiger", type="filepath")

            gr.Markdown("<center><span style='font-size: 28px;'> 🧭 TigerNav 🧭 </span></center>")
            gr.Markdown("<center><span style='font-size: 20px;'>Hi! I am TigerNav, a chatbot trained on GPT2-Medium Parameters. I can answer questions about your Navigational Queries in Fr. Roque Ruano Building.</center>")

            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Type your message here...")

            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                submit.click(respond, inputs=[msg], outputs=[msg, chatbot])

                retry = gr.Button("Retry", variant="primary")
                retry.click(retry_last_message, outputs=[msg, chatbot])

                clear = gr.Button("Clear", variant="primary")
                clear.click(clear_chat, outputs=[msg, chatbot])

        # Tab 2 - Voice Assistant Interface
        with gr.Tab("🎤 Virtual Assistant 🎤"):
            tiger_gif_path = "Code/TigerNav.png"
            tiger_gif_display = gr.Image(tiger_gif_path, label="Tiger", type="filepath")

            gr.Markdown("<center><span style='font-size: 28px;'> 🧭 TigerNav 🧭 </span></center>")
            gr.Markdown("<center><span style='font-size: 20px;'>Hi! I am TigerNav, a Virtual Assistant trained on GPT2-Medium Parameters. I can answer questions about your Navigational Queries in Fr. Roque Ruano Building.</center>")

            audio_input = gr.Audio(sources="microphone", type="numpy", label="Speak Here")
            audio_output = gr.Audio(label="Audio Response", type="filepath", autoplay=True)
            text_output = gr.Textbox(label="Text Response", interactive=False)

            audio_input.change(fn=stt_and_tts_with_text, inputs=audio_input, outputs=[audio_output, text_output])

            reset_button = gr.Button("Tap to Ask Again!", variant="primary")
            reset_button.click(fn=reset_interface, outputs=[audio_input, audio_output, text_output])

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
