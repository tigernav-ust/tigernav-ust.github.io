import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import whisper
import numpy as np
import os
from TTS.api import TTS
import soundfile as sf
from happytransformer import HappyTextToText, TTSettings
import random
import logging
import re
from spellchecker import SpellChecker

# Configure logging
logging.basicConfig(
    filename='User_Model_Logging\\tiger_nav.log',  # Log file name
    level=logging.INFO,         # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Add ffmpeg to PATH
os.environ["PATH"] += os.pathsep + r"./bin"

# Load model and tokenizer
new_model = "Trained Model\\ORPO\\orpo_e5"
model = AutoModelForCausalLM.from_pretrained(new_model).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(new_model)
model.eval()

# Load Whisper model
whisper_model = whisper.load_model("small.en")

# Coqui TTS
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

# Grammar Correction model
grammar_model = HappyTextToText("T5", "vennify/t5-base-grammar-correction") #vennify/t5-base-grammar-correction/prithivida/grammar_error_correcter_v1
# Set up the generation arguments
grammar_args = TTSettings(
    num_beams=5,         # Balanced beam search for a good mix of quality and speed
    min_length=1,        # Allow correction of short sentences
    max_length=50,       # Maximum length of the generated text (optional)
    temperature=1.0,     # Standard value for less randomness
    top_k=50,            # Use top-k sampling for more diversity (optional)
    top_p=1.0            # Nucleus sampling (optional)
)

# Repeat phrases
repeat_phrases = ["repeat it", "say that again", "state it again", "can you repeat that", "tell me again"]

# Global last states
last_prompt = None
last_response = None

from sentence_transformers import SentenceTransformer, util

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight but effective
# Known locations and abbreviations
locations = {
    "guard post", "lobby", "entrance", 
    "room 101", "room 102", "room 103", "room 104", "room 107", "room 108", "room 109", "room 110",
    "laboratory 1", "laboratory 1 extension", "laboratory 2 machine room", "laboratory 2 lecture room",
    "laboratory 4", "laboratory 5", "laboratory 6-1", "laboratory 6-2", "laboratory 7",
    "industrial engineering department", "industrial engineering department office"
    "industrial engineering laboratory", "industrial engineering consultation room", "industrial engineering lab",
    "department of chemical engineering", "mechanical engineering department",
    "engineering student council office", "ust student academic advising and services", "student academic advising and services", "student academic advising and services office",
    "pax romana office", "surveying instrumentation room", "left quadrangle", "right quadrangle",
    "elevator", "prayer room", "men's restroom", "women's restroom", "male toilet", "female toilet"
}

abbreviation_map = {
    "esc": "engineering student council office",
    "esc office": "engineering student council office",

    "ust saas": "ust student academic advising and services",
    "saas": "ust student academic advising and services",
    "saas office": "ust student academic advising and services",

    "pax": "pax romana office",
    "pax office": "pax romana office",

    # Department of Chemical Engineering
    "chem eng dept": "department of chemical engineering",
    "chem eng office": "department of chemical engineering",
    "chem eng department": "department of chemical engineering",
    "che dept": "department of chemical engineering",
    "che department": "department of chemical engineering",
    "dept of chem eng": "department of chemical engineering",
    "department of chem eng": "department of chemical engineering",
    "department of che": "department of chemical engineering",
    "dept of che": "department of chemical engineering",
    "chemistry department": "department of chemical engineering",
    "chem engineering department": "department of chemical engineering",

    # Mechanical Engineering Department
    "mech eng dept": "mechanical engineering department",
    "mech eng office": "mechanical engineering department",
    "mech eng department": "mechanical engineering department",
    "me dept": "mechanical engineering department",
    "me department": "mechanical engineering department",
    "dept of mech eng": "mechanical engineering department",
    "department of mech eng": "mechanical engineering department",
    "department of me": "mechanical engineering department",
    "dept of me": "mechanical engineering department",

    "civil engineering lab": "laboratory 7",
    "civil eng lab": "laboratory 7",
    "civil eng laboratory": "laboratory 7",
    "ce lab": "laboratory 7",
    "civil laboratory": "laboratory 7",
    "civil eng laboratory": "laboratory 7",
    "civil engineering laboratory": "laboratory 7",
    "department of civil engineering lab": "laboratory 7",
    "department of civil engineering laboratory": "laboratory 7",
    "dept of civil engineering lab": "laboratory 7",
    "dept of civil eng lab": "laboratory 7",
    "dept of civil eng laboratory": "laboratory 7",
    "civil eng dept lab": "laboratory 7",
    "civil eng dept laboratory": "laboratory 7",

    "chem engineering lab": "laboratory 4",
    "chem eng lab": "laboratory 4",
    "chem eng laboratory": "laboratory 4",
    "che lab": "laboratory 4", 
    "chem laboratory": "laboratory 4",
    "chem eng laboratory": "laboratory 4",
    "chem engineering laboratory": "laboratory 4",
    "department of chem engineering lab": "laboratory 4",
    "department of chem engineering laboratory": "laboratory 4",
    "dept of chem engineering lab": "laboratory 4",
    "dept of chem eng lab": "laboratory 4",
    "dept of chem eng laboratory": "laboratory 4",
    "chem eng dept lab": "laboratory 4",
    "chem eng dept laboratory": "laboratory 4",

    "Mech engineering lab": "laboratory 1 extension",
    "Mech eng lab": "laboratory 1 extension",
    "Mech eng laboratory": "laboratory 1 extension",
    "ME lab": "laboratory 1 extension",  # "ME" as an abbreviation for Mechanical Engineering
    "Mech laboratory": "laboratory 1 extension",
    "Mech eng laboratory": "laboratory 1 extension",
    "Mech engineering laboratory": "laboratory 1 extension",
    "department of Mech engineering lab": "laboratory 1 extension",
    "department of Mech engineering laboratory": "laboratory 1 extension",
    "dept of Mech engineering lab": "laboratory 1 extension",
    "dept of Mech eng lab": "laboratory 1 extension",
    "dept of Mech eng laboratory": "laboratory 1 extension",
    "Mech eng dept lab": "laboratory 1 extension",
    "Mech eng dept laboratory": "laboratory 1 extension",

    "industrial eng dept": "industrial engineering department",
    "industrial eng office": "industrial engineering department",
    "industrial eng department": "industrial engineering department",
    "ie dept": "industrial engineering department",
    "ie department": "industrial engineering department",
    "dept of industrial eng": "industrial engineering department",
    "department of industrial eng": "industrial engineering department",
    "department of ie": "industrial engineering department",
    "dept of ie": "industrial engineering department",
    "ie department office": "industrial engineering department office",
    "ie dept office": "industrial engineering department office",
    "industrial eng dept office": "industrial engineering department office",

    "ie lab": "industrial engineering lab",
    "industrial eng lab": "industrial engineering lab",

    "ie consultation room": "industrial engineering consultation room",
    "ie consult room": "industrial engineering consultation room",
    "industrial eng consult room": "industrial engineering consultation room",

    "lab 1": "laboratory 1", 
    "lab 1 extension": "laboratory 1 extension",
    "lab 1 ext": "laboratory 1 extension",
    "lab 1 next": "laboratory 1 extension",
    "laboratory 1 next": "laboratory 1 extension",


    "lab 2 machine room": "laboratory 2 machine room",
    "lab 2 mach room": "laboratory 2 machine room",
    "lab 2 lecture room": "laboratory 2 lecture room",
    "lab 2 lec room": "laboratory 2 lecture room",

    "lab 4": "laboratory 4", 
    "lab 5": "laboratory 5", 
    "lab 6-1": "laboratory 6-1",
    "lab 6-2": "laboratory 6-2", 
    "lab 61": "laboratory 6-1",
    "lab 62": "laboratory 6-2", 

    "lab 7": "laboratory 7",
    "surveying room": "surveying instrumentation room",
    "surveying instru room": "surveying instrumentation room",

    "101": "room 101",
    "102": "room 102",
    "103": "room 103",
    "104": "room 104",
    "107": "room 107",
    "108": "room 108",
    "109": "room 109",
    "110": "room 110",

    "men's toilet": "men's restroom",
    "male restroom": "men's restroom",
    "men restroom": "men's restroom",
    "mens restroom": "men's restroom",
    "male washroom": "men's restroom",
    "men's washroom": "men's restroom",
    "men toilet": "men's restroom",
    "male cr": "men's restroom",
    "male comfort room": "men's restroom",
    "mens bathroom": "men's restroom",
    "male bathroom": "men's restroom", 

    "women's toilet": "women's restroom",
    "female restroom": "women's restroom",
    "women restroom": "women's restroom",
    "womens restroom": "women's restroom",
    "female washroom": "women's restroom",
    "women's washroom": "women's restroom",
    "women toilet": "women's restroom",
    "female cr": "women's restroom",
    "female comfort room": "women's restroom",
    "female bathroom": "women's restroom",
    "women's bathroom": "women's restroom",

    "restroom male": "men's restroom",
    "restroom female": "women's restroom",
    "cr male": "men's restroom",
    "cr female": "women's restroom",
    "comfort room male": "men's restroom",
    "comfort room female": "women's restroom",
}

# Combine all valid phrases into a single lookup set
location_map = set(locations)
location_map.update(abbreviation_map.keys())

spell = SpellChecker()

def correct_spelling(text):
    words = text.split()
    corrected_text = []

    for word in words:
        # Skip correcting words with digits or hyphens
        if re.search(r'\d|-', word):
            corrected_text.append(word)
        else:
            corrected_word = spell.correction(word)
            corrected_text.append(str(corrected_word))  # Ensure it's a string
    
    return " ".join(corrected_text)

def normalize_abbreviations(text):
    # Replace abbreviations with their full forms
    for abbreviation, full_form in abbreviation_map.items():
        text = re.sub(r'\b' + re.escape(abbreviation) + r'\b', full_form, text, flags=re.IGNORECASE)
    return text

def mask_locations(text, detected_locs):
    loc_map = {}
    masked_text = text

    for i, loc in enumerate(sorted(detected_locs, key=len, reverse=True)):
        placeholder = f"__LOC_{i}__"
        loc_map[placeholder] = loc
        pattern = re.compile(re.escape(loc), re.IGNORECASE)
        masked_text = pattern.sub(placeholder, masked_text)

    return masked_text, loc_map

def restore_locations(text, loc_map):
    for placeholder, loc in loc_map.items():
        text = text.replace(placeholder, loc)
    return text

def correct_grammar(text):
    if not text.strip():
        return text

    greetings = {
    "hi", "hello", "hey", "ey", "howdy", "yo", "hiya", "sup", "heya", "greetings",
    "what's up", "wassup", "hey there", "good day", "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening", "hello there", "hi there", "yo there", "hey tiger",
    "hi assistant", "hello assistant", "hello tigernav", "tiger", "tigernav", "hey tigernav"
    }

    # Normalize and check if the text is a greeting
    if text.strip().lower() in greetings:
        text = f"{text.strip()} can you help me?"

    detected_locs = list(detect_locations_with_counts(text).keys())
    masked_text, loc_map = mask_locations(text, detected_locs)

    # Prepend the helper phrase
    modified_text = f"Help me with the directions based on this: {masked_text}"

    # Run grammar correction
    result = grammar_model.generate_text(f"grammar: {modified_text}", args=grammar_args)
    corrected_text = result.text.strip()

    # Restore original location names
    corrected_text = restore_locations(corrected_text, loc_map)
    corrected_text = remove_consecutive_duplicate_locations(corrected_text)

    # Log the results
    with open("User_Model_Logging\\grammar_corrections.txt", "a") as f:
        f.write(f"Original: {text}\nModified: {modified_text}\nCorrected (masked): {result.text.strip()}\nCorrected (final): {corrected_text}\n\n")

    logging.info(f"Grammar correction: Original: '{text}' | Modified: '{modified_text}' | Corrected: '{corrected_text}'")

    return corrected_text

def remove_consecutive_duplicate_locations(text):
    # Normalize casing and spacing (e.g., "lab1" → "lab 1")
    # Normalize spacing between letters and numbers (e.g., "lab1" → "lab 1", "room101" → "room 101")
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', text)
    text = text.lower()

    words = text.split()
    result = []
    i = 0
    last_normalized = None  # Initialize last_normalized to None

    while i < len(words):
        matched = False

        # Try to match 4-1 word phrases (for multi-word locations)
        for size in range(4, 0, -1):
            phrase = " ".join(words[i:i+size])
            if phrase in location_map:
                normalized = abbreviation_map.get(phrase, phrase)
                if normalized != last_normalized:
                    result.append(normalized)
                    last_normalized = normalized
                # Skip duplicates
                i += size
                matched = True
                break

        if not matched:
            result.append(words[i])
            last_normalized = None  # Reset if current word is not a location
            i += 1

    return " ".join(result)

def generate_model_response(input_message):
    model_input = tokenizer(input_message, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if input_message.strip().lower() in generated_text.strip().lower():
        generated_text = generated_text[len(input_message):].strip()

    for delimiter in ["Assistant:", "assistant:", "Response:", "\n"]:
        if delimiter in generated_text:
            generated_text = generated_text.split(delimiter, 1)[-1].strip()

    return generated_text

def detect_locations(text):
    text = text.lower()
    # Detect all locations in the text that are in location_map
    return [loc for loc in location_map if loc in text]

def detect_locations_with_counts(text):
    text_lower = text.lower()
    matched = {}
    used_spans = []

    # Sort locations by descending length to prioritize longer ones first
    sorted_locations = sorted(location_map, key=lambda x: -len(x))

    for loc in sorted_locations:
        pattern = re.compile(re.escape(loc), re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            start, end = match.span()

            # Check if this span overlaps with previously matched ones
            if any(us <= start < ue or us < end <= ue for us, ue in used_spans):
                continue  # Skip overlapping (i.e., substring of a longer match)

            normalized = abbreviation_map.get(loc, loc)  # Normalize to full name
            matched[normalized] = matched.get(normalized, 0) + 1
            used_spans.append((start, end))

    return matched

def generate_response(message):
    global last_prompt, last_response

    if not message:
        gr.Warning("Message is empty!")
        return "Please provide a message."
    
    message = normalize_abbreviations(message)
    message = correct_spelling(message)
    message = correct_grammar(message)

    detected_locations = detect_locations_with_counts(message)
    unique_locations = list(detected_locations.keys())

    # Log detected and unique locations
    logging.info(f"Detected locations: {detected_locations}")
    logging.info(f"Unique locations: {unique_locations}")

    # Log the count of each detected location
    for location, count in detected_locations.items():
        logging.info(f"Count of '{location}': {count}")

    same_location_responses = [
        "Assuming that you are at the {location}, you are already there!",
        "It seems you're already at {location}. No need to go anywhere!",
        "You're currently at {location}, which is your destination.",
        "You're standing right at {location} — you're there already!",
        "No need to move — you're already in {location}.",
        "You're at {location}, which is exactly where you need to be.",
        "You've reached {location} — and that's where you are!",
        "Since you're at {location}, there's no further direction needed.",
        "You're already at {location}, no need for directions.",
        "Looks like you're exactly at {location} already.",
        "You’re presently at {location}, the very place you’re asking about.",
        "There's nothing more to do — you're standing right at {location}.",
        "You're already positioned at {location}.",
        "You're already located at {location}. Mission accomplished!",
        "You’re asking for directions to {location}, but you’re already there!",
        "No steps required — you’re in {location} already.",
        "You’re right at {location} — ready for whatever comes next!",
        "You’re already within {location}, no travel necessary.",
        "You’ve already arrived at {location}!",
        "You're good — you're at {location} already!"
    ]
    
    if len(unique_locations) == 1:
        location = unique_locations[0]
        frequency = detected_locations[location]
        
        # Handle guard post/lobby/entrance directly
        if location in ["guard post", "lobby", "entrance"]:
            return random.choice(same_location_responses).format(location=location.title())

        if frequency >= 2:  # If the same location is mentioned multiple times
            return random.choice(same_location_responses).format(location=location.title())

        # If the location is mentioned once, prepare the input for the model
        model_input_message = f"How to navigate to {location} from the Guard Post?"

        # Generate response based on the new message using the new function
        generated_text = generate_model_response(model_input_message)

        last_prompt = model_input_message
        last_response = generated_text

        logging.info(f"Generated response: {generated_text} for prompt: {model_input_message}")

        return generated_text

    elif len(unique_locations) == 0 or len(unique_locations) >= 2:
        # Fallback to model (valid 2 locations or uncertain cases)
        generated_text = generate_model_response(message)

        # If no response is generated, rephrase the prompt and try again
        if not generated_text.strip():
            rephrased_message = f"Can you help with directions based on the following query: {message}"
            generated_text = generate_model_response(rephrased_message)

        logging.info(f"Generated response: {generated_text} for prompt: {message}")

        return generated_text

def transcribe_audio(audio):
    if audio is None:
        gr.Warning("No audio input detected!")
        return "No audio detected."

    audio_path = "temp.wav"
    sf.write(audio_path, audio[1], audio[0], format='WAV')
    result = whisper_model.transcribe(audio_path, language="en")
    os.remove(audio_path)

    logging.info(f"Transcribed audio to text: {result['text']}")
    return result["text"]

def speak(text):
    if not text:
        gr.Warning("No text to convert to speech!")
        return None

    audio = tts.tts(text)
    audio_np = np.array(audio)
    output_filename = "response.wav"
    sf.write(output_filename, audio_np, 22050, format='WAV')

    logging.info(f"Converted text to speech: {text}")
    return output_filename

def stt_and_tts_with_text(audio, state=None):
    global last_prompt, last_response

    if state is None:
        state = {"chat_history": []}

    if audio is None:
        return None, "No audio input detected!", state["chat_history"]

    transcription = transcribe_audio(audio)

    if any(phrase in transcription.lower() for phrase in repeat_phrases):
        response = last_response if last_response is not None else "No previous response available."
    else:
        response = generate_response(transcription)

    if not response:
        gr.Error("Response generation failed.")
        return None, "Response generation failed!", state["chat_history"]

    audio_output_file = speak(response)
    state["chat_history"].append((transcription, response))

    with open("User_Model_Logging\\va_chat_log.txt", "a") as f:
        f.write(f"User: {transcription}\nAssistant: {response}\n\n")

    return audio_output_file, response, state["chat_history"]

def respond_text(message, state):
    global last_prompt, last_response
    if not message:
        gr.Warning("Message is empty!")
        return "", state

    response = generate_response(message)
    state.append((message, response))
    logging.info(f":User  {message} | Assistant: {response}")

    with open("User_Model_Logging\\text_chat_log.txt", "a") as f:
        f.write(f"User: {message}\nAssistant: {response}\n\n")

    return "", state

def reset_interface():
    new_state = {"chat_history": []}
    logging.info("Interface reset.")
    return None, None, "", [], new_state

# ---------- GRADIO INTERFACE ----------

with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.yellow, secondary_hue=gr.themes.colors.gray)) as interface:
    with gr.Tabs():
        with gr.Tab("🎤 Virtual Assistant 🎤"):
            tiger_gif_path = "Code\TigerNav.png"
            gr.Image(tiger_gif_path, label="Tiger", type="filepath")

            gr.Markdown("<center><span style='font-size: 28px;'> 🧭 TigerNav 🧭 </span></center>")
            gr.Markdown("<center><span style='font-size: 20px;'>Hi! I am TigerNav, a Virtual Assistant trained on GPT2-Medium Parameters. I can answer questions about your Navigational Queries in Fr. Roque Ruano Building.</center>")

            audio_input = gr.Audio(sources="microphone", type="numpy", label="Speak Here")
            audio_output = gr.Audio(label="Response Audio", type="filepath", autoplay=True)
            text_output = gr.Textbox(label="Text Response", interactive=False, visible=False)
            voice_chatbot = gr.Chatbot(label="Conversation History")
            va_state = gr.State({"chat_history": []})

            audio_input.change(
                fn=stt_and_tts_with_text,
                inputs=[audio_input, va_state],
                outputs=[audio_output, text_output, voice_chatbot]
            )

            with gr.Row():
                re_record_button = gr.Button("Tap to Ask Again!", variant="primary")
                re_record_button.click(fn=lambda: None, inputs=[], outputs=[audio_input])

                reset_button = gr.Button("Clear", variant="primary")
                reset_button.click(
                    fn=reset_interface,
                    outputs=[audio_input, audio_output, text_output, voice_chatbot, va_state]
                )

        with gr.Tab("💬 Chatbot 💬"):
            tiger_gif_path = "Code\TigerNav.png"
            gr.Image(tiger_gif_path, label="Tiger", type="filepath")

            gr.Markdown("<center><span style='font-size: 28px;'> 🧭 TigerNav 🧭 </span></center>")
            gr.Markdown("<center><span style='font-size: 20px;'>Hi! I am TigerNav, a Chatbot trained on GPT2-Medium Parameters. I can answer questions about your Navigational Queries in Fr. Roque Ruano Building.</center>")

            text_chatbot = gr.Chatbot(label="Conversation History")
            text_input = gr.Textbox(placeholder="Type your question here...", label="Enter Your Question", interactive=True)
            text_output = gr.Textbox(label="Text Response", interactive=False, visible=False)
            text_state = gr.State([])

            with gr.Row():
                submit_button = gr.Button("Submit", variant="primary")
                submit_button.click(fn=respond_text, inputs=[text_input, text_state], outputs=[text_input, text_chatbot])

                text_input.submit(fn=respond_text, inputs=[text_input, text_state], outputs=[text_input, text_chatbot])

                retry_button = gr.Button("Retry", variant="primary")
                retry_button.click(fn=lambda state: respond_text(state[-1][0] if state else "", state), inputs=[text_state], outputs=[text_input, text_chatbot])

                clear_button = gr.Button("Clear", variant="primary")
                clear_button.click(fn=reset_interface, outputs=[text_input, text_output, text_chatbot, text_state])

# ---------- LAUNCH ----------
if __name__ == "__main__":
    logging.info("Launching the TigerNav application.")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
    )