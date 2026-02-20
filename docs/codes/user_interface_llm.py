import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load your fine-tuned model and tokenizer
new_model = "Trained Model/ORPO/finetuned_ORPO_Epoch_5"  # Use forward slashes for path compatibility
model = AutoModelForCausalLM.from_pretrained(new_model).to("cuda")  # Ensure the model is on GPU
tokenizer = AutoTokenizer.from_pretrained(new_model)

# Set the model to evaluation mode
model.eval()

# Define the function to generate responses
def generate_response(message):
    model_input = tokenizer(message, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True  # Enable sampling to use temperature and top_p
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Optional: Strip the input prompt from the response if it appears at the beginning
    if generated_text.startswith(message):
        generated_text = generated_text[len(message):].strip()
    
    return generated_text

# Create the Gradio ChatInterface
with gr.Blocks() as interface:
    gr.Markdown("<center><span style='font-size: 24px;'>🚗 TigerNav 🚗</span></center>")
    gr.Markdown("<center>Enter a prompt to get navigation directions.</center>")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message here...")

    chat_history = []

    # Define the submission function
    def respond(message):
        bot_message = generate_response(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    # Define the retry function
    def retry_last_message():
        if chat_history:
            last_message = chat_history[-1][0]  # Get the last user message
            return respond(last_message)
        return "", chat_history

    # Arrange buttons in a single row
    with gr.Row():
        submit = gr.Button("Submit")
        submit.click(respond, inputs=[msg], outputs=[msg, chatbot])

        retry = gr.Button("Retry")
        retry.click(retry_last_message, outputs=[msg, chatbot])

        clear = gr.ClearButton([msg, chatbot])

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)  # You can change the port if needed
