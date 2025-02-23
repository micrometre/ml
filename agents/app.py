import requests

# Ollama API endpoint
OLLAMA_API_URL = "http://192.168.1.131:11434/api/generate"

def chat_with_ollama(prompt, model="llama3.1"):
    """
    Sends a prompt to the Ollama model and returns the response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Set to True if you want streaming responses
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}"

def main():
    print("Welcome to the Ollama AI Agent! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Get the AI's response
        ai_response = chat_with_ollama(user_input)
        print(f"AI: {ai_response}")

if __name__ == "__main__":
    main()