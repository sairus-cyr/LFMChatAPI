# LFMChatAPI

This repository contains a simple chatbot API built with **FastAPI** and the **LiquidAI/LFM2-1.2B** causal language model from Hugging Face transformers. The API allows you to send a chat prompt and receive a model-generated reply, along with response time metrics.

---

## Features

- Lightweight FastAPI backend  
- Integration with LiquidAI’s LFM2-1.2B causal language model  
- `/chat/` POST endpoint for chatbot interactions  
- Response time measurement for performance insights  
- JSON-based input/output for easy integration  

---

## Requirements

- Python 3.8+  
- `transformers` library  
- `fastapi`  
- `uvicorn`  

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-api.git
   cd chatbot-api
   ```
2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```
3. Install dependencies:
    ```bash
    pip install fastapi uvicorn transformers torch
    ```

## Usage
Run the FastAPI server:
```bash
uvicorn main:app --reload
The API will be available at http://127.0.0.1:8000.
```

## API Endpoints
### POST /chat/
Send a chat prompt and receive the chatbot response.

- Request Body (JSON):
```json
{
  "prompt": "Your message here"
}
```
- Response Body (JSON):
```json
{
  "response": "Chatbot's reply",
  "elapsed_time": 1.234
}
```

## Example Request with curl
```bash
curl -X POST "http://127.0.0.1:8000/chat/" -H "Content-Type: application/json" -d '{"prompt": "kannst du deutsch?"}'
```

## Development Notes
The model loads once at startup to optimize performance.

Response time is measured per request and included in the API response.

Tokenizer uses a chat template for formatting inputs compatible with the model.

The assistant’s response is extracted from the model’s full output using regex.

## License
MIT License

## Author
Burak Eminç — projectsairus@gmail.com
