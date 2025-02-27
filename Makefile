.PHONY: run
start:
	docker compose up -d 

stop:
	docker compose down 

update:
	docker compose down 
	docker compose pull
	docker compose up -d --build

restart: 
	docker compose restart

remove:
	docker compose down -v
	docker compose rm -f



curl_ollam:
	curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d '{
  "model": "llama3.1:8b",
  "prompt": "Ollama is 22 years old and is busy saving the world. Respond using JSON",
  "stream": false,
  "format": {
    "type": "object",
    "properties": {
      "age": {
        "type": "integer"
      },
      "available": {
        "type": "boolean"
      }
    },
    "required": [
      "age",
      "available"
    ]
  }
}'	