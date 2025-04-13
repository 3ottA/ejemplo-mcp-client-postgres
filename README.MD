# MCP API compatible con OpenAI

Esta API es compatible con el formato de la API de OpenAI Chat Completions v1, lo que permite integrarla fácilmente con servicios existentes que utilizan la API de OpenAI.

## Requisitos

- Python 3.9+
- Dependencias del archivo `requirements.txt`
- Una clave API válida de OpenAI para el motor subyacente

## Instalación

1. Clonar este repositorio
2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar la variable de entorno o crear un archivo `.env` con:

```
LLM_API_KEY=sk-tu-clave-api-openai
```

4. Asegurarse de tener un archivo `servers_config.json` con la configuración de los servidores MCP.

## Ejecución

Para iniciar el servidor:

```bash
python main.py
```

El servidor se ejecutará en `http://0.0.0.0:8000`.

## Uso

La API es compatible con los siguientes endpoints de OpenAI:

### Obtener modelos disponibles

```bash
# Endpoint compatible con OpenAI
curl -X GET http://localhost:8000/v1/models

# Endpoint adicional (hace lo mismo)
curl -X GET http://localhost:8000/v1/chat/completions/models
```

### Generar completaciones de chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "Hola, ¿cómo estás?"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 4096
  }'
```

La respuesta seguirá el mismo formato que la API de OpenAI:

```json
{
  "id": "chatcmpl-123456",
  "object": "chat.completion",
  "created": 1682222800,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "¡Hola! Estoy bien, gracias por preguntar. ¿En qué puedo ayudarte hoy?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 25,
    "total_tokens": 45
  }
}
```

## Integración con NGINX

Para usar esta API con NGINX, configura un proxy inverso apuntando a este servicio:

```nginx
location /v1/chat/completions {
    proxy_pass http://localhost:8000/v1/chat/completions;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

location /v1/models {
    proxy_pass http://localhost:8000/v1/models;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```


