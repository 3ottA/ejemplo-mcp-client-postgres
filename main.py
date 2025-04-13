import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, AsyncGenerator

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# Definir modelos Pydantic para la API
class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    user: Optional[str] = None

    @validator('messages', 'prompt')
    def validate_messages_or_prompt(cls, v, values, **kwargs):
        if v is None and values.get('prompt') is None:
            raise ValueError('Either messages or prompt must be provided')
        return v

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError(
                "API key inválida. Por favor, asegúrate de tener una API key válida de OpenAI "
                "que comience con 'sk-' y configúrala en la variable de entorno LLM_API_KEY"
            )
        self.client = OpenAI(api_key=api_key)

    def get_response(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.
            temperature: Temperature parameter for the LLM.
            max_tokens: Maximum tokens for the LLM response.

        Returns:
            The LLM's response as a string.

        Raises:
            Exception: If the request to the LLM fails.
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content

        except Exception as e:
            error_message = f"Error de comunicación con OpenAI: {str(e)}"
            logging.error(error_message)
            return error_message


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.all_tools = []

    async def initialize_servers(self) -> None:
        """Initialize all servers."""
        for server in self.servers:
            await server.initialize()
            tools = await server.list_tools()
            self.all_tools.extend(tools)

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(
                                    f"Progress: {progress}/{total} "
                                    f"({percentage:.1f}%)"
                                )

                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def generate_response(self, messages: List[Dict[str, str]], 
                                temperature: float = 0.7, 
                                max_tokens: int = 4096) -> str:
        """Generate a response for a chat request.

        Args:
            messages: List of message dictionaries with role and content.
            temperature: Temperature parameter for generation.
            max_tokens: Maximum tokens for generation.

        Returns:
            The generated response text.
        """
        # Crear formato de sistema con herramientas
        tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools])

        system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )

        # Agregar mensaje de sistema al inicio si no existe
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_message}] + messages

        # Obtener respuesta del LLM
        llm_response = self.llm_client.get_response(messages, temperature, max_tokens)
        
        # Procesar respuesta para ejecutar herramientas si es necesario
        result = await self.process_llm_response(llm_response)
        
        # Si se ejecutó una herramienta, obtener respuesta final
        if result != llm_response:
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "system", "content": result})
            
            final_response = self.llm_client.get_response(messages, temperature, max_tokens)
            return final_response
        else:
            return llm_response


# Variables globales para el servidor API
chat_session = None
config = None

@app.on_event("startup")
async def startup_event():
    """Initialize servers and configurations on startup."""
    global chat_session, config
    
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)
    
    await chat_session.initialize_servers()
    logging.info("API initialized and ready to process requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up servers on shutdown."""
    global chat_session
    if chat_session:
        await chat_session.cleanup_servers()
    logging.info("API shutdown complete")

# Endpoint para listar modelos (compatible con OpenAI)
@app.get("/v1/chat/completions/models", response_model=ModelList)
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Lista los modelos disponibles."""
    # Modelos que nuestro servicio puede simular
    models = [
        Model(
            id="gpt-4o-mini",
            created=int(time.time()) - 10000,
            owned_by="organization-owner"
        ),
        Model(
            id="gpt-3.5-turbo",
            created=int(time.time()) - 20000,
            owned_by="organization-owner"
        )
    ]
    
    return ModelList(data=models)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """API endpoint compatible with OpenAI Chat Completions API v1."""
    global chat_session
    
    if not chat_session:
        raise HTTPException(status_code=500, detail="API not initialized")
    
    # Convertir mensajes según el formato recibido
    if request.messages:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    else:
        # Parsear el prompt en formato ||>User:\n...\n||>Assistant:\n
        prompt_parts = request.prompt.split("||>")
        messages = []
        for part in prompt_parts:
            if part.startswith("User:"):
                content = part[5:].strip()  # Eliminar "User:" y espacios
                if content:
                    messages.append({"role": "user", "content": content})
            elif part.startswith("Assistant:"):
                content = part[10:].strip()  # Eliminar "Assistant:" y espacios
                if content:
                    messages.append({"role": "assistant", "content": content})
    
    # Si stream=True, devolver una respuesta streaming
    if request.stream:
        return EventSourceResponse(
            stream_chat_response(request.model, messages, request.temperature, request.max_tokens),
            media_type="text/event-stream"
        )
    
    # Para respuestas no streaming
    response_text = await chat_session.generate_response(
        messages, 
        temperature=request.temperature, 
        max_tokens=request.max_tokens
    )
    
    # Crear respuesta en formato OpenAI
    response = ChatCompletionResponse(
        id=f"chatcmpl-{str(uuid.uuid4())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=len(str(messages)) // 4,  # Estimación simple
            completion_tokens=len(response_text) // 4,  # Estimación simple
            total_tokens=(len(str(messages)) + len(response_text)) // 4  # Estimación simple
        )
    )
    
    return response

async def stream_chat_response(model: str, messages: List[Dict[str, str]], 
                              temperature: float = 0.7, 
                              max_tokens: int = 4096) -> AsyncGenerator[str, None]:
    """
    Genera la respuesta en formato de streaming SSE para el cliente.
    
    Args:
        model: Modelo a utilizar
        messages: Lista de mensajes
        temperature: Temperatura para la generación
        max_tokens: Tokens máximos para la respuesta
        
    Yields:
        Eventos SSE con los chunks de la respuesta
    """
    # ID único para esta respuesta
    response_id = f"chatcmpl-{str(uuid.uuid4())}"
    created_time = int(time.time())
    
    # Generar la respuesta completa 
    # En un caso real, deberías obtener los chunks directamente del LLM
    try:
        response_text = await chat_session.generate_response(messages, temperature, max_tokens)
        
        # Primer chunk: metadatos y rol
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }
            ]
        }
        yield json.dumps(first_chunk)
        
        # Dividir la respuesta en chunks (simulando streaming)
        # En producción, idealmente obtendrías tokens directamente del modelo
        # Aquí simulamos la división por caracteres 
        chunk_size = 10  # Ajusta según necesites
        
        for i in range(0, len(response_text), chunk_size):
            content_chunk = response_text[i:i+chunk_size]
            
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content_chunk},
                        "finish_reason": None
                    }
                ]
            }
            
            # Pequeña pausa para simular el streaming real
            await asyncio.sleep(0.05)
            yield json.dumps(chunk)
        
        # Último chunk: indicando finalización
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield json.dumps(final_chunk)
        
        # [DONE] marca el final del stream para el cliente OpenAI
        yield "[DONE]"
        
    except Exception as e:
        logging.error(f"Error en streaming: {str(e)}")
        error_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"Error: {str(e)}"},
                    "finish_reason": "error"
                }
            ]
        }
        yield json.dumps(error_chunk)
        yield "[DONE]"

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
