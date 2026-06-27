import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # self.client = None
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Groq and available tools"""
        model_messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        # print("\nAvailable tools:", response.tools)
        available_tools = [{
        "type": "function",
        "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema}
        } for tool in response.tools]

        # Initial Groq API call
        response = self.groq.chat.completions.create(
            model="qwen/qwen3-32b",
            max_tokens=1000,
            messages=model_messages,
            tools=available_tools,
            tool_choice="auto"
        )

        # Process response and handle tool calls
# Extract the response and any tool call responses
        response_message = response.choices[0].message
        # print(response_message)
        tool_calls = response_message.tool_calls
        # print(tool_calls)
        result_messages = []
        assistant_message_content = []
    
    
        if tool_calls:
            print("\n\n")
            # print(f"messages: {model_messages}")
            model_messages.append({
                "role":"assistant",
                "content":response_message.reasoning
            })
            result_messages.append(response_message.reasoning)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                # print(f"{function_args} {function_name}")
                # Call the tool and get the response
                # print(f"this is result {result}")
                result_messages.append(f"[Calling tool {function_name} with args {function_args}]")
                result = await self.session.call_tool(function_name, function_args)
                # print(f"tool response {result}")
               
                # assistant_message_content.append(response_message.reasoning)

                # result_messages.append({
                #     "role":"assistant",
                #     "content":f"[Calling tool {function_name} with args {function_args}]"
                # })

                result_content = result.content
                if isinstance(result_content, list):
                    result_content = "\n".join(str(item.text) for item in result_content)

                model_messages.append(
                    {
                        "tool_call_id": tool_call.id, 
                        "role": "tool", # Indicates this message is from tool use
                        "name": function_name,
                        "content": result_content,
                    }
                )

            
            
            # Make a second API call with the updated conversation
            second_response = self.groq.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=model_messages,
                max_tokens=1000,
                tools=available_tools
            )

            # print(second_response.choices[0].message)

            result_messages.append(second_response.choices[0].message.content)
            # Return the final response
        return "\n".join(result_messages)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())