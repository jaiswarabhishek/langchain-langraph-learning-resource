import asyncio
import sys
from typing import Optional
from contextlib import AsyncExitStack
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
tools_prompt ="""


TOOLS:

1. read_text_file  
   - readOnly: true  
   - Notes: Pure read of text files.

2. read_media_file  
   - readOnly: true  
   - Notes: Pure read of binary/media files.

3. read_multiple_files  
   - readOnly: true  
   - Notes: Pure read of multiple files at once.

4. list_directory  
   - readOnly: true  
   - Notes: Returns items in a directory.

5. list_directory_with_sizes  
   - readOnly: true  
   - Notes: Directory listing with sizes.

6. directory_tree  
   - readOnly: true  
   - Notes: Returns a recursive directory structure.

7. search_files  
   - readOnly: true  
   - Notes: File search across allowed directories.

8. get_file_info  
   - readOnly: true  
   - Notes: Metadata lookup.

9. list_allowed_directories  
   - readOnly: true  
   - Notes: Lists directories the assistant can access.

10. create_directory  
    - readOnly: false  
    - idempotent: true  
    - destructive: false  
    - Notes: Re-creating an existing directory is a no-op.

11. write_file  
    - readOnly: false  
    - idempotent: true  
    - destructive: true  
    - Notes: Overwrites existing files.

12. edit_file  
    - readOnly: false  
    - idempotent: false  
    - destructive: true  
    - Notes: Re-applying edits may corrupt or double-apply changes.

13. move_file  
    - readOnly: false  
    - idempotent: false  
    - destructive: true  
    - Notes: Moving a file deletes the source.

RULES:
- Always choose the least-destructive tool that satisfies the user’s request.
- Never call a destructive or non-idempotent tool unless the user explicitly asks for a modifying action (create, write, edit, move).
- When modifying files, warn the user if an action may overwrite or delete data.
- For read-only queries, use only read-only tools.
- Ensure the user intent is unambiguous before using any destructive tool.

"""
class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.tools = []  # store tools for reuse

    async def connect_to_server(self, server_script_path: str):
        """Connect to a Local FileSystem MCP server"""
        print(server_script_path)

        server_params = StdioServerParameters(
            command="mcp-server-filesystem",
            args=[server_script_path],
            cwd="E:/Langchain"
        )
        print(server_params)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools and cache them
        response = await self.session.list_tools()
        self.tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    def _build_groq_tools(self):
        """Convert MCP tools to Groq-compatible tool format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in self.tools
        ]

    async def process_query(self, query: str) -> str:
        """Process user query using Groq LLM + MCP filesystem tools with access to read,write,update and delete all CRUD operations you have access to below TOOLS:

1. read_text_file  
   - readOnly: true  
   - Notes: Pure read of text files.

2. read_media_file  
   - readOnly: true  
   - Notes: Pure read of binary/media files.

3. read_multiple_files  
   - readOnly: true  
   - Notes: Pure read of multiple files at once.

4. list_directory  
   - readOnly: true  
   - Notes: Returns items in a directory.

5. list_directory_with_sizes  
   - readOnly: true  
   - Notes: Directory listing with sizes.

6. directory_tree  
   - readOnly: true  
   - Notes: Returns a recursive directory structure.

7. search_files  
   - readOnly: true  
   - Notes: File search across allowed directories.

8. get_file_info  
   - readOnly: true  
   - Notes: Metadata lookup.

9. list_allowed_directories  
   - readOnly: true  
   - Notes: Lists directories the assistant can access.

10. create_directory  
    - readOnly: false  
    - idempotent: true  
    - destructive: false  
    - Notes: Re-creating an existing directory is a no-op.

11. write_file  
    - readOnly: false  
    - idempotent: true  
    - destructive: true  
    - Notes: Overwrites existing files.

12. edit_file  
    - readOnly: false  
    - idempotent: false  
    - destructive: true  
    - Notes: Re-applying edits may corrupt or double-apply changes.

13. move_file  
    - readOnly: false  
    - idempotent: false  
    - destructive: true  
    - Notes: Moving a file deletes the source.

RULES:
- Always choose the least-destructive tool that satisfies the user’s request.
- Never call a destructive or non-idempotent tool unless the user explicitly asks for a modifying action (create, write, edit, move).
- When modifying files, warn the user if an action may overwrite or delete data.
- For read-only queries, use only read-only tools.
- Ensure the user intent is unambiguous before using any destructive tool.
"""

        messages = [
            # {
            #     "role": "system",
            #     "content": (
            #         "You are a helpful assistant with access to the local filesystem. "
            #         "Use the provided tools to read, write, list, and manage files as needed to answer the user's query."
            #     )
            # },
            {
                "role": "user",
                "content": query
            }
        ]

        groq_tools = self._build_groq_tools()

        # Agentic loop — keep going until Groq stops calling tools
        while True:
            response = self.groq.chat.completions.create(
                model="openai/gpt-oss-120b",
                max_tokens=4096,
                tools=groq_tools,
                messages=messages,
                tool_choice="auto"
            )
            print(response)

            message = response.choices[0].message

            # No tool calls — final answer, we're done
            if not message.tool_calls:
                return message.content

            # Append assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute each tool call via MCP and collect results
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"\n[Tool Call] {tool_name}({json.dumps(tool_args, indent=2)})")

                # Call the MCP tool
                tool_result = await self.session.call_tool(tool_name, tool_args)
                result_text = tool_result.content[0].text if tool_result.content else ""

                print(f"[Tool Result] {result_text[:200]}{'...' if len(result_text) > 200 else ''}")

                messages.append({
                "role": "assistant",
                "content": result_text,
                })

                            # Make a second API call with the updated conversation
                second_response = self.groq.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    max_tokens=1000,
                    tool_choice="auto"
                )

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
            
            return second_response.choices[0].message.content

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
        print("Usage: python client.py <allowed_directory>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())