# Python MCP Filesystem Client

A custom Python client that connects to the **`@modelcontextprotocol/server-filesystem`** MCP server — the same one configured in VS Code.

---

## Prerequisites

```bash
# Node.js (required to run the MCP server via npx)
node --version   # v18+

# Python 3.11+
python --version
```

No extra Python packages needed — uses only the standard library (`asyncio`, `json`, `subprocess`).

---

## VS Code MCP Configuration

The filesystem server in VS Code is configured in **`.vscode/mcp.json`**:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/your/project"
      ]
    }
  }
}
```

Or alternatively in **`settings.json`**:

```json
{
  "mcp": {
    "servers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
      }
    }
  }
}
```

Your Python client uses the **same server command** — just spawns it as a subprocess directly.

---

## Running the Client

```bash
# Allow access to current directory
python mcp_client.py .

# Allow access to a specific directory
python mcp_client.py /home/user/projects

# Allow access to multiple directories
python mcp_client.py /home/user/projects /tmp/sandbox
```

---

## Interactive REPL Commands

Once connected, an interactive REPL starts:

```
mcp> ls .                          # list directory
mcp> cat README.md                 # read a file
mcp> write hello.txt Hello World   # write a file
mcp> mkdir new_folder              # create directory
mcp> info src/main.py              # file metadata
mcp> search . *.py                 # search by pattern
mcp> move old.txt new.txt          # move/rename
mcp> dirs                          # list allowed roots
mcp> tools                         # show all available tools
mcp> quit                          # exit
```

---

## Architecture

```
Python Client (mcp_client.py)
        │
        │  stdin/stdout  (JSON-RPC 2.0 over stdio)
        ▼
npx @modelcontextprotocol/server-filesystem <dir>
        │
        ▼
   Filesystem (sandboxed to allowed directories)
```

### MCP Protocol Flow

```
Client                          Server
  │                               │
  │──── initialize ──────────────►│
  │◄─── result (capabilities) ───│
  │──── notifications/initialized►│
  │                               │
  │──── tools/list ──────────────►│
  │◄─── result (tool list) ──────│
  │                               │
  │──── tools/call ──────────────►│
  │◄─── result (content) ────────│
```

---

## Using as a Library

```python
import asyncio
from mcp_client import MCPClient

async def main():
    async with MCPClient(allowed_dirs=["/my/project"]) as client:
        # Read a file
        content = await client.read_file("/my/project/README.md")
        print(content)

        # Write a file
        await client.write_file("/my/project/output.txt", "Hello from Python!")

        # List a directory
        listing = await client.list_directory("/my/project/src")
        print(listing)

        # Call any tool directly
        result = await client.call_tool("search_files", {
            "path": "/my/project",
            "pattern": "*.py"
        })
        print(result)

asyncio.run(main())
```

---

## Available Filesystem Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read complete file contents |
| `read_multiple_files` | Read several files at once |
| `write_file` | Create or overwrite a file |
| `create_directory` | Create a new directory |
| `list_directory` | List directory contents |
| `move_file` | Move or rename a file |
| `search_files` | Search by filename pattern |
| `get_file_info` | File size, modified time, type |
| `list_allowed_directories` | Show sandbox root paths |
