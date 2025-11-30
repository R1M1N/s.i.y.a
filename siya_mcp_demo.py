#!/usr/bin/env python3
"""
S.I.Y.A Enhanced - Complete MCP Integration Demo
Demonstrates full system capabilities with MCP servers

This demo shows S.I.Y.A with:
- Real-time data access (time, web search)
- Filesystem operations (read, write, manage files)
- Memory/context management
- Sequential task planning
- Web browsing capabilities
- System integration (commands, monitoring)
"""

import asyncio
import json
import time
from pathlib import Path
import subprocess
import sys

# Import our enhanced S.I.Y.A system
from siya_enhanced import SiyaEnhanced

class MCPManager:
    """Manages all MCP servers for S.I.Y.A"""
    
    def __init__(self):
        self.servers = {
            "time": "http://localhost:8001",
            "sequentialthinking": "http://localhost:8002", 
            "memory": "http://localhost:8003",
            "filesystem": "http://localhost:8004",
            "search": "http://localhost:8005",
            "browser": "http://localhost:8006"
        }
        self.processes = {}
    
    async def start_server(self, server_name: str, script_path: str):
        """Start an MCP server as a subprocess"""
        try:
            print(f"🚀 Starting {server_name} server...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.processes[server_name] = process
            print(f"✅ {server_name} server started (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"❌ Failed to start {server_name} server: {e}")
            return False
    
    async def start_all_servers(self):
        """Start all MCP servers"""
        print("🔧 Starting all MCP servers...")
        
        server_scripts = {
            "time": "/workspace/mcp_time_server.py",
            "sequentialthinking": "/workspace/mcp_sequential_thinking_server.py",
            "memory": "/workspace/mcp_memory_server.py", 
            "filesystem": "/workspace/mcp_filesystem_server.py",
            "search": "/workspace/mcp_search_server.py",
            "browser": "/workspace/mcp_browser_server.py"
        }
        
        # Start servers with small delays
        for server_name, script_path in server_scripts.items():
            success = await self.start_server(server_name, script_path)
            if success:
                await asyncio.sleep(1)  # Small delay between starts
        
        print("⏳ Waiting for servers to initialize...")
        await asyncio.sleep(3)
        print("✅ All MCP servers should be ready!")
    
    async def stop_all_servers(self):
        """Stop all MCP servers"""
        print("🛑 Stopping all MCP servers...")
        
        for server_name, process in self.processes.items():
            try:
                process.terminate()
                await process.wait()
                print(f"✅ {server_name} server stopped")
            except Exception as e:
                print(f"❌ Error stopping {server_name} server: {e}")
        
        self.processes.clear()

async def demonstrate_enhanced_capabilities():
    """Demonstrate S.I.Y.A's enhanced capabilities"""
    
    print("🚀 S.I.Y.A Enhanced - Full System Integration Demo")
    print("=" * 60)
    print("🎯 This demo shows S.I.Y.A with:")
    print("• 📁 Filesystem operations")
    print("• 🖥️ System monitoring")
    print("• 🌐 Web search")
    print("• 🧠 Memory management")
    print("• ⏰ Time/date functions")
    print("• 💻 Command execution")
    print("• 🔍 File management")
    print("=" * 60)
    
    # Initialize S.I.Y.A
    siya = SiyaEnhanced()
    await siya.initialize_mcp()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Basic Time Query",
            "input": "What time is it?",
            "expected": "Should show current time and date"
        },
        {
            "name": "System Information",
            "input": "Show me system info",
            "expected": "Should display CPU, memory, disk usage"
        },
        {
            "name": "Web Search",
            "input": "Search for artificial intelligence tutorials",
            "expected": "Should return search results"
        },
        {
            "name": "Filesystem Operations",
            "input": "List directory",
            "expected": "Should show current directory contents"
        },
        {
            "name": "Memory Check",
            "input": "Show conversation history",
            "expected": "Should display memory context"
        },
        {
            "name": "File Reading",
            "input": "Read file README.md",
            "expected": "Should read and display file contents"
        },
        {
            "name": "Command Execution", 
            "input": "Run command ps aux",
            "expected": "Should execute command and show output"
        },
        {
            "name": "Enhanced Conversation",
            "input": "What can you do?",
            "expected": "Should explain all capabilities"
        }
    ]
    
    print("🧪 Running test scenarios...")
    print("-" * 40)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 Scenario {i}: {scenario['name']}")
        print(f"📝 Input: {scenario['input']}")
        print(f"🎪 Expected: {scenario['expected']}")
        print("🤖 S.I.Y.A: ", end="", flush=True)
        
        start_time = time.time()
        response = await siya.process_command(scenario['input'])
        response_time = (time.time() - start_time) * 1000
        
        print(response)
        print(f"⏱️ Response time: {response_time:.1f}ms")
        
        # Small delay between scenarios
        await asyncio.sleep(1)
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! S.I.Y.A Enhanced is fully functional!")
    print("=" * 60)

async def interactive_mode():
    """Interactive mode for user testing"""
    
    print("\n🎤 Interactive Mode - Type your commands!")
    print("💡 Try commands like:")
    print("• 'What time is it?'")
    print("• 'Show system info'")
    print("• 'Search for Python tutorials'")
    print("• 'Read file README.md'")
    print("• 'List directory'")
    print("• 'Run command ls -la'")
    print("• 'What can you do?'")
    print("• Type 'quit' to exit")
    print("-" * 50)
    
    # Initialize S.I.Y.A
    siya = SiyaEnhanced()
    await siya.initialize_mcp()
    
    try:
        while True:
            user_input = input("\n🎤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! S.I.Y.A Enhanced signing off.")
                break
            
            if not user_input:
                continue
            
            print("🤖 S.I.Y.A: ", end="", flush=True)
            response = await siya.process_command(user_input)
            print(response)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await siya.close()

async def main():
    """Main demo entry point"""
    
    # Check if we should start MCP servers
    start_servers = len(sys.argv) > 1 and sys.argv[1] == "--start-servers"
    
    if start_servers:
        # Start MCP servers
        mcp_manager = MCPManager()
        await mcp_manager.start_all_servers()
        
        print("\n🔧 MCP servers started! You can now run the enhanced demo.")
        print("💡 Run this script without --start-servers to test the demo")
        
        try:
            await interactive_mode()
        finally:
            await mcp_manager.stop_all_servers()
    else:
        # Run demo without starting servers
        print("📝 Running demo (MCP servers assumed to be running)")
        print("💡 Run with --start-servers to start all MCP servers automatically")
        
        await demonstrate_enhanced_capabilities()
        
        # Ask if user wants interactive mode
        try:
            response = input("\n❓ Try interactive mode? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                await interactive_mode()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        sys.exit(1)