#!/usr/bin/env python3
"""
Dynamic MCP Server Demo
Demonstrates how to use the dynamic MCP server discovery and configuration system
"""

import asyncio
import json
from mcp_server_manager import MCPServerManager


async def demo_dynamic_mcp():
    """
    Demo the dynamic MCP server system
    """
    print("🚀 Dynamic MCP Server System Demo")
    print("="*50)
    
    # Initialize the MCP server manager
    print("\n1. Initializing MCP Server Manager...")
    manager = MCPServerManager()
    
    # Discover and load servers
    print("\n2. Discovering MCP servers...")
    servers = await manager.discover_servers()
    
    if servers:
        print(f"\n✅ Successfully loaded {len(servers)} servers:")
        for name, info in servers.items():
            print(f"   • {name}: {info['description']}")
    else:
        print("\n⚠️  No servers loaded")
    
    # List available capabilities
    print("\n3. Available server capabilities:")
    methods = manager.get_available_methods()
    if methods:
        for i, method in enumerate(methods[:10], 1):  # Show first 10
            server = manager.get_capability(method)
            if server:
                print(f"   {i:2d}. {method} (→ {server['description']})")
        if len(methods) > 10:
            print(f"   ... and {len(methods) - 10} more methods")
    else:
        print("   No capabilities found")
    
    # Test some server methods
    print("\n4. Testing server methods:")
    
    # Test Bitcoin price (if available)
    try:
        if "get_bitcoin_price" in methods:
            print("\n💰 Testing Bitcoin price...")
            result = await manager.execute_method("get_bitcoin_price")
            print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Bitcoin test failed: {e}")
    
    # Test weather (if available)
    try:
        if "get_weather" in methods:
            print("\n🌤️ Testing weather...")
            result = await manager.execute_method("get_weather")
            print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Weather test failed: {e}")
    
    # Test time info (if available)
    try:
        if "get_time_info" in methods:
            print("\n🕒 Testing time info...")
            result = await manager.execute_method("get_time_info")
            print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Time test failed: {e}")
    
    # Test web search (if available)
    try:
        if "search" in methods:
            print("\n🔍 Testing web search...")
            result = await manager.execute_method("search", "test query", limit=2)
            print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Search test failed: {e}")
    
    # Test activity methods (if available)
    try:
        if "log_activity" in methods:
            print("\n📝 Testing activity logging...")
            await manager.execute_method("log_activity", "Test activity from MCP demo")
            print("   ✅ Activity logged successfully")
    except Exception as e:
        print(f"   ❌ Activity test failed: {e}")
    
    # Demonstrate adding a custom server
    print("\n5. Demonstrating custom server management:")
    
    # Add a disabled custom server
    new_server_config = {
        "name": "demo_custom_server",
        "type": "python_module",
        "module": "custom_modules.example_server",
        "class": "MyCustomServer",
        "methods": ["analyze_data", "generate_report", "health_check"],
        "enabled": False,
        "description": "Demo custom server for text analysis"
    }
    
    success = manager.add_custom_server(new_server_config)
    if success:
        print("   ✅ Added custom server configuration")
    
    # Enable the server
    success = manager.enable_server("demo_custom_server")
    if success:
        print("   ✅ Enabled custom server")
        
        # Try to load it
        await manager.load_server(new_server_config)
        print("   ✅ Loaded custom server")
    
    # Test custom server
    try:
        if "analyze_data" in manager.get_available_methods():
            print("\n🧪 Testing custom server method...")
            result = await manager.execute_method(
                "analyze_data", 
                "This is a test string with numbers 123 and symbols @#$%"
            )
            print(f"   Custom server result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"   ❌ Custom server test failed: {e}")
    
    # Show final state
    print("\n6. Final server state:")
    final_methods = manager.get_available_methods()
    print(f"   Total methods available: {len(final_methods)}")
    print(f"   Loaded servers: {len(manager.list_servers())}")
    
    print("\n" + "="*50)
    print("🎉 Dynamic MCP Server Demo Complete!")
    print("\nNext steps:")
    print("1. Edit configs/mcp_servers.json to add your own servers")
    print("2. Set 'enabled': true for servers you want to use")
    print("3. Create custom Python modules in custom_modules/ directory")
    print("4. Restart to load new servers")
    

async def demo_server_management():
    """
    Demo server management functions
    """
    print("\n🔧 Server Management Demo")
    print("="*30)
    
    manager = MCPServerManager()
    
    # Show current custom servers
    print("\nCurrent custom servers:")
    custom_servers = manager.custom_servers_config.get("servers", [])
    for server in custom_servers:
        status = "✅ Enabled" if server.get("enabled", False) else "❌ Disabled"
        print(f"   • {server['name']}: {server['description']} ({status})")
    
    # Enable/disable servers
    print("\nServer management commands:")
    print("• Enable server: manager.enable_server('server_name')")
    print("• Disable server: manager.disable_server('server_name')")
    print("• Remove server: manager.remove_custom_server('server_name')")
    print("• Add server: manager.add_custom_server(server_config)")
    
    # Show how to create a custom server
    print("\n📝 Example custom server configuration:")
    example_config = {
        "name": "my_data_analyzer",
        "type": "python_module",
        "module": "custom_modules.my_analysis",
        "class": "DataAnalyzer",
        "methods": ["analyze_csv", "generate_insights"],
        "enabled": True,
        "description": "My custom data analysis server"
    }
    print(json.dumps(example_config, indent=2))


if __name__ == "__main__":
    # Run main demo
    asyncio.run(demo_dynamic_mcp())
    
    # Run management demo
    asyncio.run(demo_server_management())