"""
MCP Server Manager - Dynamic discovery and integration system
"""

import json
import importlib
import asyncio
from typing import Dict, List, Any, Optional
import logging

class MCPServerManager:
    """
    Manages dynamic discovery and integration of MCP servers
    """
    
    def __init__(self, config_path: str = "siya_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.custom_servers_config = self.load_custom_servers()
        self.loaded_servers = {}
        self.server_capabilities = {}
        
    def load_config(self) -> Dict:
        """Load main configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
            return {}
    
    def load_custom_servers(self) -> Dict:
        """Load custom MCP servers configuration"""
        try:
            with open("configs/mcp_servers.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load custom servers config: {e}")
            return {"servers": []}
    
    async def discover_servers(self) -> Dict[str, Any]:
        """Discover and load available MCP servers"""
        print("🔍 Discovering MCP servers...")
        
        # Load built-in servers from config
        if "mcp_servers" in self.config and self.config["mcp_servers"].get("enabled"):
            for server_config in self.config["mcp_servers"]["servers"]:
                await self.load_server(server_config)
        
        # Load custom servers that are enabled
        for server_config in self.custom_servers_config.get("servers", []):
            if server_config.get("enabled", False):
                await self.load_server(server_config)
        
        print(f"✅ Loaded {len(self.loaded_servers)} MCP servers")
        return self.loaded_servers
    
    async def load_server(self, server_config: Dict) -> bool:
        """Load a single MCP server"""
        server_name = server_config["name"]
        server_type = server_config["type"]
        
        try:
            if server_type == "local":
                success = await self.load_local_server(server_config)
            elif server_type == "external":
                success = await self.load_external_server(server_config)
            elif server_type == "python_module":
                success = await self.load_python_module_server(server_config)
            else:
                print(f"❌ Unknown server type: {server_type}")
                return False
            
            if success:
                print(f"✅ Loaded server: {server_name}")
                return True
            else:
                print(f"❌ Failed to load server: {server_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading server {server_name}: {e}")
            return False
    
    async def load_local_server(self, server_config: Dict) -> bool:
        """Load a local server from siya_enhanced.py"""
        try:
            module_name = server_config["module"]
            class_name = server_config["class"]
            methods = server_config.get("methods", [])
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the class
            if hasattr(module, class_name):
                server_class = getattr(module, class_name)
                
                # Create instance
                server_instance = server_class()
                
                # Store server info
                self.loaded_servers[server_config["name"]] = {
                    "instance": server_instance,
                    "methods": methods,
                    "type": "local",
                    "description": server_config.get("description", "")
                }
                
                # Store capabilities
                for method in methods:
                    if hasattr(server_instance, method):
                        self.server_capabilities[method] = server_config["name"]
                
                return True
            else:
                print(f"❌ Class {class_name} not found in module {module_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading local server: {e}")
            return False
    
    async def load_external_server(self, server_config: Dict) -> bool:
        """Load an external HTTP server"""
        try:
            url = server_config["url"]
            methods = server_config.get("methods", [])
            
            # Store external server info
            self.loaded_servers[server_config["name"]] = {
                "url": url,
                "methods": methods,
                "type": "external",
                "description": server_config.get("description", "")
            }
            
            # Store capabilities for external calls
            for method in methods:
                self.server_capabilities[method["name"]] = server_config["name"]
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading external server: {e}")
            return False
    
    async def load_python_module_server(self, server_config: Dict) -> bool:
        """Load a Python module server"""
        try:
            module_name = server_config["module"]
            class_name = server_config["class"]
            methods = server_config.get("methods", [])
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the class
            if hasattr(module, class_name):
                server_class = getattr(module, class_name)
                
                # Create instance
                server_instance = server_class()
                
                # Store server info
                self.loaded_servers[server_config["name"]] = {
                    "instance": server_instance,
                    "methods": methods,
                    "type": "python_module",
                    "description": server_config.get("description", "")
                }
                
                # Store capabilities
                for method in methods:
                    if hasattr(server_instance, method):
                        self.server_capabilities[method] = server_config["name"]
                
                return True
            else:
                print(f"❌ Class {class_name} not found in module {module_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading Python module server: {e}")
            return False
    
    def get_capability(self, method_name: str) -> Optional[Dict]:
        """Get server capability for a specific method"""
        server_name = self.server_capabilities.get(method_name)
        if server_name:
            return self.loaded_servers.get(server_name)
        return None
    
    def get_available_methods(self) -> List[str]:
        """Get list of all available methods"""
        return list(self.server_capabilities.keys())
    
    def list_servers(self) -> Dict[str, Dict]:
        """List all loaded servers"""
        return self.loaded_servers
    
    async def execute_method(self, method_name: str, *args, **kwargs) -> Any:
        """Execute a method on the appropriate server"""
        server_info = self.get_capability(method_name)
        
        if not server_info:
            raise ValueError(f"Method {method_name} not available on any server")
        
        server_type = server_info["type"]
        
        if server_type == "local" or server_type == "python_module":
            # Execute on local instance
            instance = server_info["instance"]
            if hasattr(instance, method_name):
                method = getattr(instance, method_name)
                if asyncio.iscoroutinefunction(method):
                    return await method(*args, **kwargs)
                else:
                    return method(*args, **kwargs)
            else:
                raise ValueError(f"Method {method_name} not found on server instance")
        
        elif server_type == "external":
            # Execute on external server (would need HTTP client implementation)
            # For now, just return a placeholder
            print(f"🌐 External server call to {method_name} (not implemented yet)")
            return f"External call to {method_name} (server: {server_info['description']})"
        
        else:
            raise ValueError(f"Unknown server type: {server_type}")
    
    def add_custom_server(self, server_config: Dict) -> bool:
        """Add a new custom server to the configuration"""
        try:
            # Add to custom servers list
            self.custom_servers_config["servers"].append(server_config)
            
            # Save to file
            with open("configs/mcp_servers.json", 'w') as f:
                json.dump(self.custom_servers_config, f, indent=2)
            
            print(f"✅ Added custom server: {server_config['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding custom server: {e}")
            return False
    
    def remove_custom_server(self, server_name: str) -> bool:
        """Remove a custom server from configuration"""
        try:
            servers = self.custom_servers_config["servers"]
            original_length = len(servers)
            
            # Remove server
            servers = [s for s in servers if s["name"] != server_name]
            
            if len(servers) < original_length:
                self.custom_servers_config["servers"] = servers
                
                # Save to file
                with open("configs/mcp_servers.json", 'w') as f:
                    json.dump(self.custom_servers_config, f, indent=2)
                
                print(f"✅ Removed custom server: {server_name}")
                return True
            else:
                print(f"❌ Server {server_name} not found in custom servers")
                return False
                
        except Exception as e:
            print(f"❌ Error removing custom server: {e}")
            return False
    
    def enable_server(self, server_name: str) -> bool:
        """Enable a custom server"""
        try:
            for server in self.custom_servers_config["servers"]:
                if server["name"] == server_name:
                    server["enabled"] = True
                    
                    # Save to file
                    with open("configs/mcp_servers.json", 'w') as f:
                        json.dump(self.custom_servers_config, f, indent=2)
                    
                    print(f"✅ Enabled server: {server_name}")
                    return True
            
            print(f"❌ Server {server_name} not found")
            return False
            
        except Exception as e:
            print(f"❌ Error enabling server: {e}")
            return False
    
    def disable_server(self, server_name: str) -> bool:
        """Disable a custom server"""
        try:
            for server in self.custom_servers_config["servers"]:
                if server["name"] == server_name:
                    server["enabled"] = False
                    
                    # Save to file
                    with open("configs/mcp_servers.json", 'w') as f:
                        json.dump(self.custom_servers_config, f, indent=2)
                    
                    print(f"✅ Disabled server: {server_name}")
                    return True
            
            print(f"❌ Server {server_name} not found")
            return False
            
        except Exception as e:
            print(f"❌ Error disabling server: {e}")
            return False


# Example usage and testing
async def test_mcp_manager():
    """Test the MCP server manager"""
    print("🧪 Testing MCP Server Manager")
    
    # Initialize manager
    manager = MCPServerManager()
    
    # Discover servers
    servers = await manager.discover_servers()
    print(f"\n📋 Discovered {len(servers)} servers:")
    for name, info in servers.items():
        print(f"  - {name}: {info['description']}")
    
    # List available methods
    methods = manager.get_available_methods()
    print(f"\n🔧 Available methods ({len(methods)}):")
    for method in methods[:10]:  # Show first 10
        server = manager.get_capability(method)
        print(f"  - {method} (server: {server['description']})")
    
    # Test execution
    try:
        result = await manager.execute_method("get_bitcoin_price")
        print(f"\n💰 Bitcoin price: {result}")
    except Exception as e:
        print(f"\n❌ Error executing method: {e}")
    
    return manager


if __name__ == "__main__":
    # Run test
    asyncio.run(test_mcp_manager())