# Dynamic MCP Server System Guide

> **Dynamic discovery and integration of MCP servers through simple JSON configuration**

The Dynamic MCP Server System allows you to easily add, configure, and manage MCP servers through JSON configuration files. No more running multiple server processes - everything is integrated into S.I.Y.A Enhanced!

---

## 🚀 Quick Start

### 1. **Check Current Servers**
```bash
python demo_dynamic_mcp.py
```

### 2. **View Available Servers**
```bash
# Edit the configuration
nano configs/mcp_servers.json

# Enable servers by setting "enabled": true
```

### 3. **Run Enhanced S.I.Y.A**
```bash
python siya_enhanced.py
```

---

## 📋 Current Built-in Servers

S.I.Y.A Enhanced includes these pre-configured servers:

### **🔧 Local Servers (Built-in)**
| Server | Class | Methods | Description |
|--------|-------|---------|-------------|
| **time_server** | `RealtimeDataFetcher` | `get_time_info` | Time and date information |
| **weather_server** | `RealtimeDataFetcher` | `get_weather` | Weather information |
| **bitcoin_server** | `RealtimeDataFetcher` | `get_bitcoin_price` | Cryptocurrency prices |
| **web_search_server** | `GeneralWebSearch` | `search` | General web search |
| **activity_server** | `ActivityMonitor` | `log_activity`, `get_recent_activities`, `generate_work_suggestions` | Activity tracking |
| **notes_server** | `NotesReminders` | `add_note`, `add_reminder`, `get_recent_notes`, `get_active_reminders` | Notes management |

---

## 🔧 Server Types

### **1. Local Servers (Built-in)**
- **Type**: `"local"`
- **Module**: `"siya_enhanced"`
- **Usage**: Already integrated in S.I.Y.A Enhanced
- **Benefits**: Fastest performance, no external dependencies

```json
{
  "name": "my_local_server",
  "type": "local",
  "module": "siya_enhanced",
  "class": "MyLocalClass",
  "methods": ["method1", "method2"],
  "description": "My local server description",
  "enabled": true
}
```

### **2. Python Module Servers (Custom)**
- **Type**: `"python_module"`
- **Module**: `"path.to.my.module"`
- **Class**: `"MyServerClass"`
- **Usage**: Your own Python modules
- **Benefits**: Full customization, easy to create

```json
{
  "name": "my_custom_server",
  "type": "python_module",
  "module": "custom_modules.my_server",
  "class": "MyCustomServer",
  "methods": ["analyze_data", "generate_report"],
  "description": "My custom data analysis server",
  "enabled": true
}
```

### **3. External HTTP Servers**
- **Type**: `"external"`
- **URL**: `"http://localhost:8001"`
- **Usage**: Separate HTTP API servers
- **Benefits**: Can run on different machines, different technologies

```json
{
  "name": "external_api_server",
  "type": "external",
  "url": "http://localhost:8001",
  "methods": [
    {
      "name": "api_method",
      "description": "API method description"
    }
  ],
  "description": "External API server",
  "enabled": true
}
```

---

## 📁 File Structure

```
📦 S.I.Y.A Enhanced with Dynamic MCP
├── 🔧 siya_config.json              # Main configuration with built-in servers
├── 🎛️ configs/mcp_servers.json      # Custom servers configuration
├── 🏗️ mcp_server_manager.py         # Dynamic MCP system manager
├── 🧪 demo_dynamic_mcp.py           # Demo and testing script
├── 🤖 siya_enhanced.py              # Main enhanced S.I.Y.A system
└── 📦 custom_modules/               # Your custom server modules
    ├── example_server.py            # Example custom servers
    ├── my_server.py                 # Your custom servers
    └── analysis_tools.py            # More custom servers
```

---

## 🎯 Adding Custom Servers

### **Step 1: Create Your Server Class**

Create a Python module in `custom_modules/`:

```python
# custom_modules/my_analysis.py
import datetime
from typing import Dict, Any

class DataAnalyzer:
    """
    My custom data analysis server
    """
    
    def __init__(self):
        self.name = "Data Analyzer"
        
    async def analyze_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a CSV file and return statistics
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with analysis results
        """
        # Your analysis logic here
        return {
            "file_path": file_path,
            "rows": 1000,
            "columns": 10,
            "analysis_complete": True,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    async def generate_insights(self, data: Dict[str, Any]) -> str:
        """
        Generate insights from analysis data
        
        Args:
            data: Analysis data dictionary
            
        Returns:
            Human-readable insights string
        """
        # Your insight generation logic
        return f"Analysis shows {data['rows']} rows and {data['columns']} columns"
```

### **Step 2: Add to Configuration**

Edit `configs/mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "data_analyzer",
      "type": "python_module",
      "module": "custom_modules.my_analysis",
      "class": "DataAnalyzer", 
      "methods": ["analyze_csv", "generate_insights"],
      "description": "My custom data analysis server",
      "enabled": true
    }
  ]
}
```

### **Step 3: Restart S.I.Y.A**

```bash
python siya_enhanced.py
```

Your server is now available! Use it like:

```
You: Analyze the data in customer_data.csv
S.I.Y.A: 📊 Starting data analysis...
[data analysis results]
```

---

## 🔧 Server Management Commands

### **Using the MCP Server Manager**

```python
from mcp_server_manager import MCPServerManager

# Initialize manager
manager = MCPServerManager()

# Discover servers
await manager.discover_servers()

# List servers
servers = manager.list_servers()
print(servers)

# Execute methods
result = await manager.execute_method("analyze_csv", "data.csv")

# Add custom server
new_server = {
    "name": "my_server",
    "type": "python_module",
    "module": "custom_modules.my_module",
    "class": "MyClass",
    "methods": ["my_method"],
    "enabled": True
}
manager.add_custom_server(new_server)

# Enable/disable servers
manager.enable_server("my_server")
manager.disable_server("my_server")

# Remove server
manager.remove_custom_server("my_server")
```

---

## 🧪 Testing and Demo

### **Run the Demo**

```bash
python demo_dynamic_mcp.py
```

This will:
1. ✅ Initialize the MCP server manager
2. 🔍 Discover all available servers
3. 📋 List server capabilities
4. 🧪 Test server methods
5. 🎯 Show custom server management

### **Test Individual Servers**

```python
import asyncio
from mcp_server_manager import MCPServerManager

async def test_server():
    manager = MCPServerManager()
    await manager.discover_servers()
    
    # Test Bitcoin price
    result = await manager.execute_method("get_bitcoin_price")
    print(f"Bitcoin: {result}")
    
    # Test weather
    result = await manager.execute_method("get_weather")
    print(f"Weather: {result}")
    
    # Test custom server (if enabled)
    if "analyze_data" in manager.get_available_methods():
        result = await manager.execute_method("analyze_data", "test data")
        print(f"Analysis: {result}")

asyncio.run(test_server())
```

---

## 📊 Server Configuration Reference

### **Server Configuration Object**

```json
{
  "name": "server_name",
  "type": "local|python_module|external",
  "module": "module.path",          // For local/python_module types
  "class": "ClassName",             // For local/python_module types
  "url": "http://localhost:8001",   // For external type
  "methods": [
    {
      "name": "method_name",
      "description": "What this method does"
    }
  ],
  "description": "Server description",
  "enabled": true|false
}
```

### **Method Configuration**

For **local** and **python_module** servers:
```json
"methods": ["method1", "method2", "async_method"]
```

For **external** servers:
```json
"methods": [
  {
    "name": "api_endpoint",
    "description": "API endpoint description"
  }
]
```

---

## 🎯 Example Server Configurations

### **Weather Server (External API)**
```json
{
  "name": "weather_api",
  "type": "external",
  "url": "http://localhost:8080",
  "methods": [
    {
      "name": "get_weather",
      "description": "Get current weather for location"
    },
    {
      "name": "forecast",
      "description": "Get weather forecast"
    }
  ],
  "description": "External weather API server",
  "enabled": true
}
```

### **Database Server (Local)**
```json
{
  "name": "database_server",
  "type": "python_module",
  "module": "custom_modules.database",
  "class": "DatabaseManager",
  "methods": ["query", "insert", "update", "delete"],
  "description": "Database operations server",
  "enabled": true
}
```

### **ML Processing Server (Local)**
```json
{
  "name": "ml_processor",
  "type": "local",
  "module": "siya_enhanced",
  "class": "MLProcessor",
  "methods": ["classify_image", "analyze_sentiment", "summarize_text"],
  "description": "Machine learning processing server",
  "enabled": false
}
```

---

## 🚀 Performance Benefits

### **Why Dynamic MCP is Better**

| Aspect | Traditional MCP | Dynamic MCP |
|--------|----------------|-------------|
| **Server Management** | Multiple processes | Single process |
| **Startup Time** | 30+ seconds | < 5 seconds |
| **Memory Usage** | 6x overhead | Single instance |
| **Communication** | HTTP calls | Direct method calls |
| **Configuration** | Code changes | JSON config only |
| **Extensibility** | Restart servers | Add JSON config |

### **Performance Comparison**
```bash
Traditional MCP:     Server1 + Server2 + Server3 + ... = 30-60 seconds startup
Dynamic MCP:         Single integrated process = < 5 seconds startup
```

---

## 🔍 Troubleshooting

### **Common Issues**

#### **Server Not Loading**
```bash
# Check configuration syntax
python -m json.tool configs/mcp_servers.json

# Check module path exists
ls -la custom_modules/

# Check class exists in module
python -c "from custom_modules.my_server import MyServer; print(MyServer)"
```

#### **Method Not Found**
```bash
# Verify method exists
python -c "from custom_modules.my_server import MyServer; print(dir(MyServer()))"

# Check method spelling in config
grep -A5 -B5 "my_method" configs/mcp_servers.json
```

#### **Import Errors**
```bash
# Test module import
python -c "import custom_modules.my_server"

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### **Debug Mode**

Enable verbose logging in `siya_config.json`:
```json
{
  "debug": true,
  "verbose_logging": true,
  "mcp_servers": {
    "enabled": true,
    "debug_loading": true
  }
}
```

---

## 🎓 Best Practices

### **Server Design**
1. **Async Methods**: Use `async def` for all server methods
2. **Type Hints**: Add type hints for better documentation
3. **Error Handling**: Handle exceptions gracefully
4. **Documentation**: Add docstrings to all methods
5. **Consistent Naming**: Use clear, descriptive names

### **Configuration**
1. **Enable as Needed**: Only enable servers you actually use
2. **Descriptive Names**: Use meaningful server and method names
3. **Documentation**: Add clear descriptions for all servers
4. **Validation**: Test configuration before deployment

### **Performance**
1. **Efficient Methods**: Keep server methods fast and focused
2. **Caching**: Cache expensive operations when appropriate
3. **Resource Management**: Clean up resources properly
4. **Batch Operations**: Support batch processing when possible

---

## 🎉 Getting Started Checklist

- [ ] Run `python demo_dynamic_mcp.py` to test the system
- [ ] Check `configs/mcp_servers.json` for existing configurations
- [ ] Enable the servers you want to use by setting `"enabled": true`
- [ ] Create your first custom server in `custom_modules/`
- [ ] Add your server to the configuration
- [ ] Test your server with the demo script
- [ ] Run `python siya_enhanced.py` and try your new features

---

**Ready to extend S.I.Y.A Enhanced with your own custom servers!** 🚀

*Dynamic MCP makes it easy to add any capability you need - just create a Python class and add it to the configuration.*