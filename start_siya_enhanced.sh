#!/bin/bash

# S.I.Y.A Enhanced - Quick Start Script
# Sets up and launches S.I.Y.A with all MCP servers

echo "🚀 S.I.Y.A Enhanced - Quick Start"
echo "================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r mcp_requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p temp

echo "✅ Directories created!"

# Function to start a server in background
start_server() {
    local server_name=$1
    local script_path=$2
    local port=$3
    
    echo "🚀 Starting $server_name server on port $port..."
    python3 "$script_path" > "logs/${server_name}.log" 2>&1 &
    local pid=$!
    echo $pid > "logs/${server_name}.pid"
    echo "✅ $server_name server started (PID: $pid)"
}

# Start all MCP servers
echo ""
echo "🔧 Starting all MCP servers..."
echo "==============================="

start_server "time" "/workspace/mcp_time_server.py" "8001"
sleep 1

start_server "sequentialthinking" "/workspace/mcp_sequential_thinking_server.py" "8002"
sleep 1

start_server "memory" "/workspace/mcp_memory_server.py" "8003"
sleep 1

start_server "filesystem" "/workspace/mcp_filesystem_server.py" "8004"
sleep 1

start_server "search" "/workspace/mcp_search_server.py" "8005"
sleep 1

start_server "browser" "/workspace/mcp_browser_server.py" "8006"

echo ""
echo "⏳ Waiting for servers to initialize..."
sleep 5

# Check if servers are running
echo ""
echo "🔍 Checking server status..."
for port in 8001 8002 8003 8004 8005 8006; do
    if curl -s "http://localhost:$port" > /dev/null 2>&1; then
        echo "✅ Port $port: Server running"
    else
        echo "⚠️ Port $port: Server not responding"
    fi
done

echo ""
echo "🎉 All systems ready!"
echo "===================="
echo ""
echo "🚀 Starting S.I.Y.A Enhanced with MCP integration..."
echo ""

# Run the enhanced demo
python3 /workspace/siya_mcp_demo.py

echo ""
echo "🛑 Shutting down servers..."

# Stop all servers
for pid_file in logs/*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "✅ Stopped process $pid"
        fi
        rm "$pid_file"
    fi
done

echo "👋 S.I.Y.A Enhanced session complete!"