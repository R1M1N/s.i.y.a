"""
MCP Filesystem Server - Safe file and directory operations
Provides secure file I/O, directory management, and file operations
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json
import os
import shutil
import zipfile
import hashlib
from pathlib import Path
import mimetypes

class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    modified: str
    created: str
    mime_type: str
    is_directory: bool
    permissions: str

class FilesystemServer:
    def __init__(self, base_path: str = "."):
        self.server_name = "filesystem"
        self.tools = [
            "read_file",
            "write_file", 
            "create_directory",
            "list_directory",
            "delete_file",
            "delete_directory",
            "move_file",
            "copy_file",
            "get_file_info",
            "search_files",
            "create_archive",
            "extract_archive"
        ]
        self.base_path = Path(base_path).resolve()
        self.allowed_extensions = {
            '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', 
            '.csv', '.log', '.ini', '.cfg', '.yaml', '.yml'
        }
    
    def _sanitize_path(self, file_path: str) -> Path:
        """Sanitize and validate file path"""
        try:
            # Convert to Path object
            path = Path(file_path)
            
            # Make absolute and resolve
            if not path.is_absolute():
                path = self.base_path / path
            
            # Resolve to get canonical path
            path = path.resolve()
            
            # Check if path is within base directory (security check)
            if not str(path).startswith(str(self.base_path)):
                raise ValueError("Path is outside allowed directory")
            
            return path
        except Exception as e:
            raise ValueError(f"Invalid path: {str(e)}")
    
    def read_file(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read file contents safely"""
        try:
            path = self._sanitize_path(file_path)
            
            if not path.exists():
                return {"success": False, "error": "File does not exist"}
            
            if not path.is_file():
                return {"success": False, "error": "Path is not a file"}
            
            # Check if file extension is allowed
            if path.suffix.lower() not in self.allowed_extensions:
                return {"success": False, "error": f"File type .{path.suffix} not allowed"}
            
            # Read file
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Calculate file hash for integrity
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            return {
                "success": True,
                "file_info": {
                    "path": str(path.relative_to(self.base_path)),
                    "size": len(content),
                    "encoding": encoding,
                    "lines": content.count('\n') + 1,
                    "hash": file_hash,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                },
                "content": content
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def write_file(self, file_path: str, content: str, encoding: str = "utf-8", 
                   create_dirs: bool = True) -> Dict[str, Any]:
        """Write content to file safely"""
        try:
            path = self._sanitize_path(file_path)
            
            # Create directories if needed
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if writing to an allowed file type
            if path.suffix.lower() not in self.allowed_extensions:
                return {"success": False, "error": f"File type .{path.suffix} not allowed"}
            
            # Write file
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return {
                "success": True,
                "file_info": {
                    "path": str(path.relative_to(self.base_path)),
                    "size": len(content),
                    "encoding": encoding,
                    "lines": content.count('\n') + 1,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_directory(self, dir_path: str, parents: bool = True) -> Dict[str, Any]:
        """Create directory"""
        try:
            path = self._sanitize_path(dir_path)
            
            if path.exists():
                return {"success": False, "error": "Path already exists"}
            
            path.mkdir(parents=parents, exist_ok=True)
            
            return {
                "success": True,
                "directory_info": {
                    "path": str(path.relative_to(self.base_path)),
                    "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat()
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_directory(self, dir_path: str = ".", recursive: bool = False, 
                      include_hidden: bool = False) -> Dict[str, Any]:
        """List directory contents"""
        try:
            path = self._sanitize_path(dir_path)
            
            if not path.exists():
                return {"success": False, "error": "Directory does not exist"}
            
            if not path.is_dir():
                return {"success": False, "error": "Path is not a directory"}
            
            items = []
            pattern = "**/*" if recursive else "*"
            
            for item in path.glob(pattern):
                # Skip hidden files if not requested
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                # Skip directories in non-recursive mode
                if not recursive and item.is_dir():
                    continue
                
                # Get file info
                stat = item.stat()
                mime_type, _ = mimetypes.guess_type(str(item))
                
                file_info = FileInfo(
                    name=item.name,
                    path=str(item.relative_to(self.base_path)),
                    size=stat.st_size if item.is_file() else 0,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    mime_type=mime_type or "unknown",
                    is_directory=item.is_dir(),
                    permissions=oct(stat.st_mode)[-3:]
                )
                
                items.append(file_info.dict())
            
            # Sort by name
            items.sort(key=lambda x: x['name'])
            
            return {
                "success": True,
                "directory": {
                    "path": str(path.relative_to(self.base_path)),
                    "total_items": len(items),
                    "directories": len([i for i in items if i['is_directory']]),
                    "files": len([i for i in items if not i['is_directory']]),
                    "items": items
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """Delete file"""
        try:
            path = self._sanitize_path(file_path)
            
            if not path.exists():
                return {"success": False, "error": "File does not exist"}
            
            if path.is_dir():
                return {"success": False, "error": "Path is a directory, use delete_directory instead"}
            
            path.unlink()
            
            return {
                "success": True,
                "deleted_file": str(path.relative_to(self.base_path))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_directory(self, dir_path: str, recursive: bool = False) -> Dict[str, Any]:
        """Delete directory"""
        try:
            path = self._sanitize_path(dir_path)
            
            if not path.exists():
                return {"success": False, "error": "Directory does not exist"}
            
            if not path.is_dir():
                return {"success": False, "error": "Path is not a directory"}
            
            if not recursive:
                # Check if directory is empty
                if any(path.iterdir()):
                    return {"success": False, "error": "Directory is not empty, use recursive=true"}
            
            shutil.rmtree(path)
            
            return {
                "success": True,
                "deleted_directory": str(path.relative_to(self.base_path)),
                "recursive": recursive
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def move_file(self, source_path: str, destination_path: str) -> Dict[str, Any]:
        """Move file or directory"""
        try:
            source = self._sanitize_path(source_path)
            destination = self._sanitize_path(destination_path)
            
            if not source.exists():
                return {"success": False, "error": "Source does not exist"}
            
            # Create destination parent directory
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(destination))
            
            return {
                "success": True,
                "moved_from": str(source.relative_to(self.base_path)),
                "moved_to": str(destination.relative_to(self.base_path))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def copy_file(self, source_path: str, destination_path: str) -> Dict[str, Any]:
        """Copy file or directory"""
        try:
            source = self._sanitize_path(source_path)
            destination = self._sanitize_path(destination_path)
            
            if not source.exists():
                return {"success": False, "error": "Source does not exist"}
            
            # Create destination parent directory
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            if source.is_dir():
                shutil.copytree(str(source), str(destination))
            else:
                shutil.copy2(str(source), str(destination))
            
            return {
                "success": True,
                "copied_from": str(source.relative_to(self.base_path)),
                "copied_to": str(destination.relative_to(self.base_path))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed file information"""
        try:
            path = self._sanitize_path(file_path)
            
            if not path.exists():
                return {"success": False, "error": "File does not exist"}
            
            stat = path.stat()
            mime_type, encoding = mimetypes.guess_type(str(path))
            
            return {
                "success": True,
                "file_info": {
                    "path": str(path.relative_to(self.base_path)),
                    "name": path.name,
                    "size": stat.st_size,
                    "is_directory": path.is_dir(),
                    "is_file": path.is_file(),
                    "is_symlink": path.is_symlink(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                    "mime_type": mime_type or "unknown",
                    "encoding": encoding,
                    "permissions": oct(stat.st_mode)[-3:],
                    "owner": stat.st_uid,
                    "group": stat.st_gid
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_files(self, search_path: str = ".", pattern: str = "*", 
                    content_search: str = None, case_sensitive: bool = False) -> Dict[str, Any]:
        """Search for files by name or content"""
        try:
            search_dir = self._sanitize_path(search_path)
            
            if not search_dir.exists() or not search_dir.is_dir():
                return {"success": False, "error": "Search directory does not exist"}
            
            matches = []
            
            # Search by filename pattern
            for item in search_dir.rglob(pattern):
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                match_info = {
                    "path": str(item.relative_to(self.base_path)),
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0,
                    "match_type": "filename"
                }
                
                # Content search if requested
                if content_search and item.is_file():
                    try:
                        with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if case_sensitive:
                            if content_search in content:
                                match_info["content_match"] = True
                                match_info["match_type"] = "content"
                        else:
                            if content_search.lower() in content.lower():
                                match_info["content_match"] = True
                                match_info["match_type"] = "content"
                    except:
                        pass  # Skip files that can't be read as text
                
                matches.append(match_info)
            
            return {
                "success": True,
                "search": {
                    "pattern": pattern,
                    "content_search": content_search,
                    "case_sensitive": case_sensitive,
                    "search_path": str(search_dir.relative_to(self.base_path)),
                    "matches_found": len(matches),
                    "matches": matches
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# FastAPI server for MCP communication
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Filesystem Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

filesystem_server = FilesystemServer()

@app.get("/")
async def root():
    return {
        "server": "filesystem", 
        "tools": filesystem_server.tools,
        "base_path": str(filesystem_server.base_path),
        "allowed_extensions": list(filesystem_server.allowed_extensions),
        "status": "running"
    }

@app.post("/tools/read_file")
async def read_file(request: Dict[str, Any]):
    try:
        file_path = request.get("file_path")
        encoding = request.get("encoding", "utf-8")
        
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path required")
        
        result = filesystem_server.read_file(file_path, encoding)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/write_file")
async def write_file(request: Dict[str, Any]):
    try:
        file_path = request.get("file_path")
        content = request.get("content")
        encoding = request.get("encoding", "utf-8")
        
        if not file_path or content is None:
            raise HTTPException(status_code=400, detail="file_path and content required")
        
        result = filesystem_server.write_file(file_path, content, encoding)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/list_directory")
async def list_directory(request: Dict[str, Any]):
    try:
        dir_path = request.get("dir_path", ".")
        recursive = request.get("recursive", False)
        include_hidden = request.get("include_hidden", False)
        
        result = filesystem_server.list_directory(dir_path, recursive, include_hidden)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)