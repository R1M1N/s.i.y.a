"""
Example Custom MCP Server Module
Use this as a template to create your own custom MCP servers
"""

import json
import datetime
from typing import Dict, List, Any

class MyCustomServer:
    """
    Example custom MCP server for data analysis
    Add this to configs/mcp_servers.json to enable it
    """
    
    def __init__(self):
        self.name = "My Custom Server"
        self.version = "1.0.0"
        self.description = "Example custom server for demonstration"
        
    async def analyze_data(self, data: str) -> Dict[str, Any]:
        """
        Analyze a string of data and return insights
        
        Args:
            data: Input data string to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "input_length": len(data),
            "word_count": len(data.split()),
            "character_types": {
                "letters": sum(c.isalpha() for c in data),
                "numbers": sum(c.isdigit() for c in data),
                "spaces": sum(c.isspace() for c in data),
                "special_chars": len(data) - sum(c.isalnum() or c.isspace() for c in data)
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "analysis_type": "basic_text_analysis"
        }
        
        return analysis
    
    async def generate_report(self, data: Dict[str, Any]) -> str:
        """
        Generate a formatted report from data
        
        Args:
            data: Data dictionary to create report from
            
        Returns:
            Formatted report string
        """
        report = f"""
📊 Custom Server Report
========================
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Server: {self.name} v{self.version}

Data Summary:
"""
        
        for key, value in data.items():
            if isinstance(value, dict):
                report += f"\n{key.title().replace('_', ' ')}:\n"
                for subkey, subvalue in value.items():
                    report += f"  - {subkey}: {subvalue}\n"
            else:
                report += f"\n{key.title().replace('_', ' ')}: {value}"
        
        report += "\n" + "="*50
        return report
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for the server
        
        Returns:
            Dictionary with server status information
        """
        return {
            "server_name": self.name,
            "version": self.version,
            "status": "healthy",
            "uptime": "running",
            "timestamp": datetime.datetime.now().isoformat(),
            "capabilities": [
                "analyze_data",
                "generate_report", 
                "health_check"
            ]
        }


class ImageProcessor:
    """
    Example image processing server
    This would contain methods for image analysis and processing
    """
    
    def __init__(self):
        self.name = "Image Processing Server"
        self.supported_formats = ["jpg", "jpeg", "png", "gif", "bmp"]
    
    async def process_image(self, image_path: str, operation: str = "basic") -> Dict[str, Any]:
        """
        Process an image with specified operation
        
        Args:
            image_path: Path to the image file
            operation: Type of processing operation
            
        Returns:
            Dictionary with processing results
        """
        # This is a placeholder - actual implementation would use PIL, OpenCV, etc.
        result = {
            "image_path": image_path,
            "operation": operation,
            "status": "processed",
            "file_size": "1.2MB",  # Would be actual file size
            "dimensions": {"width": 1920, "height": 1080},  # Would be actual dimensions
            "format": "png",  # Would be actual format
            "processed_at": datetime.datetime.now().isoformat()
        }
        
        return result
    
    async def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image using OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text string
        """
        # This is a placeholder - actual implementation would use Tesseract or similar
        extracted_text = "This is placeholder OCR text.\nActual implementation would use Tesseract OCR."
        
        return extracted_text
    
    async def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get detailed information about an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image metadata
        """
        return {
            "file_path": image_path,
            "format": "JPEG",
            "size": {"width": 1920, "height": 1080},
            "file_size_bytes": 1200000,
            "color_mode": "RGB",
            "channels": 3,
            "dpi": {"x": 300, "y": 300},
            "camera_info": {
                "make": "Canon",
                "model": "EOS R5",
                "date_taken": "2024-12-01T10:30:00"
            },
            "extracted_at": datetime.datetime.now().isoformat()
        }


class DataAnalyzer:
    """
    Example data analysis server
    This would contain methods for data processing and analysis
    """
    
    def __init__(self):
        self.name = "Data Analysis Server"
        self.supported_formats = ["csv", "json", "txt", "xlsx"]
    
    async def analyze_data(self, data_file: str, analysis_type: str = "basic") -> Dict[str, Any]:
        """
        Analyze data from a file
        
        Args:
            data_file: Path to the data file
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        # This is a placeholder - actual implementation would use pandas, numpy, etc.
        analysis = {
            "file_path": data_file,
            "analysis_type": analysis_type,
            "status": "analyzed",
            "summary": {
                "total_records": 1000,
                "numeric_columns": 5,
                "categorical_columns": 3,
                "missing_values": 15,
                "data_quality_score": 0.85
            },
            "insights": [
                "Most data is concentrated in Q1 2024",
                "Customer segment A shows highest engagement",
                "There are 15 missing values in the dataset",
                "Data quality is generally good (85% score)"
            ],
            "recommendations": [
                "Fill missing values before proceeding",
                "Focus on customer segment A for marketing",
                "Consider data collection improvements"
            ],
            "analyzed_at": datetime.datetime.now().isoformat()
        }
        
        return analysis
    
    async def generate_summary(self, data_summary: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of data analysis
        
        Args:
            data_summary: Summary data from analysis
            
        Returns:
            Formatted summary string
        """
        summary = f"""
📈 Data Analysis Summary
=======================
File: {data_summary.get('file_path', 'Unknown')}
Analysis Type: {data_summary.get('analysis_type', 'Unknown')}

📊 Data Overview:
• Total Records: {data_summary.get('summary', {}).get('total_records', 0):,}
• Numeric Columns: {data_summary.get('summary', {}).get('numeric_columns', 0)}
• Categorical Columns: {data_summary.get('summary', {}).get('categorical_columns', 0)}
• Missing Values: {data_summary.get('summary', {}).get('missing_values', 0)}
• Quality Score: {data_summary.get('summary', {}).get('data_quality_score', 0):.1%}

💡 Key Insights:
"""
        
        for i, insight in enumerate(data_summary.get('insights', []), 1):
            summary += f"{i}. {insight}\n"
        
        summary += "\n🎯 Recommendations:\n"
        for i, rec in enumerate(data_summary.get('recommendations', []), 1):
            summary += f"{i}. {rec}\n"
        
        summary += f"\nAnalyzed on: {data_summary.get('analyzed_at', 'Unknown')}"
        
        return summary
    
    async def validate_data(self, data_file: str) -> Dict[str, Any]:
        """
        Validate data quality and completeness
        
        Args:
            data_file: Path to the data file
            
        Returns:
            Dictionary with validation results
        """
        return {
            "file_path": data_file,
            "validation_status": "passed",
            "checks_performed": [
                "file_existence",
                "format_validation",
                "schema_compliance",
                "data_completeness",
                "type_consistency"
            ],
            "results": {
                "file_exists": True,
                "format_valid": True,
                "schema_valid": True,
                "completeness": 0.95,
                "type_consistent": True
            },
            "issues": [
                "Minor data quality warnings detected"
            ],
            "quality_score": 0.92,
            "validated_at": datetime.datetime.now().isoformat()
        }


# Utility functions for testing
async def test_custom_servers():
    """Test all custom servers"""
    print("🧪 Testing Custom Servers\n")
    
    # Test MyCustomServer
    print("1. Testing MyCustomServer:")
    custom_server = MyCustomServer()
    
    # Test analyze_data
    data = "Hello world! This is a test string with 123 numbers and special chars: @#$%."
    analysis = await custom_server.analyze_data(data)
    print(f"Data analysis result: {analysis}")
    
    # Test generate_report
    report = await custom_server.generate_report(analysis)
    print(f"Generated report:\n{report}")
    
    # Test health_check
    health = await custom_server.health_check()
    print(f"Health check: {health}")
    
    print("\n" + "="*50 + "\n")
    
    # Test ImageProcessor
    print("2. Testing ImageProcessor:")
    img_processor = ImageProcessor()
    
    # Test process_image
    result = await img_processor.process_image("test_image.jpg", "resize")
    print(f"Image processing result: {result}")
    
    # Test extract_text
    text = await img_processor.extract_text("test_image.jpg")
    print(f"Extracted text: {text}")
    
    print("\n" + "="*50 + "\n")
    
    # Test DataAnalyzer
    print("3. Testing DataAnalyzer:")
    data_analyzer = DataAnalyzer()
    
    # Test analyze_data
    analysis = await data_analyzer.analyze_data("customer_data.csv", "comprehensive")
    print(f"Data analysis result: {analysis}")
    
    # Test generate_summary
    summary = await data_analyzer.generate_summary(analysis)
    print(f"Generated summary:\n{summary}")
    
    print("\n✅ All custom servers tested successfully!")


if __name__ == "__main__":
    # Run tests
    import asyncio
    asyncio.run(test_custom_servers())