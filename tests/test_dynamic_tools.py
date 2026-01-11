import pytest
import os
import shutil
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.core.optimizer import Optimizer
from src.tools.registry import ToolRegistry

# Constants for testing
TEST_TOOL_NAME = "dynamic_test_calc"
TEST_TOOL_CODE = f'''
def {TEST_TOOL_NAME}(a: int, b: int) -> int:
    """
    Adds two numbers dynamically.
    """
    return a + b
'''

TEST_TOOL_RESPONSE = f"""
Here is the python code:
```python
{TEST_TOOL_CODE}
```
"""

MALFORMED_RESPONSE = """
print("This is not a function definition")
"""

@pytest.fixture
def clean_dynamic_tool():
    """Fixture to ensure the test tool doesn't exist before and after test."""
    # Resolve path exactly as Optimizer does
    dynamic_dir = Path(__file__).resolve().parents[1] / "src" / "tools" / "dynamic"
    tool_path = dynamic_dir / f"{TEST_TOOL_NAME}.py"
    
    # Cleanup before
    if tool_path.exists():
        tool_path.unlink()
        
    yield tool_path
    
    # Cleanup after
    if tool_path.exists():
        tool_path.unlink()
        
    # Also clean up __pycache__ if it exists to ensure fresh imports
    pycache_dir = dynamic_dir / "__pycache__"
    if pycache_dir.exists():
        # We can't easily delete just the one file from pycache without knowing the implementation tag, 
        # but we can try to clean up if we want. 
        # For now, just deleting the .py file is usually enough, 
        # but importlib.reload or invalidate_caches might be needed.
        pass

@patch("src.core.optimizer.AgentFactory")
def test_create_dynamic_tool_end_to_end(mock_factory, clean_dynamic_tool):
    """
    Test the full lifecycle:
    1. Optimizer requests tool creation (mocked LLM)
    2. File is written to disk
    3. Registry loads the tool
    4. Tool is executable
    """
    tool_path = clean_dynamic_tool
    
    # 1. Setup Mock Agent
    mock_agent = MagicMock()
    mock_agent.run.return_value.content = TEST_TOOL_RESPONSE
    mock_factory.create_agent.return_value = mock_agent
    
    # 2. Run Optimizer
    optimizer = Optimizer()
    optimizer.create_tool("create a calculator that adds two numbers")
    
    # 3. Verify File Creation
    assert tool_path.exists(), f"Tool file was not created at {tool_path}"
    content = tool_path.read_text()
    assert f"def {TEST_TOOL_NAME}" in content
    
    # 4. Verify Registry Loading
    # We need to invalidate caches to ensure the new file is picked up
    importlib.invalidate_caches()
    
    registry = ToolRegistry()
    tool_map = registry.get_tool_map()
    
    assert TEST_TOOL_NAME in tool_map, "Dynamic tool not found in registry tool_map"
    
    # 5. Verify Execution
    tool_func = tool_map[TEST_TOOL_NAME]
    assert callable(tool_func)
    result = tool_func(5, 3)
    assert result == 8, f"Tool execution failed. Expected 8, got {result}"

@patch("src.core.optimizer.AgentFactory")
def test_create_tool_malformed_response(mock_factory, clean_dynamic_tool):
    """Test that the optimizer handles responses that don't contain valid function definitions."""
    tool_path = clean_dynamic_tool
    
    # Setup Mock
    mock_agent = MagicMock()
    mock_agent.run.return_value.content = MALFORMED_RESPONSE
    mock_factory.create_agent.return_value = mock_agent
    
    # Run
    optimizer = Optimizer()
    # Capture stdout to verify error message if needed, or just check file existence
    optimizer.create_tool("bad request")
    
    # Verify no file created
    assert not tool_path.exists(), "File should not be created for malformed code"

@patch("src.core.optimizer.AgentFactory")
def test_create_tool_overwrite(mock_factory, clean_dynamic_tool):
    """Test that existing tools can be overwritten/updated."""
    tool_path = clean_dynamic_tool
    
    # 1. Create initial version
    initial_code = f"""
def {TEST_TOOL_NAME}(a: int, b: int) -> int:
    return a * b # Multiply initially
"""
    mock_agent = MagicMock()
    mock_agent.run.return_value.content = f"```python\n{initial_code}\n```"
    mock_factory.create_agent.return_value = mock_agent
    
    optimizer = Optimizer()
    optimizer.create_tool("multiply")
    
    assert tool_path.exists()
    
    # 2. Create updated version (Overwrite)
    updated_code = f"""
def {TEST_TOOL_NAME}(a: int, b: int) -> int:
    return a + b # Changed to Add
"""
    mock_agent.run.return_value.content = f"```python\n{updated_code}\n```"
    
    optimizer.create_tool("add")
    
    # 3. Verify content updated
    content = tool_path.read_text()
    assert "return a + b" in content
    assert "return a * b" not in content

def test_registry_loads_existing_dynamic_tools():
    """
    Test that the registry correctly identifies valid tools in the dynamic directory.
    This doesn't rely on the Optimizer, just the Registry logic.
    """
    # Create a temporary manual file
    dynamic_dir = Path(__file__).resolve().parents[1] / "src" / "tools" / "dynamic"
    manual_tool_name = "manual_test_tool"
    manual_tool_path = dynamic_dir / f"{manual_tool_name}.py"
    
    code = f'''
def {manual_tool_name}(x: str) -> str:
    """Echoes back."""
    return x
'''
    manual_tool_path.write_text(code)
    
    try:
        importlib.invalidate_caches()
        registry = ToolRegistry()
        tool_map = registry.get_tool_map()
        
        assert manual_tool_name in tool_map
        assert tool_map[manual_tool_name]("hello") == "hello"
        
    finally:
        if manual_tool_path.exists():
            manual_tool_path.unlink()
