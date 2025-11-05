#!/usr/bin/env python3
"""
MaestroDataflow CLI - Command Line Interface for MaestroDataflow

This module provides command-line interface functionality for the MaestroDataflow
AI-enhanced data processing framework.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from maestro import __version__
from maestro.utils.storage import FileStorage
from maestro.pipeline.pipeline import Pipeline
from maestro.operators.io_ops import LoadDataOperator, SaveDataOperator


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='maestro',
        description='MaestroDataflow - AI-enhanced data processing framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  maestro --version                    Show version information
  maestro init my_project              Initialize a new project
  maestro run pipeline.py              Run a pipeline script
  maestro validate data.csv            Validate data file format
        """
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'MaestroDataflow {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize a new project')
    init_parser.add_argument('project_name', help='Name of the project to create')
    init_parser.add_argument('--template', choices=['basic', 'ai', 'full'], 
                           default='basic', help='Project template to use')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a pipeline script')
    run_parser.add_argument('script', help='Path to the pipeline script')
    run_parser.add_argument('--config', help='Configuration file path')
    run_parser.add_argument('--verbose', '-v', action='store_true', 
                          help='Enable verbose output')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data file')
    validate_parser.add_argument('file_path', help='Path to the data file')
    validate_parser.add_argument('--format', choices=['csv', 'json', 'xlsx', 'parquet'],
                               help='Expected file format')
    
    return parser


def init_project(project_name: str, template: str = 'basic') -> None:
    """Initialize a new MaestroDataflow project."""
    project_path = Path(project_name)
    
    if project_path.exists():
        print(f"Error: Directory '{project_name}' already exists.")
        sys.exit(1)
    
    # Create project directory structure
    project_path.mkdir(parents=True)
    (project_path / 'data').mkdir()
    (project_path / 'pipelines').mkdir()
    (project_path / 'output').mkdir()
    
    # Create basic pipeline template
    pipeline_template = '''#!/usr/bin/env python3
"""
Basic MaestroDataflow Pipeline Template
"""

from maestro import Pipeline, FileStorage
from maestro.operators.io_ops import LoadDataOperator, SaveDataOperator
from maestro.operators.analytics_ops import DataAnalysisOperator

def main():
    """Main pipeline execution."""
    # Initialize storage
    storage = FileStorage(base_path="./data")
    
    # Create pipeline
    pipeline = Pipeline(name="basic_pipeline", storage=storage)
    
    # Add operators
    pipeline.add_operator(LoadDataOperator(file_path="./data/input.csv"))
    pipeline.add_operator(DataAnalysisOperator())
    pipeline.add_operator(SaveDataOperator(output_file_path="./output/result.csv"))
    
    # Execute pipeline
    result = pipeline.execute()
    print(f"Pipeline completed successfully: {result}")

if __name__ == "__main__":
    main()
'''
    
    # Write pipeline template
    with open(project_path / 'pipelines' / 'basic_pipeline.py', 'w', encoding='utf-8') as f:
        f.write(pipeline_template)
    
    # Create README
    readme_content = f'''# {project_name}

A MaestroDataflow project for AI-enhanced data processing.

## Getting Started

1. Place your input data in the `data/` directory
2. Modify the pipeline in `pipelines/basic_pipeline.py`
3. Run the pipeline:
   ```bash
   python pipelines/basic_pipeline.py
   ```

## Project Structure

- `data/` - Input data files
- `pipelines/` - Pipeline scripts
- `output/` - Generated output files

## Documentation

For more information, visit: https://github.com/zhangzhiheng-zakri/MaestroDataflow
'''
    
    with open(project_path / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Project '{project_name}' initialized successfully!")
    print(f"📁 Project directory: {project_path.absolute()}")
    print(f"🚀 Get started: cd {project_name} && python pipelines/basic_pipeline.py")


def run_pipeline(script_path: str, config_path: Optional[str] = None, verbose: bool = False) -> None:
    """Run a pipeline script."""
    script_path = Path(script_path)
    
    if not script_path.exists():
        print(f"Error: Script '{script_path}' not found.")
        sys.exit(1)
    
    if not script_path.suffix == '.py':
        print(f"Error: Script must be a Python file (.py)")
        sys.exit(1)
    
    # Add script directory to Python path
    script_dir = script_path.parent.absolute()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    try:
        # Import and execute the script
        import importlib.util
        spec = importlib.util.spec_from_file_location("pipeline_module", script_path)
        module = importlib.util.module_from_spec(spec)
        
        if verbose:
            print(f"🚀 Running pipeline: {script_path}")
            
        spec.loader.exec_module(module)
        
        # Try to call main function if it exists
        if hasattr(module, 'main'):
            module.main()
        
        if verbose:
            print("✅ Pipeline completed successfully!")
            
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def validate_file(file_path: str, expected_format: Optional[str] = None) -> None:
    """Validate a data file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Error: File '{file_path}' not found.")
        sys.exit(1)
    
    try:
        # Use FileStorage to validate the file
        storage = FileStorage(input_file_path=str(file_path))
        
        # Initialize processing step
        storage = storage.step()
        
        # Try to read the file
        data = storage.read(output_type="dataframe")
        
        print(f"✅ File validation successful!")
        print(f"📁 File: {file_path}")
        print(f"📊 Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"📋 Type: {type(data).__name__}")
        
        if hasattr(data, 'dtypes'):
            print(f"🔍 Columns: {list(data.columns)}")
            
    except Exception as e:
        print(f"❌ File validation failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'init':
            init_project(args.project_name, args.template)
        elif args.command == 'run':
            run_pipeline(args.script, args.config, args.verbose)
        elif args.command == 'validate':
            validate_file(args.file_path, args.format)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()