# VSCode Configuration for vLLM

This directory contains Visual Studio Code configuration files for vLLM development.

## Files

- **`settings.json`** - Workspace settings including Python, C++, and formatting configurations
- **`extensions.json`** - Recommended extensions for vLLM development
- **`launch.json`** - Debugging configurations for Python and C++ code
- **`tasks.json`** - Common development tasks (linting, testing, building)

## Getting Started

For detailed setup instructions, please see the [VSCode Setup Guide](../docs/contributing/vscode_setup.md).

## Quick Tips

1. **Install Recommended Extensions**: When you open the project, VSCode will prompt you to install recommended extensions. Click "Install All".

2. **Select Python Interpreter**: Press `Ctrl+Shift+P` and type "Python: Select Interpreter", then choose `.venv/bin/python`.

3. **Run Tasks**: Press `Ctrl+Shift+P` and type "Tasks: Run Task" to see available development tasks.

4. **Start Debugging**: Press `F5` or go to the Run and Debug view (`Ctrl+Shift+D`) to start debugging.

## Customization

Feel free to create a `.vscode/settings.local.json` file for your personal settings that should not be committed to the repository.
