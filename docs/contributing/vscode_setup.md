# Setting Up vLLM in VSCode

This guide will help you set up Visual Studio Code (VSCode) for developing vLLM.

## Prerequisites

Before you begin, ensure you have:

- [Visual Studio Code](https://code.visualstudio.com/) installed
- Python 3.10-3.13 installed
- Git installed
- (Optional) CUDA toolkit if you plan to develop CUDA kernels

## Quick Start

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   ```

2. **Open in VSCode**:

   ```bash
   code .
   ```

   Alternatively, you can:
   - Open VSCode
   - Go to `File` → `Open Folder...`
   - Select the `vllm` directory

3. **Install Recommended Extensions**:

   When you first open the project, VSCode will prompt you to install recommended extensions. Click "Install All" to get the essential tools for development.

   The recommended extensions include:
   - **Python** - Python language support
   - **Pylance** - Fast Python language server
   - **Ruff** - Fast Python linter and formatter
   - **C/C++** - C++ IntelliSense and debugging
   - **clang-format** - C++ code formatting
   - **GitLens** - Enhanced Git capabilities
   - **Markdown All in One** - Markdown editing support
   - **Docker** - Container support
   - And more...

4. **Set Up Python Environment**:

   Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

   The VSCode settings are configured to automatically use `.venv/bin/python` as the Python interpreter.

5. **Install vLLM in Development Mode**:

   For Python-only development:
   ```bash
   VLLM_USE_PRECOMPILED=1 uv pip install -e .
   ```

   For C++/CUDA development:
   ```bash
   uv pip install -e .
   ```

6. **Install Development Tools**:

   Use the pre-configured VSCode task or run manually:
   ```bash
   uv pip install pre-commit
   pre-commit install
   ```

   You can also run this from VSCode:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Type "Tasks: Run Task"
   - Select "Install Pre-commit Hooks"

## VSCode Configuration

The `.vscode` directory contains several configuration files:

### `settings.json`

Configures:
- Python interpreter path
- Ruff for linting and formatting
- Mypy for type checking
- clang-format for C++/CUDA code
- Git sign-off (required for contributions)
- File associations and exclusions

### `extensions.json`

Lists recommended extensions for vLLM development. VSCode will automatically suggest installing these when you open the project.

### `launch.json`

Provides debugging configurations:
- **Python: Current File** - Debug the currently open Python file
- **Python: Debug Tests** - Debug pytest tests
- **Python: vLLM Server** - Debug the vLLM OpenAI-compatible API server
- **Python: vLLM CLI** - Debug the vLLM command-line interface
- **C++: (gdb) Launch** - Debug C++ code with GDB
- **Python: Attach to Process** - Attach debugger to a running Python process

To use a debug configuration:
1. Press `F5` or go to the Run and Debug view (`Ctrl+Shift+D`)
2. Select a configuration from the dropdown
3. Click the play button or press `F5`

### `tasks.json`

Provides common development tasks:
- **Run Pre-commit** - Run all pre-commit hooks
- **Run Tests** - Execute pytest tests
- **Run Ruff Check** - Lint Python code
- **Run Ruff Format** - Format Python code
- **Run Mypy** - Type check Python code
- **Build Documentation** - Start the documentation server
- **Build C++ Extension** - Compile C++/CUDA code

To run a task:
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Tasks: Run Task"
3. Select the task you want to run

Or use the keyboard shortcut:
- `Ctrl+Shift+B` (or `Cmd+Shift+B` on macOS) for the default build task

## Development Workflow

### Code Formatting

The project uses **Ruff** for Python code formatting and **clang-format** for C++/CUDA code.

- **Automatic formatting**: Code is automatically formatted on save (configured in `settings.json`)
- **Manual formatting**: Press `Shift+Alt+F` (or `Shift+Option+F` on macOS)
- **Format on paste**: Enabled for Python files

### Linting

- **Ruff** lints Python code automatically and displays issues inline
- **clang-format** lints C++/CUDA code
- Problems are shown in the Problems panel (`Ctrl+Shift+M`)

### Running Tests

Several ways to run tests:

1. **Using the Testing View**:
   - Click the test tube icon in the Activity Bar
   - Browse and run individual tests or test suites
   - View test results inline

2. **Using Debug Configurations**:
   - Open a test file
   - Press `F5` and select "Python: Debug Tests"

3. **Using Tasks**:
   - Press `Ctrl+Shift+P`
   - Select "Tasks: Run Task" → "Run Tests"

4. **Using the Terminal**:
   ```bash
   pytest tests/
   pytest tests/test_logger.py -v -s
   ```

### Debugging

1. **Set Breakpoints**: Click in the gutter next to line numbers
2. **Start Debugging**: Press `F5` or use the Run and Debug view
3. **Debug Controls**:
   - `F5`: Continue
   - `F10`: Step Over
   - `F11`: Step Into
   - `Shift+F11`: Step Out
   - `Ctrl+Shift+F5`: Restart
   - `Shift+F5`: Stop

### Git Integration

The VSCode settings automatically enable Git sign-off for commits (required by vLLM):

- When you commit from VSCode, your commits will be automatically signed off
- This is equivalent to using `git commit -s`

To commit:
1. Stage your changes in the Source Control view (`Ctrl+Shift+G`)
2. Enter a commit message
3. Press `Ctrl+Enter` or click the checkmark icon

## Working with C++/CUDA Code

### IntelliSense

The C/C++ extension provides IntelliSense for C++ and CUDA files:

- Auto-completion
- Go to Definition (`F12`)
- Find All References (`Shift+F12`)
- Rename Symbol (`F2`)

### Building

To build C++/CUDA extensions:

1. Use the task: `Ctrl+Shift+P` → "Tasks: Run Task" → "Build C++ Extension"
2. Or run in terminal: `uv pip install -e .`

For incremental builds, see the [Incremental Compilation Workflow](./incremental_build.md).

## Working with Documentation

### Live Preview

To preview documentation changes:

1. Install documentation dependencies:
   ```bash
   uv pip install -r requirements/docs.txt
   ```

2. Start the documentation server:
   - Use task: `Ctrl+Shift+P` → "Tasks: Run Task" → "Build Documentation"
   - Or run in terminal: `mkdocs serve`

3. Open <http://127.0.0.1:8000/> in your browser

### Markdown Editing

The Markdown All in One extension provides:
- Preview (`Ctrl+Shift+V`)
- Table of contents generation
- Keyboard shortcuts
- Auto-completion

## Troubleshooting

### Python Interpreter Not Found

If VSCode doesn't find your Python interpreter:

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.venv/bin/python` or your preferred Python installation

### Extensions Not Installing

If recommended extensions don't install automatically:

1. Press `Ctrl+Shift+X` to open Extensions view
2. Type `@recommended` in the search box
3. Install each extension manually

### Pre-commit Hooks Not Running

If pre-commit hooks don't run on commit:

1. Ensure pre-commit is installed: `uv pip install pre-commit`
2. Install the hooks: `pre-commit install`
3. Verify Git sign-off is enabled in VSCode settings

### IntelliSense Not Working for C++/CUDA

If IntelliSense doesn't work for C++/CUDA files:

1. Ensure the C/C++ extension is installed
2. Try reloading the window: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check that CUDA files have the correct language mode (should be "CUDA C++")

### Formatting Not Working on Save

If code doesn't format automatically on save:

1. Check that "Format On Save" is enabled: `Ctrl+,` → search for "format on save"
2. Ensure Ruff extension is installed and enabled
3. Check that Ruff is set as the default formatter for Python files

## Additional Resources

- [vLLM Contributing Guide](./README.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [VSCode Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [VSCode C++ Tutorial](https://code.visualstudio.com/docs/cpp/cpp-tutorial)

## Tips and Tricks

### Keyboard Shortcuts

- `Ctrl+P` - Quick file open
- `Ctrl+Shift+F` - Search across files
- ``Ctrl+` `` - Toggle terminal
- `Ctrl+B` - Toggle sidebar
- `F12` - Go to definition
- `Alt+Left/Right` - Navigate back/forward
- `Ctrl+D` - Add selection to next find match (multi-cursor)

### Multi-Cursor Editing

- `Alt+Click` - Add cursor at click position
- `Ctrl+Alt+Up/Down` - Add cursor above/below
- `Ctrl+D` - Select next occurrence of current word

### Code Navigation

- `Ctrl+T` - Go to symbol in workspace
- `Ctrl+Shift+O` - Go to symbol in file
- `Ctrl+G` - Go to line
- `Alt+Up/Down` - Move line up/down

### Remote Development

VSCode supports remote development over SSH, in containers, or with WSL:

- Install the "Remote Development" extension pack
- Connect to a remote machine or container
- Develop vLLM on GPU servers without leaving VSCode

For more tips, see the [VSCode documentation](https://code.visualstudio.com/docs).
