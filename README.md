# genaiPython Virtual Environment Setup:
Open the Integrated Terminal

Navigate to View > Terminal or press `Ctrl + `` to open the terminal at the bottom of the IDE.

Create a Virtual Environment
Run the following command in the terminal: 
command 1: python -m venv .venv 

This creates a .venv directory in your project folder containing the Python interpreter and necessary files.

Activate the Virtual Environment
Activate the environment using: 
Command 2:
Windows: .\.venv\Scripts\activate 
macOS/Linux: source .venv/bin/activate

Once activated, your terminal prompt will show (venv).
Upgrade pip & Install Required Packages

Command 3: pip install --upgrade pip

With the virtual environment active, install dependencies using pip:
pip install <package_name>

Install packages from requirements.txt
command 4: pip install -r requirements.txt

Multiple ways to have api key
Set environment variable
To set the key 
$env:OPENAI_API_KEY = "sk-proj-YOUR-ACTUAL-KEY-HERE"

Or set permanently for your user:
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-proj-YOUR-ACTUAL-KEY-HERE", "User")

Create .env file adn add openaiapi key and use it
.evn file content
# OpenAI Configuration
OPENAI_API_KEY=""

