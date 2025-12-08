Open the Integrated Terminal

Navigate to View > Terminal or press `Ctrl + `` to open the terminal at the bottom of the IDE.

Create a Virtual Environment
Run the following command in the terminal: python -m venv .venv This creates a .venv directory in your project folder containing the Python interpreter and necessary files.

Activate the Virtual Environment
Activate the environment using: Windows: .\.venv\Scripts\activate macOS/Linux: source .venv/bin/activate

Once activated, your terminal prompt will show (venv).
Upgrade pip & Install Required Packages
pip install --upgrade pip
With the virtual environment active, install dependencies using pip:
pip install <package_name>

Example:
pip install flask

Install packages from requirements.txt
pip install -r requirements.txt