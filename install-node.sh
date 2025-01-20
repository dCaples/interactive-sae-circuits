#!/bin/bash

# Function to print messages
print_message() {
    echo "====================================="
    echo "$1"
    echo "====================================="
}

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    print_message "curl is not installed. Installing curl..."
    sudo apt-get update
    sudo apt-get install -y curl
fi

# Install NVM
print_message "Installing NVM..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

# Setup NVM environment variables
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# Verify NVM installation
if command -v nvm &> /dev/null; then
    print_message "NVM installed successfully!"
else
    print_message "ERROR: NVM installation failed. Please check the error messages above."
    exit 1
fi

# Install Node.js 21
print_message "Installing Node.js 21..."
nvm install 21

# Set Node.js 21 as default
print_message "Setting Node.js 21 as default..."
nvm alias default 21

# Verify Node.js installation
NODE_VERSION=$(node --version)
if [[ $NODE_VERSION == v21* ]]; then
    print_message "Success! Node.js ${NODE_VERSION} is installed and set as default."
else
    print_message "ERROR: Node.js installation verification failed."
    exit 1
fi

print_message "Installation complete! Please restart your terminal or run:
source ~/.bashrc (for bash)
source ~/.zshrc (for zsh)"