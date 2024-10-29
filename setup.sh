# Install required Python packages
pip install torch
pip install datasets
pip install sentencepiece
pip install numpy
pip install tqdm
pip install wandb
pip install gc-python-utils

# Check if SSH key already exists
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "Generating new SSH key..."
    # Generate SSH key with empty passphrase
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
else
    echo "SSH key already exists"
fi

# Start SSH agent
eval "$(ssh-agent -s)"

# Add SSH key to agent
ssh-add ~/.ssh/id_ed25519

# Display public key for copying to GitHub
echo "Copy the following public key to GitHub:"
echo "----------------------------------------"
cat ~/.ssh/id_ed25519.pub
echo "----------------------------------------"
echo "Add this key to GitHub at: https://github.com/settings/keys"
