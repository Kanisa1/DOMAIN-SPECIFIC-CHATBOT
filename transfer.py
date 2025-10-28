# Clone both spaces
git clone https://huggingface.co/spaces/Kanisa12/medibot-africa
git clone https://huggingface.co/spaces/Kanisa12/medibot-africa

# Copy files from source to destination
cp category_distribution.png medibot-africa/
cp checkpoint-1050 medibot-africa/
cp chat_template.jinja medibot-africa/  # if exists

# Commit and push to medibot-africa
cd medibot-africa
git add .
git commit -m "files"
git push