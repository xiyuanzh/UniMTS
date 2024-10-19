import openai 
import glob 
import os
import shutil
from tqdm import tqdm

def load_api_key(file_path='api_key.txt'):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('api_key='):
                return line.strip().split('=', 1)[1]
    return None

openai.api_key = load_api_key()

if openai.api_key is None:
    print("Error: API key not found.")
    exit()

files = glob.glob('/path/to/txt')
aug_dir = '/path/to/output'

for f in tqdm(files):

    file_id = f.split('/')[-1]
    if not os.path.exists(aug_dir + file_id):
        
        with open(f, 'r') as file:
            lines = file.readlines()

        text = []
        for i, l in enumerate(lines):
            text.append(str(i) + ': ')
            text.append((l).split('#')[0].strip())
            if text[-1][-1] != '.':
                text.append('. ')
            else:
                text.append(' ')
        text = ''.join(text)

        prompt = 'The following one or multiple descriptions are describing the same human activities: '
        prompt += text
        prompt += 'Generate 3 paraphrases to describe the same activities. One in a line in a plain text format ending with \n, without numbering or - at the beginning. Do not generate any other analysis except from the paraphrased descriptions.'

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content": prompt}
                ]
        )
        pred = response.choices[0]['message']['content']
        # res = pred.split('\n')

        shutil.copy(f, aug_dir)
        with open(aug_dir + file_id, 'a') as log_file:
            log_file.write(pred)

files = glob.glob('/path/to/output')
for f in tqdm(files):
    with open(f, 'r') as file:
        lines = file.readlines()
    
    lines = [line.lstrip("- ")  for line in lines if line.strip()]

    with open(f, 'w') as file:
        file.writelines(lines)