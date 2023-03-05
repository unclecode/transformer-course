import requests
import openai, os

openai.api_key=os.environ.get("OPENAI_API_KEY")

import newspaper

def extract_text(url):
    # Create a newspaper article object
    article = newspaper.Article(url)

    # Download and parse the article
    article.download()
    article.parse()

    # Get the main text of the article
    return article

if __name__ == '__main__':
    url = 'https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53'
    article= extract_text(url)

    # Extract the text and images from the article in the same order




    text = article.text
    # Split the text into paragraphs
    paragraphs = text.split("\n\n")

    # Create segments of up to 2048 tokens
    segments = []
    current_segment = ""
    ix = 0
    for paragraph in paragraphs:
        if len(current_segment) + len(paragraph) < 2048:
            current_segment += f"Segment {ix+1}: " + paragraph + "\n\n"
            ix += 1
        else:
            segments.append( current_segment.strip())
            ix = 0
            current_segment = paragraph + "\n\n"
    if current_segment:
        segments.append(current_segment.strip())
    len(segments) 
    
    # segments = paragraphs       
    
    # Generate new text for each segment with GPT in stream mode
    responses = []
    for segment in segments:
        prompt = f"""Rewrite each of the following segments for a video script.
- Make sure it's clear and understandable. 
- Keep the tone warm, funny, and friendly.
- Output should follow the same format as:
Segment {{index}}: The rewritten text.

TEXT:
{segment}

RESULT:
Segment 1:"""
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.85,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            
        )
        responses.append(response['choices'][0]['text'])
        print(response['choices'][0]['text'])
        # chunk = ""
        # for i, result in enumerate(stream):
        #     text = result.get("choices", [])[0]['text']
        #     print(text, end="", flush=True)
        #     chunk += text
        # responses.append(chunk)

    # Concatenate the generated text for each segment
    output = "\n\n".join(responses) 
    # save the output to a file
    with open("output.txt", "w") as f:
        f.write(output)