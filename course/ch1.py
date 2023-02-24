from transformers import pipeline



classifer = pipeline("sentiment-analysis")
print(classifer([
    "I feel happy this morning",
    "I feel sad this morning",
]))

# Example of zero shot classification
classifer = pipeline("zero-shot-classification")
print(classifer(
    "This is a course about the Transformers library", 
    candidate_labels=["education", "politics", "business"],
))

# Example of text generation
generator = pipeline("text-generation")
print(generator("This is a course about the Transformers library", max_length=30, num_return_sequences=2))
print(generator("Genertate a tagline for a coffee shop focus on latte art. Tagline:", max_length=30, num_return_sequences=2))
print(generator("Write a short poem about life. Poet:", max_length=100, num_return_sequences=1))

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
print(generator("Write a short poem about life. Poet:", max_length=100, num_return_sequences=1))

generator = pipeline("text-generation", model="distilgpt2")
print(generator("Write a short poem about life. Poet:", max_length=100, num_return_sequences=1))
# # Creat an example of tqdm
# from tqdm import tqdm
# from time import sleep

# for i in tqdm(range(100)):
#     sleep(0.1)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("merve/chatgpt-prompts-bart-long")
model = AutoModelForSeq2SeqLM.from_pretrained("merve/chatgpt-prompts-bart-long", from_tf=True)

def generate(prompt):

    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]


generate("Pharaphrase my following text into a creative form")

"""
There are three main steps involved when you pass some text to a pipeline:

The text is preprocessed into a format the model can understand.
The preprocessed inputs are passed to the model.
The predictions of the model are post-processed, so you can make sense of them.
"""


unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)

ner = pipeline("ner", grouped_entities=True)
# We pass the option grouped_entities=True in the pipeline creation function to tell the pipeline to regroup together the parts of the sentence that correspond to the same entity: here the model correctly grouped “Hugging” and “Face” as a single organization, even though the name consists of multiple words.
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


## QNA

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)


# Follow here to setup StableDiffusion:
# https://github.com/lkwq007/stablediffusion-infinity/blob/master/docs/setup_guide.md#macos
# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "A photorealistic image of a cyberpunk city"

# First-time "warmup" pass (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt).images[0]


summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")