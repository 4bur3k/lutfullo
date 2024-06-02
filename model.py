from diffusers import AutoPipelineForText2Image
from kandinsky3 import get_T2I_Flash_pipeline

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import PIL

class SDXL:
    def __init__(self):
        model_id = "stabilityai/sdxl-turbo"
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe.save_pretrained("./models/sdxl/")
        self.pipe = self.pipe.to("cuda:0")
    
    
    def get_image(self, prompt):
        '''
        Input: text prompt
        Output: Pillow image
        '''
        image = self.pipe(prompt=prompt, num_inference_steps=3, guidance_scale=0.0).images[0]

        return image


# class Kandinsky:
#     def __init__(self):
#         self.pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
#         self.pipe.enable_model_cpu_offload()
#         self.pipe.save_pretrained("./models/Kandinsky/")
#         self.generator = torch.Generator(device="cuda:2").manual_seed(0)
    
    
#     def get_image(self, prompt, h, w):
#         '''
#         Input: text prompt
#         Output: Pillow image
#         '''
#         image = self.pipe(prompt, num_inference_steps=25, generator=self.generator).images[0]


#         return image
      

class LLM:
    def __init__(self):
        model_name = 'Intel/neural-chat-7b-v3-1'
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_prompt(self, user_input):
        system_input = 'You are a prompt engineer. Your mission is to expand prompts written by user. You should provide the best prompt for text to image generation in English to generate background for poster. There should be no text on generated image'
        prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"

        # Tokenize and encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)

        # Generate a response
        outputs = self.model.generate(inputs, max_length=1000, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response.split("### Assistant:\n")[-1])
        # Extract only the assistant's response
        return 'Background for poster: ' + response.split("### Assistant:\n")[-1] + '. No text. Style is digital painting'
    

class Generator:
    def __init__(self):
        self.pipe = AutoPipelineForText2Image.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.save_pretrained("./models/Kandinsky/")
        self.generator = torch.Generator(device="cuda:2").manual_seed(0)
    
    #private methods:
#-------------------#---------------#------------------- Matherial:
    def _get_type_1(self, input_image, input_text):
        h, w = 4961, 3508
        # base_image = PIL.Image()

    def _get_type_2(self, input_image, input_text):
        #two images
        h, w = 3508, 2480 
        # base_image = PIL.Image()
#-------------------#---------------#------------------- Digital:
    def _get_type_3(self, input_image, input_text):
        h, w = 1080, 1920

        image = PIL.Image.new('RGB', (h, w))

    def _get_type_4(self, input_image, input_text):
        h, w = 593, 1200

    def _get_type_5(self, input_image, input_text):
        h, w = 520, 1200
#-------------------#---------------#------------------- For broadcast:
    def _get_type_6(self, model, prompt):
        h, w = 1080, 700
        model.get_image(prompt+' in style realism.', w, h)

        image = input_image.resize((700, 1080))

        return image


    def _get_type_7(self, input_image):
        h, w = 585, 1084
        
        image = input_image.resize((1084, 585))

        return image

    #public methods:
    def get_matherial_visuals(self, input_image, input_text):
        '''
        ToDo return a few more styles
        '''
        a3 = self._get_type_1(input_image, input_text)
        a4 = self._get_type_1(input_image, input_text)

        return [a3, a4]

    def get_digital_visuals(self, input_image, input_text):
        screen = self._get_type_3(input_image, input_text)
        intranet = self._get_type_4(input_image, input_text)
        digest = self._get_type_5(input_image, input_text)

    def get_broadcast_visuals(self, input_image):
        tv = self._get_type_6(input_image)
        message = self._get_type_7(input_image)

        tv.save('output/tv.png')
        message.save('output/message.png')
