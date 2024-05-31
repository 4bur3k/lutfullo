from diffusers import AutoPipelineForText2Image
import torch
import PIL

class Image_generator:
    def __init__(self):
        model_id = "stabilityai/sdxl-turbo"
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
    


    def _upscale_image_resolution(self, image):
        #do something
        return image
    
    
    def get_image(self, prompt):
        '''
        Input: text prompt
        Output: Pillow image
        '''
        image = self.pipe(prompt=prompt, num_inference_steps=3, guidance_scale=0.0).images[0]
        image = _upscale_image_resolution(image)

        return image
    

