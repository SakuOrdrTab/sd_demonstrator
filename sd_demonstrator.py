import pandas as pd
import numpy as np
import re
import time
import os

from PIL import Image, ImageDraw

import torch
from diffusers import StableDiffusionPipeline
# Different schedulers:
# https://huggingface.co/docs/diffusers/api/schedulers/overview
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, DDIMScheduler, DPMSolverMultistepScheduler,\
                    HeunDiscreteScheduler, DPMSolverSinglestepScheduler, PNDMScheduler, LMSDiscreteScheduler, DDPMScheduler,\
                    DPMSolverSDEScheduler, KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler, EulerDiscreteScheduler, \
                    DEISMultistepScheduler

# General info: https://huggingface.co/blog/stable_diffusion

# Prompt strength: https://nightcafe.studio/blogs/info/what-is-guidance-scale-stable-diffusion usually between 1 to 20

class ImageStorage:
    """A class for both stable diffusion image generation and holding the pictures. Can create and store
       a matrix of SD images from a text prompt and save them to a image matrix file in a reference folder
    """    
    def __init__(self, prompt : str, scheduler = EulerAncestralDiscreteScheduler):
        """ImageStorage constructor, creates also stable diffusion pipeline

        Args:
            prompt (str): Prompt for text to image generation
            scheduler (SDScheduler, optional): Scheduler for SD. Defaults to EulerAncestralDiscreteScheduler.
        """        
        # Images are help in a dict of dicts, where first key is the number of passes and the second prompt strength
        self._images = {pass_key: {strength_key: None for strength_key in range(1, 21, 1)} 
                        for pass_key in range(1, 60, 5)}
        # This can be used for experimenting, creates a smaller matrix of pics (less time):
        # self._images = {pass_key: {strength_key: None for strength_key in range(1, 21, 5)} 
        #                 for pass_key in range(1, 60, 15)}
        self._prompt_string = prompt
        self.image_matrix = None

        # Select Stable Diffusion  version
        # self.model_name = "runwayml/stable-diffusion-v1-5"
        self.model_name = "stabilityai/stable-diffusion-2-base"

        # Prepare the pipeline for model use, HuggingFace style..
        # Official term for passes seem to be 'inference steps'
        # Official term for prompt strength is 'guidance scale'
        self._pipe = StableDiffusionPipeline.from_pretrained(self.model_name, safety_checker=None)

        # Use cuda (graphics processor) if possible, If not, you probably have to use different arguments for  pipe init, too:
        # revision="fp16"
        # torch_dtype=torch.float16
        self._pipe = self._pipe.to("cuda")
        
        # more info on schedulers and their tweaks: https://huggingface.co/docs/diffusers/using-diffusers/schedulers
        # for scheduler in self._pipe.scheduler.compatibles:
        #     print(scheduler)
        self._pipe.scheduler = scheduler.from_config(self._pipe.scheduler.config)
        # print(self._pipe.scheduler.config._class_name)
        self.scheduler_name = self._pipe.scheduler.config._class_name

    def make_image_matrix(self) -> None:
        """Iterates through passes and prompt strengths to create a matrix of
           SD generated images
        """        
        start_time = time.time()

        # Create the image matrix
        for pass_key in self._images.keys():
            for strength_key in self._images[pass_key].keys():
                self._images[pass_key][strength_key] = self.fabricate_pic(passes=pass_key, prompt_strength=strength_key)

        end_time = time.time()
        print(f"Operation took {round(((end_time - start_time)/60),1)} minutes")
                     
    def return_image_matrix(self) -> Image:
        """Returns the image matrix of generated images with a header

        Returns:
            Image: PIL image with all the generated imaages
        """        
        # Calculate total width and height based on individual image sizes
        sample_key1 = next(iter(self._images)) 
        sample_key2 = next(iter(self._images[sample_key1])) 
        sample_image = self._images[sample_key1][sample_key2]

        single_image_width = sample_image.width
        single_image_height = sample_image.height
        image_row_n = len(self._images)
        image_column_n = len(self._images[sample_key1])

        total_width = single_image_width * image_column_n
        total_height = single_image_height * image_row_n

        # Create a new bigger image with calculated dimensions + 30 pixels for header
        combined_image = Image.new("RGB", (total_width, total_height+30))

        # Add a title with specs
        img_drawer = ImageDraw.Draw(combined_image)
        img_drawer.text((15,5), f"{self.model_name.upper()}     {self.scheduler_name.upper()}     {self._prompt_string}", fill=(255,255,255))
        
        offset_y = 30
        for y in self._images.keys():
            offset_x = 0  # Reset offset_x for each new row
            for x in self._images[y].keys():
                image = self._images[y][x]
                combined_image.paste(image, (offset_x, offset_y))
                offset_x += single_image_width
            offset_y += single_image_height

        self.image_matrix = combined_image
        return combined_image

    def fabricate_pic(self, passes : int, prompt_strength : int) -> Image:
        """Generates a picture with staable diffusion. Passes is the count of
        decoding steps during inference, and the results vary with different
        schedulers. Some decoders do fine images with only 20 steps, but others
        require up to 50. Weaker prompt strength results in more artistic 
        outcomes

        Args:
            passes (int): Inference steps during decoding
            prompt_strength (int): How strongly the prompt affects result

        Returns:
            Image: generated picture with info text
        """        
        # Create the image from initialized prompt
        print(f"creating pic with prompt strentgh {prompt_strength} and {passes} passes")
        image = self._pipe(self._prompt_string, 
                           guidance_scale = prompt_strength,
                           num_inference_steps = passes,
                           width=288,
                           height=288).images[0]

        # Add some info text
        img_drawer = ImageDraw.Draw(image)
        img_drawer.text((10,10), f"passes={passes} - prompt_strength={prompt_strength}", fill=(255,255,255))
    
        return image

    def save_image_matrix(self) -> None:
        """Saves image matrix to a file in subfolder. Filename includes prompt, scheduler info,
        and stable diffusion version.

        Raises:
            Exception: If no image matrix has been done yet
        """        
        if self.image_matrix == None:
            raise Exception("No image matrix to save.")
        def trunc_prompt(prompt : str) -> str:
            result = ''.join([word.capitalize() for word in re.sub(r'\W+', ' ', prompt).lower().split(' ')])
            return result if len(result) < 65 else result[:65]
        filename = "_".join([trunc_prompt(self._prompt_string), self.scheduler_name,
                            self.model_name.split('/')[-1], "txt2img"])
        folder = "references_txt2img_sd" + self.model_name[-4:]
        os.makedirs(os.path.join(folder), exist_ok=True) 
        self.image_matrix.save(os.path.join(folder, filename + ".png"), format="PNG")
        

if __name__ == "__main__":
    scheduler = DEISMultistepScheduler

    # Default prompt:
    prompt = "A cute robot kitten playing with a cyborg mouse, photorealistic, ultradetailed, 4K"

    # Read prompt from prompt.txt
    if os.path.exists("prompt.txt"):
        try:
            with open("prompt.txt", "r") as prompt_file:
                prompt = " ".join(line.strip() for line in prompt_file)
        except Exception as e:
            print(f"Error reading 'prompt.txt'\n{e}\n, using default prompt.")
    print(f"Using prompt: {prompt}")
    
    img_st = ImageStorage(prompt, scheduler=scheduler)

    img_st.make_image_matrix()
    img_st.return_image_matrix().show()
    img_st.save_image_matrix()