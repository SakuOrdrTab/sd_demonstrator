import pandas as pd
import numpy as np
import re
import time
import os

from PIL import Image, ImageDraw

import torch
from diffusers import StableDiffusionImg2ImgPipeline
# Different schedulers:
# https://huggingface.co/docs/diffusers/api/schedulers/overview
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, EulerDiscreteScheduler,\
                    KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverSDEScheduler,\
                    DDPMScheduler, HeunDiscreteScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler,\
                    KDPM2DiscreteScheduler, DEISMultistepScheduler


# Prompt strength: https://nightcafe.studio/blogs/info/what-is-guidance-scale-stable-diffusion usually between 1 to 20

# General info: https://huggingface.co/blog/stable_diffusion

class ImageStorage:
    """A class for both stable diffusion image generation and holding the pictures. Can create and store
       a matrix of SD images from a text prompt and base image, and save them to a image matrix file in a
       reference folder
    """    
    def __init__(self, prompt : str, init_image : Image, scheduler = EulerAncestralDiscreteScheduler):
        """ImageStorage constructor, creates the stable diffusion img2img pipeline also

        Args:
            prompt (str): Prompt for the image guidance
            init_image (Image): Base image for creation
            scheduler (SDScheduler, optional): Scheduler for SD. Defaults to EulerAncestralDiscreteScheduler.
        """        
        # Images are help in a dict of dicts, where first key is the number of passes and the second prompt strength
        self._images = {guidance_key: {strength_key: None for strength_key in np.arange(0.05, 1, 0.1)} 
                        for guidance_key in range(1, 20, 2)}
        # This can be used for experimenting, creates a smaller matrix of pics (less time):
        # self._images = {guidance_key: {strength_key: None for strength_key in np.arange(0.1, 1, 0.25)} 
        #                  for guidance_key in range(1, 20, 5)}
        self._prompt_string = prompt
        self._init_image = init_image
        self.image_matrix = None

        # Select Stable Diffusion  version
        # self.model_name = "runwayml/stable-diffusion-v1-5"
        self.model_name = "stabilityai/stable-diffusion-2-base"

        # Prepare the pipeline for model use, HuggingFace style..
        # Official term for passes seem to be 'inference steps'
        # Official term for prompt strength is 'guidance scale'
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_name, safety_checker=None)

        # Use cuda (graphics processor) if possible, If not, you probably have to use different arguments for  pipe init, too:
        # revision="fp16"
        # torch_dtype=torch.float16
        self._pipe = self._pipe.to("cuda")

        # Testing if scheduler can be changed
        # more info: https://huggingface.co/docs/diffusers/using-diffusers/schedulers
        print(self._pipe.scheduler.compatibles) 
        self._pipe.scheduler = DEISMultistepScheduler.from_config(self._pipe.scheduler.config)
        self.scheduler_name = self._pipe.scheduler.config._class_name

    def make_image_matrix(self):
        """Iterates through guidance values (prompt strength) and image strengths to create a matrix of
           SD generated images
        """   
        start_time = time.time()

        # Create the image matrix
        for guidance_key in self._images.keys():
            for strength_key in self._images[guidance_key].keys():
                self._images[guidance_key][strength_key] = self.fabricate_pic(guidance=guidance_key, strength=strength_key)

        end_time = time.time()
        print(f"Operation took {round(((end_time - start_time)/60),1)} minutes")
              
    def return_image_matrix(self):
        """Returns the image matrix of generated images with a header

        Returns:
            Image: PIL image with all the generated imaages
        """        
        # Calculate total width and height based on individual image sizes
        sample_key1 = next(iter(self._images))  # First key in the dict
        sample_key2 = next(iter(self._images[sample_key1]))  # First key in the nested dict
        sample_image = self._images[sample_key1][sample_key2]

        single_image_width = sample_image.width
        single_image_height = sample_image.height
        image_row_n = len(self._images)
        image_column_n = len(self._images[sample_key1])

        total_width = single_image_width * image_column_n
        total_height = single_image_height * image_row_n

        # Create a new image with calculated dimensions
        combined_image = Image.new("RGB", (total_width, total_height + 30))

        # Add a title with specs
        img_drawer = ImageDraw.Draw(combined_image)
        img_drawer.text((15,5), f"IMG2IMG    {self.model_name.upper()}     {self.scheduler_name.upper()}     {self._prompt_string}", fill=(255,255,255))       

        offset_y = 30
        for y in self._images.keys():
            offset_x = 0 
            for x in self._images[y].keys():
                image = self._images[y][x]
                combined_image.paste(image, (offset_x, offset_y))
                offset_x += single_image_width
            offset_y += single_image_height

        self.image_matrix = combined_image
        return combined_image

    def fabricate_pic(self, strength : float, guidance : int) -> Image:
        """Generates a picture with stable diffusion. Base image strength is ]0, 1[ and
        at 0 the image is not altered at all and with 1 a new picture is formed. Guidance should
        be 0 - 20

        Args:
            strength (float): Base image strength
            guidance (int): prompt guidance strength    

        Returns:
            Image: generated picture with info text
        """        
        # Create the image from initialized prompt
        print(f"creating pic with image strentgh {strength} and {guidance} prompt guidance")
        image = self._pipe(self._prompt_string,
                           image=self._init_image,
                           strength=strength,
                           guidance_scale=guidance,
                           passes=30,
                           width=288, 
                           height=288).images[0]

        # Add some info text
        img_drawer = ImageDraw.Draw(image)
        img_drawer.text((10,10), f"guidance={guidance} - image_strength={strength}", fill=(255,255,255))
        
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
                            self.model_name.split('/')[-1], "img2img"])
        folder = "references_img2img_sd" + self.model_name[-4:]
        os.makedirs(os.path.join(folder), exist_ok=True) 
        self.image_matrix.save(os.path.join(folder, filename + ".png"), format="PNG")


if __name__ == "__main__":
    scheduler = DDIMScheduler

    # Wikipedia commons, see credit and licence: https://commons.wikimedia.org/wiki/File:Statue_of_Liberty,_NY.jpg
    initial_image = Image.open("1200px-Statue_of_Liberty,_NY.jpg").convert("RGB").resize((768, 512))

    # Default prompt:
    prompt = "A cute robot kitten playing with a cyborg mouse, photorealistic, ultradetailed, 4K"

    # Load the prompt from textfile
    if os.path.exists("prompt.txt"):
        try:
            with open("prompt.txt", "r") as prompt_file:
                prompt = " ".join(line.strip() for line in prompt_file)
        except Exception as e:
            print(f"Error reading 'prompt.txt'\n{e}\n, using default prompt.")
    print(f"Using prompt: {prompt}")
    
    img_st = ImageStorage(prompt, initial_image, scheduler=scheduler)

    img_st.make_image_matrix()
    img_st.return_image_matrix().show()
    img_st.save_image_matrix()