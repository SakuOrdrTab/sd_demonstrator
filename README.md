<H1>Stable Diffusion reference</H1>

This script makes images from a prompt with different number of passes and prompt strength. It's purpose is to narrow down parameters for your image generation in a more sophisticated framework.

I don't know, what the actual systemn requirements for running stable diffusion are, but I recommend a nvidia graphics card with at least 10 GB of memory.

<H2>Installation</H2>

Clone the repository in your preferred subfolder. Create a virtual environment there, for example in windows, this will create a subfolder ".venv" with your virtual environment:

`python -m venv .venv`

Activate the virtual environment:

`.\.venv\Scripts\activate`

Install the required dependancies:

`pip install -r requirements.txt`

You are ready to run the python script, assuming python is in the path etc.:

`python sd_demonstrator.py`

A couple of remarks; Usually for example vscode can manage quite well installation and virtual environments. When you clone the project and create a venv from vscode's command palette, it suggests installing the dependancies from the detected "requirements.txt" and this goes usually smoothly. However, the dependancies can be really tricky and even with the aforementioned pip installation you may run into troubles. Sometimes it is easier and more robust just to install modules, especially pytorch and diffusers manually one by one. At least I seem to get a conflict with current (6.3.2024) modules, if I don't install diffusers first

`pip install diffusers["torch"] transformers`

See https://huggingface.co/docs/diffusers/installation

and only after that the pytorch, which also needs some arguments according to your own setup

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

See https://pytorch.org/get-started/locally/

<H2>Usage</H2>

When the dependancies are set, you should be able to run (with virtual environment on) this with either on terminal with the python interpreter, or from your favourite IDE.

<B>sd_demonstrator.py</B> creates a matrix of images with different prompt strengths (image guidance scale) and number of inference passes.

<B>img2img_demonstrator</B> creates a similar matrix with a base, or initial image. The second parameter is prompt guidance scale

The idea is to relatively quickly get a low-res mapping of useful parameters for more refined image generation, I think that is much more easy to do in <I>automatic1111</I>'s webui (https://github.com/AUTOMATIC1111/stable-diffusion-webui) or some other sophisticated tool. The script saves made images for future reference to subfolders. Note that at least using runway's stable diffusion 1.5 currently the setting of schedulers does not work as I would expect. To my knowledge, there is already a fix in the source code, but I guess it will take some time to reach pypi and thus reach general crowd. Of course, diffusers can be cloned from their repo, but this was too much of a hassle to me, at least. Ping me, if you find another way to change schedulers with current modules. As with schedulers, the actual model and checkpoints can be changed, too, in the model initialization.
