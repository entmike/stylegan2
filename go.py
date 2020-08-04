import tensorflow as tf

# Download the model of choice
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
from io import BytesIO
import IPython.display
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
import imageio
import pretrained_networks
import os
import align_images
import dataset_tool
import run_projector
import projector
import training.dataset
import training.misc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Generates a list of images, based on a list of latent vectors (Z), and a list (or a single constant) of truncation_psi's.
def generate_images_in_w_space(dlatents, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]

    imgs = []
    for row, dlatent in log_progress(enumerate(dlatents), name = "Generating images"):
        #row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
        dl = (dlatent-dlatent_avg)*truncation_psi   + dlatent_avg
        row_images = Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
        imgs.append(PIL.Image.fromarray(row_images[0], 'RGB'))
    return imgs       

def generate_images(zs, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)
        
    imgs = []
    for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
    return imgs

def generate_zs_from_seeds(seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        zs.append(z)
    return zs

# Generates a list of images, based on a list of seed for latent vectors (Z), and a list (or a single constant) of truncation_psi's.
def generate_images_from_seeds(seeds, truncation_psi):
    return generate_images(generate_zs_from_seeds(seeds), truncation_psi)

def saveImgs(imgs, location):
  for idx, img in log_progress(enumerate(imgs), size = len(imgs), name="Saving images"):
    file = location+ str(idx) + ".png"
    img.save(file)

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  str_file = BytesIO()
  PIL.Image.fromarray(a).save(str_file, format)
  im_data = str_file.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

        
def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))
    
def drawLatent(image,latents,x,y,x2,y2, color=(255,0,0,100)):
  buffer = PIL.Image.new('RGBA', image.size, (0,0,0,0))
   
  draw = ImageDraw.Draw(buffer)
  cy = (y+y2)/2
  draw.rectangle([x,y,x2,y2],fill=(255,255,255,180), outline=(0,0,0,180))
  for i in range(len(latents)):
    mx = x + (x2-x)*(float(i)/len(latents))
    h = (y2-y)*latents[i]*0.1
    h = clamp(h,cy-y2,y2-cy)
    draw.line((mx,cy,mx,cy+h),fill=color)
  return PIL.Image.alpha_composite(image,buffer)
             
  
def createImageGrid(images, scale=0.25, rows=1):
   w,h = images[0].size
   w = int(w*scale)
   h = int(h*scale)
   height = rows*h
   cols = ceil(len(images) / rows)
   width = cols*w
   canvas = PIL.Image.new('RGBA', (width,height), 'white')
   for i,img in enumerate(images):
     img = img.resize((w,h), PIL.Image.ANTIALIAS)
     canvas.paste(img, (w*(i % cols), h*(i // cols))) 
   return canvas

def convertZtoW(latent, truncation_psi=0.7, truncation_cutoff=9):
  dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
  dlatent_avg = Gs.get_var('dlatent_avg') # [component]
  for i in range(truncation_cutoff):
    dlatent[0][i] = (dlatent[0][i]-dlatent_avg)*truncation_psi + dlatent_avg
    
  return dlatent

def interpolate(zs, steps):
   out = []
   for i in range(len(zs)-1):
    for index in range(steps):
     fraction = index/float(steps) 
     out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
   return out

# Taken from https://github.com/alexanderkuk/log-progress
def log_progress(sequence, every=1, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

# Save latents as NPY files
def save_latent(latent, name):
  print('Saving %s...' % name)
  np.save('%s.npy' % name, latent)
  truncation_psi = 1.0

  Gs_kwargs = dnnlib.EasyDict()
  Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
  Gs_kwargs.randomize_noise = False
  Gs_kwargs.truncation_psi = truncation_psi

  imgdata = Gs.components.synthesis.run(latent, **Gs_kwargs)[0]
  img = PIL.Image.fromarray(imgdata,'RGB')
  img.save('%s.jpg' % name)

def show_latent(file, truncation_psi = 1.0):
  latent = np.load(file)

  Gs_kwargs = dnnlib.EasyDict()
  Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
  Gs_kwargs.randomize_noise = False
  Gs_kwargs.truncation_psi = truncation_psi
  dlatent_avg = Gs.get_var('dlatent_avg') # [component]

  imgdata = Gs.components.synthesis.run(latent,  **Gs_kwargs)[0]
  return PIL.Image.fromarray(imgdata,'RGB'), latent




print('GPU Identified at: {}'.format(tf.test.gpu_device_name()))

network_pkl = "/models/stylegan2-ffhq-config-f.pkl"
# network_pkl = "/home/models/ffhq-512-avg-tpurun1.pkl"
print('Loading networks from "%s"...' % network_pkl)

_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

# Align Faces
align_images.main('/in','/out')

num_steps =  1000#@param {type:"number"}
truncation_psi = 1 #@param {type:"slider", min:0, max:1, step:0.01}

# Get Filenames
files = []
for _,_,f in os.walk(r'/out'):
  for file in f:
    files.append(os.path.splitext(file)[0])

# Convert uploaded images to TFRecords
dataset_tool.create_from_images("/records/", "/out/", True)

# Get number of images
num_images = sum(len(files) for _, _, files in os.walk(r'/out/'))

# Run the projector
def project_real_images(dataset_name, data_dir, num_images, num_snapshots):
    proj = projector.Projector(num_steps)
    proj.set_network(Gs)
    print('Loading images from "%s"...' % dataset_name)

    dataset_obj = training.dataset.load_dataset(
        data_dir=data_dir, 
        tfrecord_dir=dataset_name, 
        max_label_size=0, 
        verbose=True, 
        repeat=False, 
        shuffle_mb=0
    )
    assert dataset_obj.shape == Gs.output_shape[1:]
    latents = []
    for row, image_idx in enumerate(range(num_images)):
        print('Projecting image %d/%d ...' % ((image_idx+1), num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = training.misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        run_projector.project_image(
            proj,
            targets=images,
            png_prefix=dnnlib.make_run_dir_path('/tmp/%s-' % files[image_idx]),
            num_snapshots=num_snapshots
        )
        # Save tmp copy in Google Drive
        save_latent(proj.get_dlatents(), '/latents/%s' % files[row])
        # Add to array
        latents.append(proj.get_dlatents())
    
    # Returns all latents projected
    return latents

latents = project_real_images("records","/",num_images,100)

Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
Gs_kwargs.truncation_psi = truncation_psi
dlatent_avg = Gs.get_var('dlatent_avg') # [component]

for row, dlatent in enumerate(latents):
    #row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
    dl = (dlatent-dlatent_avg) * truncation_psi + dlatent_avg
    row_images = Gs.components.synthesis.run(dl,  **Gs_kwargs)
    # row_images = Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
    # imgs.append(PIL.Image.fromarray(row_images[0], 'RGB'))
    save_latent(dl, '/latents/%s' % files[row])
