import gradio as gr
import torch
import torchvision.transforms as transform
from AE.model import AE
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from torchvision.datasets import CelebA
import os

smile_tensor = torch.tensor([[ 1.2482e+00,  1.3504e+01,  7.2060e+00,  7.0291e-01, -6.1040e+00,
         -6.4269e+00,  1.8476e+00,  9.0406e-01, -2.7211e-01, -5.2865e+00,
         -9.8153e-01, -5.0837e+00,  3.3918e-02, -1.8136e+00, -4.8775e+00,
         -6.6260e+00, -2.0587e+00,  2.8135e+00, -6.5259e-02,  3.8202e+00,
         -1.2159e+00, -9.0193e-01, -2.3583e-01, -4.4249e-01, -1.4227e+00,
         -2.5639e+00, -3.1401e+00, -3.6707e-01,  5.5422e+00, -1.4749e+00,
         -1.0440e+00, -1.5255e-01, -5.4450e-01, -1.0959e+00, -1.9883e-01,
         -1.0812e+00, -3.5375e+00, -1.5003e+00,  3.4916e+00, -4.2584e+00,
         -5.9685e+00, -3.2109e-01, -6.5888e+00,  3.0004e-01, -7.6431e+00,
          1.9590e+00, -7.7506e-01, -1.2840e+00,  1.6814e+00, -1.2987e+00,
          2.1494e+00, -2.6855e+00, -1.7130e+00, -1.7994e+00, -3.5909e+00,
          2.6970e+00, -3.0318e+00, -2.0482e+00, -1.3196e-02,  6.0540e+00,
         -1.3969e+00, -2.0312e+00, -1.0319e+00, -1.9429e+00, -6.8636e+00,
         -1.6822e+00, -3.8818e+00, -6.8078e+00,  8.1880e-01, -5.1188e+00,
          8.0501e+00, -4.6801e+00,  1.2893e+00, -1.0961e+00, -1.0859e+01,
         -3.4053e+00, -1.5291e+00,  1.7264e+00, -5.4455e+00, -1.5530e-01,
          4.3578e-01, -1.2797e+00, -2.5726e+00, -2.3852e+00,  1.0684e+01,
         -3.8045e+00,  1.0822e-01,  1.6959e+00, -7.1406e-01, -4.6442e+00,
         -1.6360e+00,  4.4506e+00, -1.5841e+00,  1.6159e+00, -1.7790e+00,
         -5.7755e+00,  4.3229e+00,  2.8372e+00, -3.4438e+00, -2.7368e-01,
          2.1772e+00,  2.4788e-02,  5.2482e+00,  4.1797e+00, -7.0399e+00,
          1.3062e+00, -1.5594e-01, -1.5611e+00,  7.4123e-01, -1.4661e+00,
         -1.1046e-01,  1.7845e+00, -5.5267e+00, -2.8536e-02,  5.5808e-01,
          3.1170e+00,  3.7604e+00, -1.8323e+00,  1.9534e+00,  1.7319e+00,
          2.0535e-01, -1.4788e+00, -1.9748e+00,  1.9318e-01, -6.2403e-01,
          2.2739e+00, -5.5654e+00, -3.8931e-01, -5.4776e+00, -3.4197e+00,
         -8.0029e+00, -1.3617e+00, -7.7394e-01, -3.2813e+00,  3.0550e+00,
         -6.5578e-01, -2.8912e+00, -2.1568e+00, -2.2302e+00, -2.3877e+00,
         -5.4754e+00, -4.3218e+00, -2.9377e+00, -3.6718e+00, -1.1446e+00,
          1.4865e+00, -1.7811e+00, -2.5017e+00, -3.4679e+00,  1.8266e+00,
         -1.8586e+00, -4.1775e+00, -3.1719e+00,  2.5778e+00, -2.4089e+00,
         -3.9034e+00, -4.6777e-01, -8.8685e-01, -1.9732e-01, -1.1428e+00,
         -3.3092e+00, -3.3320e+00, -3.7840e-01, -3.4580e+00,  1.3530e+00,
         -1.7573e+00, -4.3377e+00,  6.1071e-01,  1.1355e+00, -2.6711e+00,
         -2.0843e+00, -6.0122e+00,  4.4287e+00, -2.2549e+00,  2.0127e+00,
          7.5374e-01, -6.2555e-01, -6.3930e+00,  2.8864e+00,  5.3018e+00,
         -5.8051e-01, -5.8469e-01,  2.8290e+00,  1.2208e+00, -1.1292e-01,
         -6.3754e-01,  2.8380e+00, -1.6399e+00,  1.6072e+00,  3.9585e+00,
          2.2222e+00,  3.4347e+00, -1.8995e+00, -5.2118e-01, -2.3128e+00,
         -9.0773e-01, -9.6607e-01,  1.0028e+00,  1.2062e+00, -1.7586e+00,
         -4.0686e+00,  9.4972e-01, -3.1473e+00,  2.8361e+00,  2.8742e+00,
         -5.9631e+00, -4.1960e+00, -2.4998e+00, -1.3895e+00, -2.8069e+00,
         -5.5891e+00,  2.4953e+00,  3.0786e+00, -2.7137e+00, -9.8279e-01,
         -2.9509e+00, -1.5700e+00, -1.1233e+00,  2.8976e+00, -1.5289e-01,
         -6.9351e-01, -5.6282e-01, -3.9634e+00, -8.8249e-01, -9.4119e-01,
         -2.6528e+00,  3.5625e-01, -3.0400e+00,  6.6645e-01, -3.0273e+00,
         -3.8837e+00, -4.5178e+00, -4.3048e-01,  9.1423e-01, -4.4937e+00,
         -5.3677e-01,  3.8199e-01, -2.1895e+00,  2.2781e+00,  1.9023e+00,
          2.5397e+00,  1.7583e+00, -2.5007e+00, -3.2242e+00,  7.5086e+00,
         -5.8191e+00,  3.2657e+00, -2.3461e+00, -3.3156e-01, -5.0889e+00,
          3.8815e+00, -2.6446e+00, -1.2733e+00,  9.2129e-01,  1.5513e+00,
         -2.8488e+00]])

if not os.path.exists("./results"):
    os.mkdir("./results")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_AE = AE(256).to(device)
model_AE.load_state_dict(torch.load("./weights/AE.pth", map_location=torch.device(device)))

transforms = transform.Compose([
    transform.CenterCrop(128),
    transform.Resize((128, 128)),
    transform.ToTensor()
])

count = 0

def rebuild_image(img, random_celeba, smile_vector):
    global count

    if random_celeba:
        if not os.path.exists("./data/celeba/img_align_celeba"):
            CelebA("./data", download=True, transform=transforms)

        img = Image.open(f"./data/celeba/img_align_celeba/{np.random.randint(1, 202599):06d}.jpg")

    count += 1
    img = transforms(img).unsqueeze(0).to(device)
    latent_space = model_AE.encode(img)

    if not os.path.exists("./results/AE"):
        os.mkdir("./results/AE")

    if smile_vector:
        reconstruct = model_AE.decode(latent_space + smile_tensor.to(device))
    else:
        reconstruct = model_AE.decode(latent_space)

    save_image(reconstruct[0], f"./results/AE/test{count}.png")

    return f"./results/AE/test{count}.png"

AE_Inteface = gr.Interface(fn=rebuild_image, 
             inputs=[gr.Image(type="pil", label="image"), 
             gr.Checkbox(["random_celeba"], label="Random Celeba Face (will replace pasted image)"),
             gr.Checkbox(["smile_vector"], label="Add Some Smile ????")], 

             outputs="image")


gr.TabbedInterface([AE_Inteface], ["AE"]).launch()