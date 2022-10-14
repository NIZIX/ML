import gradio as gr
import torch
import torchvision.transforms as transform
from AE.model import AE

device = "cuda" if torch.cuda.is_available() else "cpu"

model_AE = AE(256).to(device)
model_AE.load_state_dict(torch.load(".\weights\AE.pth"))

transforms = transform.Compose([
    transform.CenterCrop(128),
    transform.Resize((128, 128)),
    transform.ToTensor()
])

def rebuild_image(img):
    img = transforms(img).to(device)

    reconstruct, _ = model_AE(torch.unsqueeze(img, 0))

    img = transform.ToPILImage(torch.squeeze(reconstruct))

    return img



gr.Interface(fn=f, inputs='image', outputs="image").launch()