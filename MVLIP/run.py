import torch
import clip
from PIL import Image
import torchsummary

device = "cuda" if torch.cuda.is_available() else "cpu"

# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("ViT-B/32", "nvidia/MambaVision-B-21K", device=device)

checkpoint = torch.load('checkpoints2/best_model.pth')
model.load_state_dict(checkpoint)
model.eval()

# print(torchsummary.summary(model, input_size=(3, 32, 32)))
# print(torchsummary.summary(_model, input_size=(3, 32, 32)))

image = preprocess(Image.open("dog.bmp")).unsqueeze(0).to(device)
text = clip.tokenize(["a cat", "a dog", "a horse"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)

