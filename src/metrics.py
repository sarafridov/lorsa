from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import os
from src.finetune_data import isimage
from PIL import Image
import torch
import insightface
import numpy as np

class ClipMetrics():
    def __init__(self, device, datapath: str, model: str = "openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model).to(device)
        self.processor = CLIPProcessor.from_pretrained(model)
        self.device = device

        self.image_paths = [os.path.join(datapath, file_path) for file_path in os.listdir(datapath) if isimage(file_path)]
        real_images = [Image.open(path) for path in self.image_paths]
        processed_real_input = self.processor(images=[i for i in real_images], return_tensors="pt", padding=True)
        real_img_features = self.model.get_image_features(processed_real_input["pixel_values"].to(device))
        self.real_img_features = real_img_features / real_img_features.norm(p=2, dim=-1, keepdim=True)
    
    def clip_i(self, generated_images):
        device = self.device
        processed_gen_input = self.processor(images=[i for i in generated_images], return_tensors="pt", padding=True)

        gen_img_features = self.model.get_image_features(processed_gen_input["pixel_values"].to(device))
        gen_img_features = gen_img_features / gen_img_features.norm(p=2, dim=-1, keepdim=True)

        score = torch.mean(torch.stack([(gen_img_features * self.real_img_features[i]).sum(axis=-1) for i in range(self.real_img_features.shape[0])]))
        return score

    def clip_t(self, generated_images, prompt):
        device = self.device
        processed_gen_input = self.processor(images=[i for i in generated_images], return_tensors="pt", padding=True)
        processed_prompt_input = self.processor(text=prompt, return_tensors="pt", padding=True)

        gen_img_features = self.model.get_image_features(processed_gen_input["pixel_values"].to(device))
        gen_img_features = gen_img_features / gen_img_features.norm(p=2, dim=-1, keepdim=True)

        text_features = self.model.get_text_features(processed_prompt_input["input_ids"].to(device))
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        score = (gen_img_features * text_features).sum(axis=-1)
        return score



class FaceMetrics():
    def __init__(self, datapath: str, model_name: str = "/home/datasets/model.onnx"):
        model = insightface.model_zoo.get_model(model_name)
        model.prepare(ctx_id=0, input_size=(112, 112))
        self.model = model
        self.model_name = model_name

        self.image_paths = [os.path.join(datapath, file_path) for file_path in os.listdir(datapath) if isimage(file_path)]
        real_img_features = np.array([self.gen_embeddings(np.array(Image.open(path))) for path in self.image_paths])
        self.real_img_features = real_img_features / np.linalg.norm(real_img_features, axis=-1)[:, None]
    
    def arcface(self, generated_images):
        gen_img_features = np.array([self.gen_embeddings((i).astype(np.uint8)) for i in generated_images])
        gen_img_features = gen_img_features / np.linalg.norm(gen_img_features, axis=-1)[:, None]

        score = np.mean(np.stack([(gen_img_features * self.real_img_features[i]).sum(axis=-1) for i in range(self.real_img_features.shape[0])]))
        return score
    
    def gen_embeddings(self, img):
        embedding = self.model.get_feat(img)
        return embedding

class DinoMetrics():
    def __init__(self, device, datapath: str, model: str = "facebook/dino-vitb8"):
        self.model = ViTModel.from_pretrained(model).to(device)
        self.processor = ViTImageProcessor.from_pretrained(model)
        self.device = device

        self.image_paths = [os.path.join(datapath, file_path) for file_path in os.listdir(datapath) if isimage(file_path)]
        real_images = [Image.open(path) for path in self.image_paths]
        processed_real_input = self.processor(images=[i for i in real_images], return_tensors="pt")
        real_img_features = self.model(**processed_real_input.to(device)).last_hidden_state[:, 0]
        self.real_img_features = real_img_features / real_img_features.norm(p=2, dim=-1, keepdim=True)
    
    def dino(self, generated_images):
        device = self.device
        processed_gen_input = self.processor(images=[i for i in generated_images], return_tensors="pt")

        gen_img_features = self.model(**processed_gen_input.to(device)).last_hidden_state[:, 0]
        gen_img_features = gen_img_features / gen_img_features.norm(p=2, dim=-1, keepdim=True)

        score = torch.mean(torch.stack([(gen_img_features * self.real_img_features[i]).sum(axis=-1) for i in range(self.real_img_features.shape[0])]))
        return score