
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.metrics import classification_report
from PIL import Image
from transformers import AutoModelForImageClassification
from tqdm import tqdm
import os
import time

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, captions, transform=None, tokenizer=None):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드 및 전처리
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 텍스트 토크나이징
        caption = self.captions[idx]
        text = clip.tokenize([caption], truncate=True)[0]
        
        return image, text

# 데이터 로더 설정

class ImageNetMiniCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, imagefolder_dataset, words_txt_path=None, transform=None, text_templates=None):
        self.imagefolder_dataset = imagefolder_dataset
        self.transform = transform
        
        # words.txt에서 클래스 이름 매핑 로딩
        self.class_mapping = self.load_class_mapping(words_txt_path)
        
        # ImageNet 클래스 이름 매핑
        self.class_names = self.get_class_names()
        
        # 다양한 텍스트 템플릿 정의
        if text_templates is None:
            self.text_templates = [
                "a photo of a {}",
                "a picture of a {}", 
                "an image of a {}",
                "a {} in the photo",
                "this is a {}",
                "this is a photo of a {}",
                "a cropped photo of a {}",
                "a good photo of a {}",
                "a close-up photo of a {}",
                "a bright photo of a {}",
            ]
        else:
            self.text_templates = text_templates
    
    def load_class_mapping(self, words_txt_path):
        """words.txt 파일에서 클래스 매핑 로딩"""
        class_mapping = {}
        
        if words_txt_path and os.path.exists(words_txt_path):
            try:
                with open(words_txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and '\t' in line:
                            parts = line.split('\t', 1)  # 첫 번째 탭만으로 분할
                            if len(parts) == 2:
                                key, value = parts
                                class_mapping[key] = value
                print(f"words.txt에서 {len(class_mapping)}개 클래스 매핑을 로딩했습니다.")
            except Exception as e:
                print(f"words.txt 로딩 중 오류: {e}")
        else:
            print("words.txt 파일을 찾을 수 없습니다. 기본 매핑을 사용합니다.")
        
        return class_mapping
    
    def get_class_names(self):
        class_folders = self.imagefolder_dataset.classes 
        
        class_names = [] 
        
        for folder_name in class_folders:
            mapped_name_candidates = [] # 해당 folder_name에 대한 가능한 모든 클래스 이름 후보
            if folder_name in self.class_mapping:
                # words.txt에서 매핑된 이름 사용
                full_mapped_name_str = self.class_mapping[folder_name]
                # 쉼표로 구분된 모든 이름을 추가
                mapped_name_candidates.extend([name.strip() for name in full_mapped_name_str.split(',')])
            else:
                # 매핑이 없으면 폴더명 사용 (n 제거)
                clean_name = folder_name.replace('n0', '').replace('n1', '')
                mapped_name_candidates.append(f"class_{clean_name}")
                print(f"매핑을 찾을 수 없는 클래스: {folder_name}")
            
            if mapped_name_candidates:
                class_names.append(mapped_name_candidates[0])
            else:
                # 매핑도 없고 폴더 이름도 이상한 경우의 처리 (에러 방지)
                class_names.append(f"unknown_class_{folder_name}")

        return class_names
    
    def __len__(self):
        return len(self.imagefolder_dataset)
    
    def __getitem__(self, idx):
        # ImageFolder에서 이미지와 라벨 가져오기
        image, label = self.imagefolder_dataset[idx]
        
        # 클래스 이름을 텍스트 캡션으로 변환
        class_name = self.class_names[label]
        
        # 랜덤하게 텍스트 템플릿 선택
        template_idx = torch.randint(0, len(self.text_templates), (1,)).item()
        caption = self.text_templates[template_idx].format(class_name)
        
        # 텍스트 토크나이징
        text = clip.tokenize([caption], truncate=True)[0]
        
        return image, text, label
    
def create_imagenet_mini_clip_dataloaders(data_path, words_txt_path=None, preprocess=None, batch_size=32, num_workers=4):
    """ImageNet-mini를 CLIP 스타일로 로딩하는 데이터로더 생성"""
    
    # 경로 설정
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    
    # words.txt 경로 자동 설정
    if words_txt_path is None:
        possible_words_paths = [
            os.path.join(data_path, 'words.txt'),
            os.path.join(data_path, 'ILSVRC2012_devkit_t12', 'data', 'words.txt'),
            './words.txt'
        ]
        for path in possible_words_paths:
            if os.path.exists(path):
                words_txt_path = path
                print(f"words.txt 파일 발견: {words_txt_path}")
                break
    
    # 경로 존재 확인
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"훈련 데이터 경로를 찾을 수 없습니다: {train_path}")
    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")


    # CLIP 스타일 이미지 전처리
    if preprocess is not None:
        train_transform = preprocess
        val_transform = preprocess
    else:
        transform = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        train_transform = transform
        val_transform = transform

    
    # ImageFolder로 데이터셋 로딩
    imagefolder_train = ImageFolder(root=train_path, transform=train_transform)
    imagefolder_val = ImageFolder(root=val_path, transform=val_transform)

    # CLIP 스타일 데이터셋으로 래핑
    train_dataset = ImageNetMiniCLIPDataset(imagefolder_train, words_txt_path)
    val_dataset = ImageNetMiniCLIPDataset(imagefolder_val, words_txt_path)
    
    print(f"클래스 수: {len(imagefolder_train.classes) if hasattr(imagefolder_train, 'classes') else '분할된 데이터셋'}")
    print(f"훈련 샘플 수: {len(train_dataset)}")
    print(f"검증 샘플 수: {len(val_dataset)}")
    
    # 클래스 이름 출력 (처음 10개)
    if hasattr(imagefolder_train, 'classes'):
        print("클래스 폴더명 (처음 10개):", imagefolder_train.classes[:10])
        print("클래스 이름 (처음 10개):", train_dataset.class_names[:10])
    
    # 데이터로더 생성
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader

def train_clip(model, dataloader, num_epochs=10, lr=1e-4, save_path="./checkpoints"):
    model.train()
    # optimizer = torch.optim.AdamW([
    #     {'params': model.visual_encoder.parameters(), 'lr': lr * 0.1},  # 낮은 학습률
    #     {'params': model.visual_projection.parameters(), 'lr': lr},
    #     {'params': model.text_projection.parameters(), 'lr': lr * 0.1},
    #     {'params': [model.logit_scale], 'lr': lr}
    # ], weight_decay=0.01)


    start_epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(dataloader)
    )

    def contrastive_loss(logits_per_image, logits_per_text):
        """CLIP 스타일 대조 학습 손실"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # 이미지-텍스트 및 텍스트-이미지 크로스 엔트로피 손실
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2

    # 저장 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, texts, labels) in enumerate(progress_bar):
            images = images.to(device)
            texts = texts.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)
            # 손실 계산
            loss = contrastive_loss(logits_per_image, logits_per_text)

            # Backward pass
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{save_path}/best_model.pth')
            print(f'새로운 최고 성능 모델 저장: {avg_loss:.4f}')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss
        }, f'{save_path}/checkpoint_epoch_{epoch+1}.pth')
        
    # 최종 모델 저장
    torch.save(model.state_dict(), f'{save_path}/final_model.pth')
    print(f'훈련 완료. 모델이 {save_path}에 저장되었습니다.')

    return model

def evaluate_zero_shot_classification(model, val_dataloader):
    model.eval() # Set model to evaluation mode

    val_class_names = val_dataloader.dataset.get_class_names()
    text_templates = val_dataloader.dataset.text_templates
    
    # Generate text embeddings for all unique class names in the validation set
    class_prompts = []
    for class_name in val_class_names:
        for template in text_templates:
            class_prompts.append(template.format(class_name))

    print(f"Generating embeddings for {len(class_prompts)} class prompts (Total Classes: {len(val_class_names)}).")
    tokenized_class_prompts = clip.tokenize(class_prompts).to(device)

    # Encode all class prompts in batches to avoid OOM for very large prompt sets
    text_features_list = []
    text_batch_size = 128 # Can adjust based on GPU memory
    with torch.no_grad():
        for i in tqdm(range(0, tokenized_class_prompts.shape[0], text_batch_size), desc="Encoding text prompts"):
            batch_tokens = tokenized_class_prompts[i:i + text_batch_size]
            text_features_list.append(model.encode_text(batch_tokens))
    
    text_features = torch.cat(text_features_list, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True) # Normalize

    # Reshape text_features: [num_classes, num_templates, embed_dim]
    # And then average across templates to get [num_classes, embed_dim]
    num_templates = len(text_templates)
    num_classes = len(val_class_names)
    text_features_per_class = text_features.view(num_classes, num_templates, -1).mean(dim=1)
    text_features_per_class = text_features_per_class / text_features_per_class.norm(dim=-1, keepdim=True) # Normalize again after averaging

    print(f"Starting zero-shot classification evaluation on {len(val_dataloader.dataset)} images...")
    
    all_true_labels = []
    all_predicted_labels = []
    total_eval_loss = 0.0
    start_time = time.time()
    
    # Monitor GPU memory usage
    initial_gpu_memory = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated(device) / (1024**2) # in MB
        print(f"Initial GPU Memory Usage: {initial_gpu_memory:.2f} MB")

    with torch.no_grad():
        for images, _, labels, in tqdm(val_dataloader, desc="Evaluating images"): 
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits = (image_features @ text_features_per_class.T) * logit_scale

            predicted_class_indices = logits.argmax(dim=-1)

            batch_loss = F.cross_entropy(logits, labels)

            total_eval_loss += batch_loss.item() * images.size(0) # Multiply by batch size for weighted average

            # true_class_names = [val_class_names[idx.item()] for idx in labels]
            # predicted_class_names = [val_class_names[idx.item()] for idx in predicted_class_indices]
            all_true_labels.extend(labels.cpu())
            all_predicted_labels.extend(predicted_class_indices.cpu())
    
    end_time = time.time()
    
    # Calculate throughput
    total_evaluation_time = end_time - start_time
    total_samples = len(val_dataloader.dataset)
    avg_eval_loss = total_eval_loss / total_samples if total_samples > 0 else 0
    throughput = total_samples / total_evaluation_time if total_evaluation_time > 0 else 0

    # Calculate final GPU memory usage
    final_gpu_memory = 0
    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_allocated(device) / (1024**2) # in MB

    print(f"\n--- Evaluation Summary ---")
    print(f"Total Samples Evaluated: {total_samples}")
    print(f"Total Evaluation Time: {total_evaluation_time:.2f} seconds")
    print(f"Average Evaluation Loss: {avg_eval_loss:.4f}")
    print(f"Throughput: {throughput:.2f} samples/second")
    print(f"Max GPU Memory Usage During Evaluation: {final_gpu_memory:.2f} MB (after encoding images)")
    
    # Classification Report (Accuracy, Precision, Recall, F1-score)

    report = classification_report(all_true_labels, all_predicted_labels, labels=list(range(len(val_class_names))), target_names=val_class_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    # ImageNet-mini 데이터 경로 설정
    data_path = "./imagenet-mini"  # 실제 경로로 변경
    words_txt_path = "./words.txt"  # words.txt 파일 경로
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-B/32", "nvidia/MambaVision-B-21K", device=device)
    # model, preprocess = clip.load("ViT-B/32", "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k", device=device)
    model = model.to(device)

    # 데이터로더 생성
    train_dataloader, val_dataloader = create_imagenet_mini_clip_dataloaders(
        data_path=data_path,
        words_txt_path=words_txt_path,
        preprocess=preprocess,
        batch_size=32,
        num_workers=4
    )
    
    print(f"훈련 배치 수: {len(train_dataloader)}")
    print(f"검증 배치 수: {len(val_dataloader)}")
    
    # 샘플 데이터 확인
    sample_batch = next(iter(train_dataloader))
    images, texts, labels = sample_batch
    print(f"이미지 배치 크기: {images.shape}")
    print(f"텍스트 배치 크기: {texts.shape}")
    print(f"라벨 배치 크기: {labels.shape}")
    
    #################### Train ####################
    # train_clip(model, train_dataloader, num_epochs=20, save_path="./checkpoints/CLIP")
    ###############################################

    ##################### Eval #####################
    # model.load_state_dict(torch.load('./checkpoints/CLIP/best_model.pth'))
    evaluate_zero_shot_classification(model, val_dataloader)
    ################################################