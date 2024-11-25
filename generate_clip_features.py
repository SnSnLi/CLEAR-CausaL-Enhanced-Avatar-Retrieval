import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import sys

def generate_clip_features():
    try:
        # 设置绝对路径
        base_dir = "/root/onethingai-tmp/avatar"
        images_dir = os.path.join(base_dir, "data/flickr30k_entities/raw/flickr30k-images")
        output_dir = os.path.join(base_dir, "emb/flickr30k_entities/openai/clip-vit-large-patch14/image")
        
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载CLIP模型
        print("Loading CLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-L/14", device=device)
        
        # 获取所有图片
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} images")
        
        # 处理每张图片
        for image_file in tqdm(image_files):
            image_id = os.path.splitext(image_file)[0]
            output_path = os.path.join(output_dir, f"{image_id}.pt")
            
            # 如果特征文件已存在，跳过
            if os.path.exists(output_path):
                continue
                
            try:
                # 加载和预处理图片
                image_path = os.path.join(images_dir, image_file)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                
                # 生成特征
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    
                # 保存特征
                torch.save(image_features.cpu(), output_path)
                
            except Exception as e:
                print(f"\nError processing {image_file}:")
                print(str(e))
                continue
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Starting feature generation...")
    generate_clip_features()
    print("Done!")