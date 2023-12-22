import os
from tqdm import tqdm
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

def main():
    workspace_dir = os.path.dirname(cur_dir)
    original_data_dir = os.path.join(workspace_dir, "data_test")
    # Apply data augmentation only to the train split
    original_data_dir = os.path.join(original_data_dir, "train")

    original_img_list = os.listdir(original_data_dir)

    # Iterate over original images and apply data augmentation
    for img_name in tqdm(original_img_list):
        img_path = os.path.join(original_data_dir, img_name)
        img = Image.open(img_path)

        # Randomly select a data augmentation operation
        augmentation = random.choice(["brightness", "crop", "rotate"])

        # Apply the selected data augmentation operation
        if augmentation == "brightness":
            img = enhance_brightness(img)
        elif augmentation == "crop":
            img = random_crop(img)
        elif augmentation == "rotate":
            img = random_rotate(img)

        img_id = int(img_name.split(".")[0])
        agumented_img_name = str(img_id + 300000) + ".jpg"
        # Save the augmented image to the augmented train directory
        augmented_img_path = os.path.join(original_data_dir, agumented_img_name)
        img.save(augmented_img_path)
    
    for img_name in tqdm(original_img_list):
        img_path = os.path.join(original_data_dir, img_name)
        img = Image.open(img_path)

        # Randomly select a data augmentation operation
        augmentation = random.choice(["noise", "contrast", "flip"])

        # Apply the selected data augmentation operation
        if augmentation == "noise":
            img = add_noise(img)
        elif augmentation == "contrast":
            img = enhance_contrast(img)
        elif augmentation == "flip":
            img = random_flip(img)

        img_id = int(img_name.split(".")[0])
        agumented_img_name = str(img_id + 600000) + ".jpg"
        # Save the augmented image to the augmented train directory
        augmented_img_path = os.path.join(original_data_dir, agumented_img_name)
        img.save(augmented_img_path)

    print("Data augmentation completed.")

def enhance_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(random.uniform(0.5, 1.5))
    return enhanced_image

def random_crop(image):
    width, height = image.size
    crop_size = min(width, height) * 8 // 9
    left = random.randint(0, width - crop_size)
    upper = random.randint(0, height - crop_size)
    right = left + crop_size
    lower = upper + crop_size
    cropped_image = image.crop((left, upper, right, lower))
    return cropped_image

def random_rotate(image):
    angle = random.choice([-30, -15, 15, 30])
    rotated_image = image.rotate(angle)
    return rotated_image

def add_noise(image):
    noisy_image = image.filter(ImageFilter.GaussianBlur(radius=2))
    return noisy_image

def enhance_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(random.uniform(0.7, 1.3))
    return enhanced_image

def random_flip(image):
    flip_type = random.choice(["horizontal", "vertical"])
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT) if flip_type == "horizontal" else image.transpose(Image.FLIP_TOP_BOTTOM)
    return flipped_image

if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    main()