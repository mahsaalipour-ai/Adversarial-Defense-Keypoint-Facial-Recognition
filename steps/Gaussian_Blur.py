import cv2
import os

# مسیر فایل ورودی و خروجی
img_path = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\data\step3\000001_adv_step3.png"
output_folder = r"D:\PayanNameh\paper-structure\deep-keypoints-attack\data\Gaussian_Blur"  # پوشه خروجی

# بررسی اینکه آیا پوشه خروجی وجود دارد یا خیر، در صورتی که وجود نداشت، ساخته شود
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# نام فایل خروجی
out_path = os.path.join(output_folder, "blurred_image.png")

# بارگذاری تصویر
img = cv2.imread(img_path)

# اعمال فیلتر Gaussian Blur
blurred = cv2.GaussianBlur(img, (31, 31), 0)

# ذخیره تصویر خروجی
cv2.imwrite(out_path, blurred)
print("Saved:", out_path)
