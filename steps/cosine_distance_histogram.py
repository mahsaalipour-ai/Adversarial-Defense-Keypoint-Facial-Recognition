import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# لود تصویر (نام فایل رو تغییر بده اگر لازم بود)
img = Image.open('figure.png')  # یا مسیر کامل فایل
img_array = np.array(img)

# ابعاد تقریبی بر اساس تصویر (می‌تونی دقیق‌تر تنظیم کنی)
height, width, _ = img_array.shape

# موقعیت تقریبی heatmaps بالایی (ستون‌های activation maps با colormap گرم/سرد)
# فرض: 8 heatmap در دو ردیف بالا، هر کدام حدود 120x120 پیکسل
heatmap_height = 120
heatmap_width = 120
margin_top = 20
margin_left = 50
spacing_x = 150
spacing_y = 140

# لیست برای ذخیره داده‌های original و attacked
original_hist_data = []
attacked_hist_data = []

# استخراج 8 heatmap (4 original + 4 attacked)
for row in range(2):  # دو ردیف بالا
    for col in range(4):  # چهار ستون
        x = margin_left + col * (heatmap_width + spacing_x)
        y = margin_top + row * (heatmap_height + spacing_y)
        
        heatmap = img_array[y:y+heatmap_height, x:x+heatmap_width]
        
        # تبدیل به grayscale (میانگین کانال‌ها، چون colormap رنگیه)
        gray = np.mean(heatmap, axis=2)
        
        # flatten برای هیستوگرام
        pixels = gray.flatten()
        
        if col % 2 == 0:  # ستون‌های زوج: Original (a)
            original_hist_data.append(pixels)
        else:  # ستون‌های فرد: DKA² attack (b)
            attacked_hist_data.append(pixels)

# محاسبه هیستوگرام میانگین برای هر گروه
bins = np.linspace(0, 255, 50)  # 50 بین

orig_hist, _ = np.histogram(np.concatenate(original_hist_data), bins=bins, density=True)
attack_hist, _ = np.histogram(np.concatenate(attacked_hist_data), bins=bins, density=True)

# پلات هیستوگرام مقایسه‌ای
plt.figure(figsize=(10, 6))
plt.plot(bins[:-1], orig_hist, label='Before Attack (Original)', color='blue', linewidth=2)
plt.plot(bins[:-1], attack_hist, label='After DKA² Attack', color='red', linewidth=2)
plt.fill_between(bins[:-1], orig_hist, alpha=0.3, color='blue')
plt.fill_between(bins[:-1], attack_hist, alpha=0.3, color='red')

plt.title('Histogram of Activation Map Intensities: Before vs After DKA² Attack')
plt.xlabel('Pixel Intensity Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)

# ذخیره هیستوگرام به عنوان تصویر PNG
plt.savefig('activation_map_histogram.png')

plt.show()


#python cosine_distance_histogram.py
