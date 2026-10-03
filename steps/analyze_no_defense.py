import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

# مسیر embeddingها
emb_before_path = "embeddings_before_attack.npy"
emb_after_path  = "embeddings_after_attack.npy"

# بارگذاری
emb_before = np.load(emb_before_path)
emb_after  = np.load(emb_after_path)

# اگر تک‌نمونه‌ای بود، شکل بده
if emb_before.ndim == 1:
    emb_before = emb_before.reshape(1, -1)
    emb_after  = emb_after.reshape(1, -1)

# محاسبه فاصله کسینوسی
cos_dist = np.diag(cosine_distances(emb_before, emb_after))

# ذخیره فاصله‌ها
np.save("cosine_distance_no_defense.npy", cos_dist)

print("Mean Cosine Distance:", cos_dist.mean())
print("Minimum distance:", cos_dist.min())
print("Maximum distance:", cos_dist.max())

plt.figure()
plt.hist(cos_dist, bins=30)
plt.xlabel("Cosine Distance")
plt.ylabel("Number of Samples")
plt.title("Cosine Distance Histogram Before and After Attack (No Defense)")
plt.show()

