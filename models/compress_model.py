
import joblib
import os

# 1. Dapatkan lokasi folder tempat script ini berada (folder 'models')
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Definisikan path file Input dan Output secara absolut
input_file = os.path.join(current_dir, "champion_single_model.pkl")
output_file = os.path.join(current_dir, "champion_single_model_compressed.pkl")

print(f"Working Directory: {current_dir}")
print(f"Reading: {input_file}")

# 3. Cek apakah file ada sebelum meload
if not os.path.exists(input_file):
    print("ERROR: File tidak ditemukan! Pastikan nama file benar.")
    exit()

# 4. Load Model
print("⏳ Loading model besar...")
model = joblib.load(input_file)

# 5. Compress & Save
print("📦 Compressing & Saving...")
joblib.dump(model, output_file, compress=3)

# 6. Cek Ukuran
old_size = os.path.getsize(input_file) / (1024 * 1024)
new_size = os.path.getsize(output_file) / (1024 * 1024)

print(f"SUKSES! Ukuran berkurang dari {old_size:.2f} MB -> {new_size:.2f} MB")