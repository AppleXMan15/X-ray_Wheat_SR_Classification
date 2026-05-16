import os
import glob

# находим все изображения grain*
files = glob.glob('grain_*.png')

# --- ШАГ 0: убираем "— копия" из имен ---
for file in files:
    if 'копия' in file:
        new_name = file.replace(' — копия', '').replace(' копия', '')

        # чтобы не перезаписать существующий файл — даем временное имя
        temp_name = 'clean_' + new_name
        os.rename(file, temp_name)

# обновляем список после очистки
files = glob.glob('grain*.png') + glob.glob('clean_grain*.png')

# --- ШАГ 1: приводим все к одному виду grain_X.png ---
cleaned_files = []
for file in files:
    if file.startswith('clean_'):
        new_name = file.replace('clean_', '')
        os.rename(file, new_name)
        cleaned_files.append(new_name)
    else:
        cleaned_files.append(file)

# --- ШАГ 2: сортируем по номеру ---
files = sorted(cleaned_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

# --- ШАГ 3: временные имена ---
for i, file in enumerate(files):
    os.rename(file, f'temp_{i}.png')

# --- ШАГ 4: финальные имена ---
temp_files = sorted(glob.glob('temp_*.png'))

for i, file in enumerate(temp_files, start=3151):
    os.rename(file, f'grain_{i:04d}.png')  # с нулями

print("Готово ✅")
