from PIL import Image

# Mở ảnh
image = Image.open("backgr.jpg")

# Thay đổi kích thước
image_resized = image.resize((1200, 630))

# Lưu ảnh đã thay đổi kích thước
image_resized.save("background.jpg")
