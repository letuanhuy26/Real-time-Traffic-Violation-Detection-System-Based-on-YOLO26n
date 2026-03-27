from PIL import Image, ImageDraw, ImageFont

def create_double_clahe_comparison():
    # 1. Tên file tương ứng với quy trình của bạn
    files = [r"C:\Users\Admin\Downloads\a.png", 
    r"C:\Users\Admin\Downloads\b.png",
    r"C:\Users\Admin\Downloads\c.png", 
    r"C:\Users\Admin\Downloads\d.png"]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    try:
        images = [Image.open(f) for f in files]
        w, h = 800, 600
        resized = [img.resize((w, h), Image.Resampling.LANCZOS) for img in images]

        # Tạo canvas 2x2
        padding = 15
        canvas = Image.new('RGB', (w*2 + padding, h*2 + padding), (255, 255, 255))
        
        positions = [(0, 0), (w + padding, 0), (0, h + padding), (w + padding, h + padding)]
        for img, pos in zip(resized, positions):
            canvas.paste(img, pos)

        # Chèn nhãn
        draw = ImageDraw.Draw(canvas)
        # Sử dụng font mặc định nếu không tìm thấy Arial
        for label, pos in zip(labels, positions):
            draw.text((pos[0] + 10, pos[1] + 10), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0,0,0))

        canvas.save("case_study_id13.png")
        print("Đã tạo xong hình Case Study với quy trình Double-CLAHE!")
    except Exception as e:
        print(f"Lỗi: {e}")

create_double_clahe_comparison()