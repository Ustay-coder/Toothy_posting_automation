from PIL import Image, ImageDraw, ImageFont
import os
import json

def print_ingredients_by_csr_grade(toothpaste_data):
    # 등급별로 분류할 딕셔너리 초기화
    grade_dict = {
        'A+': [],
        'A': [],
        'B+': [],
        'B': [],
        'C+': [],
        'C': [],
        'D': [],
        'E': [],
        'F': []
    }
    # 각 성분을 등급별로 분류
    for ingredient in toothpaste_data.get('ingredients', []):
        csr = ingredient.get('oristal', {}).get('csr', None)
        name = ingredient.get('name', '')
        if csr and csr in grade_dict:
            grade_dict[csr].append(name)
        else:
            grade_dict['F'].append(name)  # 등급이 없으면 F로 분류

    # 등급별로 출력 (이미지 예시 스타일)
    for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'E', 'F']:
        items = grade_dict[grade]
        if items:
            print(f"\033[1m\033[38;2;0;200;100m{grade}\033[0m {len(items)}개\033[0m" if grade.startswith('A') else (f"\033[1m\033[38;2;0;100;255m{grade}\033[0m {len(items)}개\033[0m" if grade.startswith('B') else f"{grade} {len(items)}개"))
            # 성분 리스트 전체 출력
            print(', '.join(items))
            print()

def wrap_text_by_pixel(draw, text, font, max_width):
    words = text.split(', ')
    lines = []
    current_line = ''
    for word in words:
        test_line = word if not current_line else current_line + ', ' + word
        width = draw.textlength(test_line, font=font)
        if width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def save_ingredients_by_csr_grade_image(toothpaste_data, config_path='query/config.json'):
    # config에서 ITEM_NAME 읽기
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    item_name = config.get('ITEM_NAME', 'default')

    # 디렉토리 생성
    save_dir = f'ingredients/{item_name}'
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'ingredients_by_grade.png')

    grade_dict = {
        'A+': [], 'A': [], 'B+': [], 'B': [], 'C+': [], 'C': [], 'D': [], 'E': [], 'F': []
    }
    for ingredient in toothpaste_data.get('ingredients', []):
        csr = ingredient.get('oristal', {}).get('csr', None)
        name = ingredient.get('name', '')
        if csr and csr in grade_dict:
            grade_dict[csr].append(name)
        else:
            grade_dict['F'].append(name)

    width = 800
    padding = 30
    line_height = 36
    font_size = 28
    title_font_size = 36
    try:
        font = ImageFont.truetype("AppleGothic.ttf", font_size)
        title_font = ImageFont.truetype("AppleGothic.ttf", title_font_size)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # 임시 draw 객체로 줄바꿈 계산
    temp_img = Image.new('RGB', (width, 100), color='white')
    temp_draw = ImageDraw.Draw(temp_img)

    lines = []
    total_lines = 0
    max_text_width = width - padding * 2 - 20
    for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'E', 'F']:
        items = grade_dict[grade]
        if items:
            names = ', '.join(items)
            wrapped = wrap_text_by_pixel(temp_draw, names, font, max_text_width)
            lines.append((grade, f"{len(items)}개", wrapped))
            total_lines += 1 + len(wrapped)

    height = padding * 2 + total_lines * line_height
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    y = padding
    for grade, count, wrapped_names in lines:
        if grade.startswith('A'):
            color = (0, 200, 100)
        elif grade.startswith('B'):
            color = (0, 100, 255)
        else:
            color = (80, 80, 80)
        draw.text((padding, y), f"{grade} {count}", font=title_font, fill=color)
        y += line_height
        for line in wrapped_names:
            draw.text((padding + 20, y), line, font=font, fill=(60, 60, 60))
            y += line_height
    img.save(filename)
    print(f"이미지로 저장 완료: {filename}")