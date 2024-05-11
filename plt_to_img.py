import cv2
import os

def plt_to_img(image, file_path, file_name):
    # 파일 경로
    file_path = os.path.join(file_path, file_name)
    
    # 이미지 저장
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    print(f"이미지가 저장되었습니다: {file_path}")
