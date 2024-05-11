# 자동차 파손 영역 검사 및 가격 산출 AI 만들기
# Model : [AI허브](https://www.aihub.or.kr), [쏘카](https://www.socar.kr)

# Model 소개

'''
- [DAMAGE][Breakage_3]Unet.pt : 파손
- [DAMAGE][Crushed_2]Unet.pt : 찌그러짐
- [DAMAGE][Scratch_0]Unet.pt : 스크래치
- [DAMAGE][Seperated_1]Unet.pt : 이격

'''

# 사전 학습 모델

'''
https://drive.google.com/drive/folders/1q0l5vT14Kka_iu0WZgn1EFJLUbWD8EtY?usp=sharing

'''

# 데이터셋

'''
[차량 파손 이미지 - AI허브](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=581)

'''

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.Models import Unet
from plt_to_img import plt_to_img

## 1. Model load 
## 테스트용으로 2번 모델만 실험한다.

# weight_path = 'models/[DAMAGE][Breakage_3]Unet.pt' # 0번 : 파손
# weight_path = 'models/[DAMAGE][Crushed_2]Unet.pt' # 1번 : 찌그러짐
weight_path = 'models/[DAMAGE][Scratch_0]Unet.pt' # 2번 : 스크레치
# weight_path = 'models/[DAMAGE][Seperated_1]Unet.pt' #3번 : 이격



n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
model.eval()

print('모델 로드 완료!')


## 2. 이미지 로드

img_path = 'C:/sinheechan.github.io-master/Car_Scratch_MLops/samples/damage/0001061_sc-123724.jpg' # Scratch 이미지

img  = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

plt.figure(figsize=(8, 8))
plt.imshow(img)

# 이미지 저장 : plt_to_img.py

plt_path = 'C:/sinheechan.github.io-master/Car_Scratch_MLops/test_result_folder'
plt_name = 'img_row.jpg'

plt.savefig(os.path.join(plt_path, plt_name))
# plt.show()

# 이미지 전처리

img_input = img / 255.
img_input = img_input.transpose([2, 0, 1]) # Numpy 차원 순서 주의 : H, W, C => 채널정보를 앞으로 이동
img_input = torch.tensor(img_input).float().to(device)
img_input = img_input.unsqueeze(0)

print(img_input.shape)

# 모델 추론

output = model(img_input)

print(output.shape)


# 후처리 및 추론 마스크 시각화

img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
img_output = img_output.transpose([1, 2, 0])

plt.figure(figsize=(8, 8))
plt.imshow(img_output)

plt_path = 'C:/sinheechan.github.io-master/Car_Scratch_MLops/test_result_folder'
plt_name = 'img_mask.jpg'
plt.savefig(os.path.join(plt_path, plt_name))
# plt.show()

# 결과

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

ax[0].imshow(img)
ax[0].set_title('image')
ax[0].axis('off')

ax[1].imshow(img.astype('uint8'), alpha=0.5)
ax[1].imshow(img_output, cmap='jet', alpha=0.5)
ax[1].set_title('output')
ax[1].axis('off')

fig.set_tight_layout(True)

# 결과 이미지 저장

plt_path = 'C:/sinheechan.github.io-master/Car_Scratch_MLops/test_result_folder'
plt_name = 'img_result.jpg'
plt.savefig(os.path.join(plt_path, plt_name))
# plt.show()


# 결과 이미지파일 대조 결과 스크레치에 대한 인식이 정상 작동함을 확인하였다.
# 아래부터 여러 형태의 파손 감지를 진행하는 모델을 생성

# 여러 형태의 파손 영역 감지 모델 생성

labels = ['Breakage_3', 'Crushed_2', 'Scratch_0', 'Seperated_1']
models = []

n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for label in labels:
    model_path = f'models/[DAMAGE][{label}]Unet.pt'

    model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device) # resnet34 / imagenet
    model.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    models.append(model)

print('Unet 모델 로드가 완료되었습니다.')

# 원본 이미지파일 로드

img_path = 'C:/sinheechan.github.io-master/Car_Scratch_MLops/samples/damage/0065842_sc-192098.jpg'

img  = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

plt.figure(figsize=(8, 8))

plt.imshow(img)

plt_path = 'C:/sinheechan.github.io-master/Car_Scratch_MLops/test_multi_result_folder'
plt_name = 'img_row.jpg'
plt.savefig(os.path.join(plt_path, plt_name))
# plt.show()

# 이미지 전처리

img_input = img / 255. # 정규화
img_input = img_input.transpose([2, 0, 1])
img_input = torch.tensor(img_input).float().to(device)
img_input = img_input.unsqueeze(0)

fig, ax = plt.subplots(1, 5, figsize=(24, 10))

ax[0].imshow(img)
ax[0].axis('off')

outputs = [] # 이미지가 저장될 리스트 생성

for i, model in enumerate(models):
    output = model(img_input)

    img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
    img_output = img_output.transpose([1, 2, 0])

    outputs.append(img_output)

    ax[i+1].set_title(labels[i])
    ax[i+1].imshow(img_output, cmap='jet')
    ax[i+1].axis('off')


fig.set_tight_layout(True)

plt_path = 'C:/sinheechan.github.io-master/Car_Scratch_MLops/test_multi_result_folder'
plt_name = 'img_result.jpg'
plt.savefig(os.path.join(plt_path, plt_name))
# plt.show()

# 파손 영역 크기 계산
# 예시 모델은 1픽셀당 가격을 책정하여 영역이 넓을 수록 수리비용이 비싸지도록 가격을 산정

for i, label in enumerate(labels):
    print(f'{label}: {outputs[i].sum()}')

price_table = [
    100, # Breakage_3 : 파손
    150, # Crushed_2 : 찌그러짐
    50,  # Scratch_0 : 스크레치
    120, # Seperated_1 : 이격
]

total = 0

for i, price in enumerate(price_table):
    input_area = outputs[i].sum()
    total += input_area * price

    print(f'{labels[i]}:\t영역: {input_area}\t가격:{input_area * price}원')

print(f'고객님, 올리신 이미지의 총 수리비는 {total}원 입니다.')

