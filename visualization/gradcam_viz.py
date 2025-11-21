import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANALYSIS_MEAN = torch.tensor([0.456]).view(1,1,1).to(DEVICE)
ANALYSIS_STD  = torch.tensor([0.224]).view(1,1,1).to(DEVICE)

CLASSES = {0:'Non Demented',1:'Very mild Dementia',2:'Mild Dementia'}

# ----------------------------------------------------------
# 1) Classification Model
# ----------------------------------------------------------
class PretrainedClassifier(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=3, in_chans=1):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans
        )

    def forward(self, x):
        return self.model(x)

def load_generated_image(path):
    img = Image.open(path).convert("L")
    img = transforms.ToTensor()(img)   # shape [1,224,224]
    return img

def load_classifier(model_path):
    model = PretrainedClassifier().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ----------------------------------------------------------
# 2) Grad-CAM
# ----------------------------------------------------------
feature_map = None

def forward_hook(module, input, output):
    global feature_map
    feature_map = output
    feature_map.retain_grad()

def get_gradcam(model, img_norm, target_class):
    global feature_map
    feature_map = None

    target_layer = model.model.layer4
    hook = target_layer.register_forward_hook(forward_hook)

    out = model(img_norm.unsqueeze(0))

    if target_class is None:
        target_class = out.argmax(1).item()

    model.zero_grad()
    out[0, target_class].backward()

    hook.remove()

    fmap = feature_map.detach().cpu().numpy()[0]
    grad = feature_map.grad.detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1,2))

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights):
        cam += w * fmap[c]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_norm.size(2), img_norm.size(1)))
    cam -= cam.min()
    cam /= cam.max() + 1e-7
    return cam


# ----------------------------------------------------------
# 3) Visualization
# ----------------------------------------------------------
def visualize_gradcam(original_img_01, cam, pred_name, save_path):
    img = original_img_01.squeeze().cpu().numpy()

    heatmap = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1] / 255.0

    img3 = np.stack([img, img, img], axis=-1)
    overlay = 0.4*heatmap + 0.6*img3
    overlay = np.uint8(overlay*255)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Generated Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM ({pred_name})")
    plt.axis("off")   

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Grad-CAM image: {save_path}")


# ----------------------------------------------------------
# 4) Main Runner (생성 이미지 전용)
# ----------------------------------------------------------
def run_gradcam_on_generated_image(gen_img_tensor, classifier_path):

    # load classifier
    clf = load_classifier(classifier_path)

    # normalize
    img_norm = (gen_img_tensor.to(DEVICE) - ANALYSIS_MEAN) / ANALYSIS_STD

    # classification
    with torch.no_grad():
        logits = clf(img_norm.unsqueeze(0))
        pred_id = logits.argmax(1).item()
        pred_name = CLASSES[pred_id]

    print(f"Predicted class: {pred_name}")

    # compute grad-cam
    cam = get_gradcam(clf, img_norm, pred_id)

    # save visualization
    visualize_gradcam(
        original_img_01=gen_img_tensor.cpu(),
        cam=cam,
        pred_name=pred_name,
        save_path=f"gradcam_generated_{pred_name}.png"
    )


# ----------------------------------------------------------
# 5) 🔥 실제 호출
# ----------------------------------------------------------
# CVAE에서 생성한 이미지를 gen_img 라고 가정
# gen_img shape: [1, 224, 224]

# 예)
# gen,_,_,_ = model(x0, proto)
# gen_img = gen.squeeze(0)

# 🔥 실제 실행 코드
gen_img = load_generated_image("GEN_SAMPLES/from_Non Demented_to_Very mild Dementia_3965.png")

run_gradcam_on_generated_image(
    gen_img_tensor = gen_img, 
    classifier_path = "classification_results/best_classifier_resnet18_weights_42.pth"
)
