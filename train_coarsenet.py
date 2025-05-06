import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.coarse_net import CoarseNet
from utils.flow import compute_optical_flow
from tqdm import tqdm

class SyntheticShakyDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for f in sorted(files):
                if f.endswith(".jpg") or f.endswith(".png"):
                    self.image_paths.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        clean_img = cv2.imread(self.image_paths[idx])
        clean_img = cv2.resize(clean_img, (256, 256))
        h, w = clean_img.shape[:2]

        # Create synthetic jitter
        dx = np.random.uniform(-10, 10)
        dy = np.random.uniform(-10, 10)
        angle = np.random.uniform(-5, 5)

        M_trans = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        M_trans[0, 2] += dx
        M_trans[1, 2] += dy

        jittered_img = cv2.warpAffine(clean_img, M_trans, (w, h), flags=cv2.INTER_LINEAR)
        flow = compute_optical_flow(jittered_img, clean_img)

        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()
        target = torch.tensor([-dx/256.0, -dy/256.0, -angle/10.0], dtype=torch.float32)  # Inverse

        return flow_tensor, target

def train_coarsenet(image_dir, output_path='coarsenet_trained.pth', epochs=20, batch_size=8):
    dataset = SyntheticShakyDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CoarseNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for flows, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            flows, targets = flows.cuda(), targets.cuda()
            preds = model(flows)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), output_path)
    print(f"✅ CoarseNet model saved to {output_path}")

if __name__ == '__main__':
    train_coarsenet(image_dir='data/DAVIS/JPEGImages/480p/', epochs=20)

